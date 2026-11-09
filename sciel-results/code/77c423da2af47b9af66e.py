"""Soft-min smoothed L-BFGS-B optimization for 300 equal circles in a unit circle."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import cKDTree

N = 300
# EVOLVE-BLOCK-START


def _raw_radius(pts):
    """True packing radius (min of pair distances /2 and boundary margin)."""
    norms = np.linalg.norm(pts, axis=1)
    boundary = 1.0 - np.max(norms)
    if boundary <= 0:
        return 0.0
    tree = cKDTree(pts)
    dists, _ = tree.query(pts, k=2)
    min_pair = float(np.min(dists[:, 1])) / 2.0
    return min(boundary, min_pair)


def _soft_min(vals, beta):
    """Log-sum-exp based soft minimum: approximates min(vals) smoothly."""
    v = np.asarray(vals, dtype=np.float64)
    m = np.min(v)
    return m - np.logaddexp.reduce(np.log(np.exp(beta * (v - m)))) / beta


def _smooth_objective_and_grad(x, beta):
    """
    Objective: -r_soft where r_soft = soft_min(pair_dists/2, boundary_margins).
    Gradient: analytic via chain rule on the soft-min weights.
    x: shape (N, 2)
    """
    pts = x.reshape(N, 2)
    # Pair distances: compute all pairwise distances (N=300 -> 44850 pairs)
    # Use efficient broadcasting for the upper triangle
    # d[i,j] = |pts[i] - pts[j]|
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    # Mask diagonal
    np.fill_diagonal(dists, np.inf)
    # Get upper triangle indices
    iu, ju = np.triu_indices(N, k=1)
    pair_dists = dists[iu, ju]  # shape (M,)
    half_dists = pair_dists / 2.0

    # Boundary margins: 1 - |pts[i]|
    norms = np.linalg.norm(pts, axis=1)
    boundary_margins = 1.0 - norms  # shape (N,)

    # Combine all "constraints" into one vector
    all_vals = np.concatenate([half_dists, boundary_margins])  # shape (M + N,)

    # Soft min
    m = np.min(all_vals)
    # exp(beta * (v - m)) for numerical stability
    exps = np.exp(beta * (all_vals - m))
    s = np.sum(exps)
    r_soft = m - np.log(s) / beta

    # Weights for soft min: w_k = exp(beta*(v_k - m)) / s
    weights = exps / s  # shape (M + N,)

    # Gradient of r_soft w.r.t. x
    # dr_soft/dx = -sum_k w_k * (beta * (v_k - m) * dv_k/dx - dv_k/dx) / s ... 
    # Actually: r_soft = m - (1/beta)*log(s)
    # dr_soft/dx = dm/dx - (1/beta) * (1/s) * ds/dx
    # dm/dx = 0 (m is a constant w.r.t. x in the gradient sense? No, m = min(all_vals))
    # Actually m depends on x too, but in the log-sum-exp formula:
    # r_soft = - (1/beta) * log(sum(exp(beta * v_k)))  + m  ... wait let me redo.
    # Standard: soft_min(v) = - (1/beta) * log(sum(exp(-beta * v_k)))
    # Let me use that form for cleaner gradients.
    # soft_min = -(1/beta) * log(sum(exp(-beta * v_k)))
    # d(soft_min)/dx = (1/beta) * [sum(exp(-beta*v_k) * (-beta) * dv_k/dx)] / sum(exp(-beta*v_k))
    #               = -sum(w_k * dv_k/dx) where w_k = exp(-beta*v_k)/sum(exp(-beta*v_j))
    
    # Let me recompute with the negative form for clarity:
    neg_beta_vals = -beta * all_vals
    neg_beta_vals -= np.max(neg_beta_vals)  # for stability
    exps_neg = np.exp(neg_beta_vals)
    s_neg = np.sum(exps_neg)
    r_soft2 = -(1.0 / beta) * (np.log(s_neg) - np.max(neg_beta_vals))
    # Actually simpler: r_soft2 = - (1/beta) * log(sum(exp(-beta*v_k)))
    # For stability: let c = max(-beta*v_k), then sum(exp(-beta*v_k)) = exp(c) * sum(exp(-beta*v_k - c))
    # log(sum(exp(-beta*v_k))) = c + log(sum(exp(-beta*v_k - c)))
    # r_soft = -(1/beta) * (c + log(sum(exp(-beta*v_k - c))))
    
    # Let me just use the first formulation and be careful:
    # r_soft = m - (1/beta) * log(sum(exp(beta*(v_k - m))))
    # This is equivalent to the standard soft-min.
    # Gradient:
    # d r_soft / d x_i = - (1/beta) * d/dx_i [log(sum_k exp(beta*(v_k - m)))]
    #                  = - (1/beta) * [sum_k exp(beta*(v_k-m)) * beta * (dv_k/dx_i - dm/dx_i)] / sum_k exp(beta*(v_k-m))
    # Since m = min(v_k), dm/dx_i = dv_{argmin}/dx_i, but in the soft formulation we treat m as the minimum value.
    # Actually, a cleaner way: use the identity that soft_min is differentiable and
    # grad(soft_min) = -sum_k w_k * grad(v_k) where w_k = softmax(-beta * v_k)
    # Let me verify: soft_min(v) = - (1/beta) log(sum exp(-beta v_k))
    # d(soft_min)/dv_j = (1/beta) * beta * exp(-beta v_j) / sum exp(-beta v_k) = exp(-beta v_j) / sum = w_j
    # Wait: d/dv_j [- (1/beta) log(sum_k exp(-beta v_k))]
    # = -(1/beta) * [1/sum] * sum_k d/dv_j exp(-beta v_k)
    # = -(1/beta) * [1/sum] * exp(-beta v_j) * (-beta)
    # = exp(-beta v_j) / sum = w_j
    # So grad_{v_j} soft_min = w_j where w_j = exp(-beta v_j) / sum_k exp(-beta v_k)
    # And by chain rule: grad_x soft_min = sum_j w_j * grad_x v_j
    
    # So: grad_x r_soft = sum_j w_j * grad_x v_j
    # And our objective is f(x) = -r_soft, so grad f = -sum_j w_j * grad_x v_j

    # Recompute weights with the negative-beta form:
    # w_j = exp(-beta * v_j) / sum_k exp(-beta * v_k)
    # For numerical stability:
    shifted = -beta * all_vals
    shifted_max = np.max(shifted)
    shifted -= shifted_max
    exps2 = np.exp(shifted)
    s2 = np.sum(exps2)
    weights2 = exps2 / s2  # shape (M + N,)
    r_soft_val = -(1.0 / beta) * (np.log(s2) + shifted_max)  # = -(1/beta)*(log(s2) + shifted_max)
    # Check: log(sum exp(-beta*v_k)) = shifted_max + log(s2)
    # r_soft = -(1/beta) * (shifted_max + log(s2))

    M = len(pair_dists)
    grad = np.zeros_like(x)

    # Contribution from pair distances v_k = |p_i - p_j| / 2 for k in upper triangle
    # dv_k/dx_i = (p_i - p_j) / (2 * |p_i - p_j|) for the i-th coordinate of pair (i,j)
    # dv_k/dx_j = (p_j - p_i) / (2 * |p_i - p_j|)
    pair_weights = weights2[:M]  # shape (M,)
    # For each pair (i, j) with distance d, contribution to grad:
    # w_k * [ (p_i - p_j)/(2d) for x_i, (p_j - p_i)/(2d) for x_j ]
    # But we want grad f = -grad r_soft = -sum w_k * grad v_k
    # So contribution to grad[x_i] from pair (i,j): -w_k * (p_i - p_j) / (2 * d)
    # And to grad[x_j]: -w_k * (p_j - p_i) / (2 * d) = w_k * (p_i - p_j) / (2 * d)

    # Vectorized: for each pair k with indices (iu[k], ju[k]) and distance d_k = pair_dists[k]
    # dir_k = (pts[iu[k]] - pts[ju[k]]) / (2 * d_k)
    # grad[pts[iu[k]]] -= w_k * dir_k
    # grad[pts[ju[k]]] += w_k * dir_k

    # Avoid division by zero
    safe_dists = np.where(pair_dists > 1e-12, pair_dists, 1e12)
    diff_pairs = pts[iu] - pts[ju]  # shape (M, 2)
    dirs = diff_pairs / (2.0 * safe_dists[:, np.newaxis])  # shape (M, 2)
    w_pair = pair_weights  # shape (M,)

    # Accumulate gradients
    np.add.at(grad, iu, -w_pair[:, np.newaxis] * dirs)
    np.add.at(grad, ju, w_pair[:, np.newaxis] * dirs)

    # Contribution from boundary margins v_k = 1 - |p_k|
    # dv_k/dx_k = -p_k / |p_k|
    # grad f contribution: -w_k * (-p_k / |p_k|) = w_k * p_k / |p_k|
    bound_weights = weights2[M:]  # shape (N,)
    safe_norms = np.where(norms > 1e-12, norms, 1e12)
    grad += bound_weights[:, np.newaxis] * pts / safe_norms[:, np.newaxis]

    return -r_soft_val, grad


def _hex_lattice_init(spacing):
    """Generate hex lattice points inside unit disk, select N closest to center."""
    k = int(1.0 / spacing) + 5
    pts = []
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            if np.hypot(x, y) <= 1.0:
                pts.append((x, y))
    if len(pts) < N:
        return None
    pts = np.array(pts)
    dists = np.linalg.norm(pts, axis=1)
    idx = np.argsort(dists, kind="stable")[:N]
    return pts[idx]


def _concentric_rings_init(n_rings=8):
    """Generate points on concentric rings with hexagonal-like spacing."""
    pts = [[0.0, 0.0]]
    # Ring radii approximately at hex lattice distances
    for k in range(1, n_rings + 1):
        r = k * 0.055
        if r > 0.95:
            break
        # Number of points on this ring
        n_pts = max(6, int(2 * np.pi * r / 0.055))
        angles = 2 * np.pi * np.arange(n_pts) / n_pts + (np.pi / n_pts) * (k % 2)
        ring_pts = np.column_stack([r * np.cos(angles), r * np.sin(angles)])
        pts.append(ring_pts)
    all_pts = np.vstack(pts)
    if len(all_pts) < N:
        # Fill remaining on outer ring
        deficit = N - len(all_pts)
        r_outer = 0.92
        angles = 2 * np.pi * np.arange(deficit) / deficit
        extra = np.column_stack([r_outer * np.cos(angles), r_outer * np.sin(angles)])
        all_pts = np.vstack([all_pts, extra])
    else:
        all_pts = all_pts[:N]
    return all_pts


def _refine_boundary(pts, r_target, max_iters=200):
    """
    Newton-like polish: ensure all points are strictly inside and separated.
    Push points inward if they violate boundary, push apart if too close.
    """
    pts = pts.copy()
    for it in range(max_iters):
        # Check boundary
        norms = np.linalg.norm(pts, axis=1)
        boundary_viol = norms + r_target - 1.0
        if np.max(boundary_viol) < 0:
            # Check pair distances
            tree = cKDTree(pts)
            dists, idxs = tree.query(pts, k=2)
            min_pair = np.min(dists[:, 1])
            if min_pair >= 2 * r_target:
                break
            # Push apart the closest pairs
            for i in range(N):
                j = idxs[i, 1]
                if dists[i, 1] < 2 * r_target:
                    d = pts[i] - pts[j]
                    dn = np.linalg.norm(d)
                    if dn > 1e-12:
                        push = (2 * r_target - dists[i, 1]) * 0.5
                        pts[i] += push * d / dn
                        pts[j] -= push * d / dn
        else:
            # Push inward the most violated points
            for i in range(N):
                if boundary_viol[i] > 0:
                    d = pts[i] / (norms[i] + 1e-12)
                    pts[i] -= boundary_viol[i] * d
    return pts


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin

    # Generate multiple initial configurations
    inits = []

    # Hex lattice with several spacings around the expected optimal
    for spacing in [0.104, 0.106, 0.108, 0.110, 0.112, 0.114, 0.116]:
        pts = _hex_lattice_init(spacing)
        if pts is not None:
            inits.append(pts)

    # Concentric rings
    for n_rings in [7, 8, 9, 10]:
        pts = _concentric_rings_init(n_rings)
        if pts is not None:
            inits.append(pts)

    # Remove duplicates (by first few coordinates)
    unique_inits = []
    for pts in inits:
        key = tuple(pts[:3].flatten().round(4))
        if key not in [tuple(p[:3].flatten().round(4)) for p in unique_inits]:
            unique_inits.append(pts)

    best_pts = None
    best_r = 0.0

    # Sort inits by initial raw radius (best first)
    scored_inits = []
    for pts in unique_inits:
        r = _raw_radius(pts)
        scored_inits.append((pts, r))
    scored_inits.sort(key=lambda x: x[1], reverse=True)

    # Optimize top candidates with L-BFGS-B
    n_optimize = min(4, len(scored_inits))
    for cand_idx in range(n_optimize):
        if time.time() > deadline - 15.0:
            break
        pts0, r0 = scored_inits[cand_idx]
        x0 = pts0.flatten()

        # Use adaptive beta: start moderate, increase for more precision
        betas = [80.0, 200.0, 500.0]
        current_x = x0.copy()

        for beta in betas:
            if time.time() > deadline - 10.0:
                break
            try:
                res = minimize(
                    _smooth_objective_and_grad,
                    current_x,
                    args=(beta,),
                    method='L-BFGS-B',
                    jac=True,
                    options={'maxiter': 300, 'maxfun': 5000, 'ftol': 1e-12, 'gtol': 1e-8,
                             'maxls': 40, 'maxcor': 10}
                )
                current_x = res.x
            except Exception:
                break

        # Evaluate final result
        final_pts = current_x.reshape(N, 2)
        # Boundary polish
        r_est = _raw_radius(final_pts)
        final_pts = _refine_boundary(final_pts, r_est, max_iters=100)
        r_final = _raw_radius(final_pts)

        if r_final > best_r:
            best_r = r_final
            best_pts = final_pts

    # If no valid result, fallback
    if best_pts is None or best_r <= 0:
        angles = 2 * np.pi * np.arange(N) / N
        return np.column_stack([0.5 * np.cos(angles), 0.5 * np.sin(angles)])

    # Final boundary safety check
    norms = np.linalg.norm(best_pts, axis=1)
    if np.max(norms) > 1.0:
        best_pts = best_pts / (np.max(norms) + 1e-6) * (1.0 - 1e-6)

    return best_pts


# EVOLVE-BLOCK-END
