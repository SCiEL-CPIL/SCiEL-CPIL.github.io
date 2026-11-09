"""Combined program: Hex lattice init + SLSQP optimization with smooth objective and hard boundary constraints."""
import time
import numpy as np
from scipy.optimize import minimize

N = 300

# EVOLVE-BLOCK-START

def _radius(pts):
    """Return the common radius for a set of centres, or -inf if invalid."""
    pts = np.asarray(pts, dtype=float)
    if pts.shape != (N, 2):
        return -np.inf
    if not np.all(np.isfinite(pts)):
        return -np.inf
    # boundary term
    dist_origin = np.linalg.norm(pts, axis=1)
    r_boundary = np.min(1.0 - dist_origin)
    if r_boundary <= 0.0:
        return -np.inf
    # pairwise term
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    d2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(d2, np.inf)
    r_pair = 0.5 * np.sqrt(np.min(d2))
    return min(r_boundary, r_pair)

def _hex_lattice():
    """Deterministic hexagonal lattice cut to the unit disk (always valid)."""
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    while True:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        pts = np.array(rows)
        rad = np.hypot(pts[:, 0], pts[:, 1])
        mask = rad <= 1.0 - spacing / 2.0
        pts, rad = pts[mask], rad[mask]
        if len(pts) >= N:
            return pts[np.argsort(rad, kind="stable")[:N]]
        spacing *= 0.98

def _soft_min(values, beta=50.0):
    """Soft minimum using log-sum-exp trick for numerical stability."""
    values = np.asarray(values)
    if len(values) == 0:
        return 0.0
    v_min = np.min(values)
    # Soft min = -soft_max(-values)
    neg_vals = -values
    neg_max = np.max(neg_vals)
    exp_vals = np.exp(beta * (neg_vals - neg_max))
    soft_max_neg = neg_max + np.log(np.sum(exp_vals)) / beta
    return -soft_max_neg

def _objective_and_grad(x, beta=50.0):
    """
    Objective: Minimize -soft_min(constraints).
    Constraints are:
    1. Pairwise distances / 2
    2. Boundary distances (1 - norm)
    
    This creates a smooth landscape where the optimizer tries to maximize the minimum distance.
    """
    pts = x.reshape(-1, 2)
    
    # Boundary constraints: 1 - |c_i|
    dists_from_origin = np.linalg.norm(pts, axis=1)
    boundary_vals = 1.0 - dists_from_origin
    
    # Pairwise constraints: dist_ij / 2
    diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = np.sum(diffs ** 2, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    dists = np.sqrt(dists_sq)
    pairwise_vals = dists / 2.0
    
    # Extract upper triangle for efficiency in concatenation (though soft_min handles all)
    # Actually, soft_min of the full matrix (excluding diagonal) is fine.
    # We need to be careful with the diagonal (inf).
    # Let's just take the upper triangle to avoid duplicate pairs and the diagonal.
    upper_idx = np.triu_indices(N, k=1)
    pairwise_flat = pairwise_vals[upper_idx]
    
    all_constraints = np.concatenate([boundary_vals, pairwise_flat])
    
    # Soft min
    score_approx = _soft_min(all_constraints, beta=beta)
    
    # We want to maximize score_approx, so minimize -score_approx
    obj = -score_approx
    
    # Gradient (numerical is slow, but analytic is complex. 
    # However, SLSQP can use finite differences if grad is None.
    # Given the dimensionality (600), finite differences might be slow but acceptable for a few iterations?
    # No, 600 dimensions * 600 evaluations per gradient step is too slow.
    # We must provide an analytic gradient or use a method that doesn't require it (like Nelder-Mead, but that's worse).
    # Or, we can simplify: The gradient of soft_min is a weighted average of the gradients of the inputs.
    # w_k = exp(beta * (-v_k - neg_max)) / sum(exp(...))
    # d(soft_min)/dx = sum_k w_k * (d(-v_k)/dx) = - sum_k w_k * (dv_k/dx)
    
    # Let's compute the weights
    neg_vals = -all_constraints
    neg_max = np.max(neg_vals)
    exp_vals = np.exp(beta * (neg_vals - neg_max))
    sum_exp = np.sum(exp_vals)
    weights = exp_vals / sum_exp
    
    # Gradient of boundary_vals[i] wrt x[2i], x[2i+1]
    # v_i = 1 - sqrt(x^2+y^2)
    # dv_i/dx = -x/sqrt(x^2+y^2)
    # dv_i/dy = -y/sqrt(x^2+y^2)
    
    grad = np.zeros_like(x)
    
    # Boundary part
    for i in range(N):
        xi = pts[i, 0]
        yi = pts[i, 1]
        norm = dists_from_origin[i]
        if norm > 1e-8:
            w = weights[i] # Weight for boundary constraint i
            # d(-soft_min)/dx = - (d soft_min/dx)
            # d soft_min/dx = sum w_k * (-dv_k/dx)
            # So d obj/dx = - (sum w_k * (-dv_k/dx)) = sum w_k * dv_k/dx
            # Wait.
            # obj = - soft_min
            # d obj/dx = - d soft_min/dx
            # soft_min = - soft_max(-v)
            # d soft_min/dx = - d soft_max(-v)/dx
            # d soft_max(-v)/dx = sum w_k * d(-v_k)/dx = - sum w_k * dv_k/dx
            # So d soft_min/dx = sum w_k * dv_k/dx
            # So d obj/dx = - sum w_k * dv_k/dx
            
            # dv_i/dx = -xi/norm
            # dv_i/dy = -yi/norm
            grad[2*i] = -w * (-xi/norm)
            grad[2*i+1] = -w * (-yi/norm)
            
    # Pairwise part
    # This is expensive to compute analytically for all pairs in a loop.
    # However, we only need the gradient of the objective.
    # The weights are distributed among the pairs.
    # For each pair (i,j) with weight w_pair:
    # v_pair = dist_ij / 2
    # dv_pair/dx_i = (x_i - x_j) / (2 * dist_ij)
    # dv_pair/dx_j = (x_j - x_i) / (2 * dist_ij)
    
    # We need to map the flat index of pairwise_vals to (i,j)
    # This is complex. 
    # Alternative: Use finite differences for the gradient? 
    # With N=300, 600 variables. 
    # If we do 20 iterations of SLSQP, and each gradient eval takes 600 objective evals... 
    # Objective eval is O(N^2) = 90000 ops.
    # 600 * 90000 = 54M ops per gradient.
    # 20 iterations * (maybe 50 linesearch evals?) = 1000 evals?
    # 1000 * 54M = 54G ops. Too slow in Python.
    
    # We MUST use analytic gradient or a cheaper objective.
    # Let's compute the analytic gradient for pairs using vectorization.
    
    # We have weights for all constraints.
    # w_boundary = weights[:N]
    # w_pair = weights[N:]
    
    # The indices of the upper triangle
    # We need to accumulate gradients.
    # grad += sum_{pairs} w_pair * dv_pair/dx
    
    # Let's create a matrix of weights for pairs
    # W_pair is N x N, symmetric, 0 on diagonal.
    W_pair = np.zeros((N, N))
    # Map flat weights to matrix
    # The order of triu_indices is (0,1), (0,2)...(0,N-1), (1,2)...
    # We can just fill the matrix
    # Actually, it's easier to loop over the upper triangle indices if N is small, but N=300 is borderline.
    # Let's try to vectorize the gradient calculation for pairs.
    
    # diffs is N x N x 2
    # dists is N x N
    # We need d(dist_ij)/dx_i = (x_i - x_j)/dist_ij
    
    # Let's compute the contribution of each pair to the gradient of point i.
    # grad_i = sum_j w_{ij} * (x_i - x_j) / dist_ij
    
    # We have w_pair in a flat array corresponding to triu_indices.
    # Let's create a full weight matrix W where W[i,j] = w_pair[k] if (i,j) is the k-th pair.
    # This is memory intensive (300x300 is small though).
    
    # Fill W_pair
    # Get the indices
    i_idx, j_idx = np.triu_indices(N, k=1)
    W_pair[i_idx, j_idx] = weights[N:]
    W_pair[j_idx, i_idx] = weights[N:] # Symmetric
    
    # Now compute gradient for pairs
    # dists is N x N
    # We need to avoid division by zero (diagonal)
    dists_safe = np.where(dists_sq > 0, dists, 1.0) # Diagonal is inf, so dists is inf? No, we filled d2 with inf, sqrt is inf.
    # Actually, dists_sq diagonal is inf. dists diagonal is inf.
    # We want 0 contribution from diagonal.
    
    # Vectorized gradient for pairs:
    # G_i = sum_j W_pair[i,j] * (x_i - x_j) / dists[i,j]
    
    # Let's use einsum or broadcasting.
    # diff is x_i - x_j
    # factor is W_pair / dists
    # factor is N x N
    factor = W_pair / dists_safe
    np.fill_diagonal(factor, 0.0)
    
    # grad_pair_x = sum_j factor[i,j] * (x_i - x_j)
    # x_i is pts[:, 0][:, None]
    # x_j is pts[:, 0][None, :]
    # term_x = factor * (pts[:, 0][:, None] - pts[:, 0][None, :])
    # grad_x = sum_j term_x[i,j]
    
    term_x = factor * (pts[:, 0][:, None] - pts[:, 0][None, :])
    term_y = factor * (pts[:, 1][:, None] - pts[:, 1][None, :])
    
    grad_pair_x = np.sum(term_x, axis=1)
    grad_pair_y = np.sum(term_y, axis=1)
    
    # Combine with boundary gradient
    # d obj/dx = - sum w_k dv_k/dx
    # We already computed boundary part as -w * (-x/norm) = w * x/norm? 
    # Let's re-verify signs.
    # obj = - soft_min
    # d obj/dx = - d soft_min/dx
    # d soft_min/dx = sum w_k dv_k/dx
    # So d obj/dx = - sum w_k dv_k/dx
    
    # Boundary: v = 1 - r. dv/dx = -x/r.
    # d obj/dx = - w * (-x/r) = w * x/r.
    # My previous code: grad[2*i] = -w * (-xi/norm) = w * xi/norm. Correct.
    
    # Pairs: v = d/2. dv/dx_i = (x_i - x_j)/d.
    # d obj/dx_i = - w_{ij} * (x_i - x_j)/d.
    # My vectorized calc:
    # term_x = factor * (x_i - x_j) where factor = W/d.
    # sum term_x = sum W/d * (x_i - x_j).
    # This is sum w_{ij} * (x_i - x_j)/d.
    # So grad_pair_x is sum w_{ij} * (x_i - x_j)/d.
    # We need d obj/dx = - (this sum).
    
    grad[0::2] += -grad_pair_x
    grad[1::2] += -grad_pair_y
    
    return obj, grad

def _refine(pts, deadline):
    """Improve pts by nudging the closest pair / boundary offender apart."""
    pts = np.array(pts, dtype=float, copy=True)
    cur = _radius(pts)
    if cur <= 0.0:
        return None

    step = max(1e-4, 0.05 * cur)
    it = 0
    while time.time() < deadline - 2.0:
        it += 1
        # find bottleneck
        dist_origin = np.linalg.norm(pts, axis=1)
        r_boundary = 1.0 - np.min(dist_origin)
        i_b = int(np.argmin(dist_origin))

        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        d2 = np.einsum('ijk,ijk->ij', diff, diff)
        np.fill_diagonal(d2, np.inf)
        ii, jj = np.unravel_index(np.argmin(d2), d2.shape)
        d = np.sqrt(d2[ii, jj])
        r_pair = 0.5 * d

        if r_pair < r_boundary:
            # separate the closest pair
            v = pts[jj] - pts[ii]
            nrm = np.linalg.norm(v)
            if nrm < 1e-12:
                v = np.array([1.0, 0.0]); nrm = 1.0
            v = v / nrm
            pts[ii] -= 0.5 * step * v
            pts[jj] += 0.5 * step * v
        else:
            # push boundary offender toward centre
            p = pts[i_b]
            nrm = np.linalg.norm(p)
            if nrm < 1e-12:
                continue
            pts[i_b] -= step * (p / nrm)

        # clamp inside (safety)
        d_o = np.linalg.norm(pts, axis=1)
        over = d_o > 0.9999
        if np.any(over):
            pts[over] *= (0.9999 / d_o[over])[:, None]

        new_r = _radius(pts)
        if new_r > cur:
            cur = new_r
            step = max(1e-5, step * 0.97)
        else:
            # revert, shrink step
            step *= 0.7
            if step < 1e-7:
                break
            # revert last move
            if r_pair < r_boundary:
                v = pts[jj] - pts[ii]
                nrm = np.linalg.norm(v)
                if nrm < 1e-12:
                    continue
                v = v / nrm
                pts[ii] += 0.5 * (step / 0.7) * v
                pts[jj] -= 0.5 * (step / 0.7) * v
            else:
                p = pts[i_b]
                nrm = np.linalg.norm(p)
                if nrm < 1e-12:
                    break
                pts[i_b] += (step / 0.7) * (p / nrm)

    return pts if cur > 0.0 else None

def construct():
    start = time.time()
    deadline = start + 145.0  # 150s wall limit, 5s safety

    # baseline (always valid)
    hex_pts = _hex_lattice()
    best_pts, best_r = hex_pts, _radius(hex_pts)

    # Optimization phase
    # Flatten the points
    x0 = hex_pts.flatten()
    
    # Bounds: keep points within [-1, 1]
    bounds = [(-1.0, 1.0)] * (2 * N)
    
    # Try SLSQP with the smooth objective
    # We use a moderate beta for smoothness
    beta = 50.0
    
    try:
        res = minimize(
            _objective_and_grad,
            x0,
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 100, 'ftol': 1e-10, 'disp': False}
        )
        
        if res.success or (hasattr(res, 'x') and res.x is not None):
            cand = res.x.reshape(N, 2)
            r = _radius(cand)
            if r > best_r:
                best_r, best_pts = r, cand
    except Exception as e:
        pass # Fallback to hex_pts if SLSQP fails

    # Refine the best seed within remaining time
    if time.time() < deadline - 3.0:
        refined = _refine(best_pts, deadline)
        if refined is not None and _radius(refined) > best_r:
            best_pts = refined
            best_r = _radius(best_pts)

    # final validity guard
    if _radius(best_pts) <= 0.0:
        return hex_pts
    return best_pts

# EVOLVE-BLOCK-END
