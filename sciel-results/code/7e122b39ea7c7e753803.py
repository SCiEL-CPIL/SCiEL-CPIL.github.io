"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _score(c):
    """
    Compute the actual score (largest common radius) for a configuration.
    c: (N, 3) array of centers.
    Returns the radius r.
    """
    if c.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(c)):
        return 0.0
    # Check boundaries: min distance to each face
    face_dists = 0.5 - np.abs(c)
    min_face = np.min(face_dists)
    if min_face < -1e-12:
        return 0.0
    # Pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    r = min(min_pair / 2.0, min_face)
    return r


def _soft_min_obj_and_grad(c_flat, beta):
    """
    Compute -soft_min(all_constraints) and its gradient w.r.t. c_flat.
    
    The soft min is: -log(sum(exp(-beta * d_k))) / beta
    where d_k are pairwise half-distances and face distances.
    
    We want to maximize min, so we minimize -soft_min.
    
    Returns (value, gradient) where gradient is w.r.t. c_flat (shape 3N).
    """
    c = c_flat.reshape(N, 3)
    
    # --- Pairwise distances / 2 ---
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)  # (N, N)
    np.fill_diagonal(dists, np.inf)
    
    iu = np.triu_indices(N, k=1)
    n_pairs = len(iu[0])
    pair_dists = dists[iu] / 2.0  # shape (M_pair,)
    
    # --- Face distances ---
    face_dists = 0.5 - np.abs(c)  # shape (N, 3)
    face_flat = face_dists.flatten()  # shape (3N,)
    
    # --- Combine ---
    all_dists = np.concatenate([pair_dists, face_flat])
    
    # Soft min computation
    d_min = np.min(all_dists)
    shifted = -beta * (all_dists - d_min)
    shifted = np.clip(shifted, -500, 0)
    exp_terms = np.exp(shifted)
    sum_exp = np.sum(exp_terms)
    weights = exp_terms / sum_exp  # shape (M_total,)
    
    r_smooth = d_min - np.log(sum_exp) / beta
    value = -r_smooth
    
    # --- Gradient (vectorized) ---
    grad = np.zeros((N, 3))
    
    # Pairwise part: d_k = |c_i - c_j| / 2
    # d(d_k)/dc_i = (c_i - c_j) / (2 * |c_i - c_j|)
    # d(-r_smooth)/dc_i = -sum_k w_k * d(d_k)/dc_i
    # So for pair (i,j): contribution to grad[i] -= w * (c_i - c_j) / (2 * d_ij)
    #                   contribution to grad[j] += w * (c_i - c_j) / (2 * d_ij)
    
    pair_weights = weights[:n_pairs]
    i_idx = iu[0]
    j_idx = iu[1]
    
    # Compute factors: w / (2 * d_ij) for each pair
    d_ij = dists[i_idx, j_idx]
    safe_d = np.where(d_ij > 1e-15, d_ij, 1.0)
    factors = pair_weights / (2.0 * safe_d)
    # Zero out factors where weight is negligible or distance is too small
    mask_valid = (pair_weights > 1e-15) & (d_ij > 1e-15)
    factors = np.where(mask_valid, factors, 0.0)
    
    # diff_ij[i] = c[i_idx[i]] - c[j_idx[i]], shape (n_pairs, 3)
    diff_ij = c[i_idx] - c[j_idx]
    
    # Accumulate gradients using np.add.at
    # grad[i_idx] -= factors[:, None] * diff_ij
    # grad[j_idx] += factors[:, None] * diff_ij
    np.add.at(grad, i_idx, -factors[:, np.newaxis] * diff_ij)
    np.add.at(grad, j_idx, factors[:, np.newaxis] * diff_ij)
    
    # Face part: d_k = 0.5 - |c_ik|
    # d(d_k)/dc_ik = -sign(c_ik)
    # d(-r_smooth)/dc_ik = -w * (-sign) = w * sign
    face_weights = weights[n_pairs:]
    signs = np.sign(c)
    signs[signs == 0] = 1.0
    grad += face_weights.reshape(N, 3) * signs
    
    return value, grad.flatten()


def _potential_obj_and_grad(c_flat, p=4.0, eps=1e-6):
    """
    Sum of inverse-power repulsive potentials: sum 1/d_k^p for all constraints.
    Minimize this to spread points out.
    
    Returns (value, gradient).
    """
    c = c_flat.reshape(N, 3)
    
    # --- Pairwise distances ---
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)  # (N, N)
    np.fill_diagonal(dists, np.inf)
    
    iu = np.triu_indices(N, k=1)
    n_pairs = len(iu[0])
    d_ij = dists[iu]  # shape (M_pair,)
    safe_d = np.maximum(d_ij, eps)
    
    # --- Face distances ---
    face_dists = 0.5 - np.abs(c)  # shape (N, 3)
    safe_face = np.maximum(face_dists, eps)
    
    # --- Objective: sum 1/d^p ---
    pair_contrib = 1.0 / (safe_d ** p)
    face_contrib = 1.0 / (safe_face ** p)
    value = np.sum(pair_contrib) + np.sum(face_contrib)
    
    # --- Gradient ---
    # d(1/d^p)/dc_i for pair (i,j) = -p * (c_i - c_j) / d^(p+2)
    # d(1/d^p)/dc_ik for face = p * sign(c_ik) / d^(p+1)
    grad = np.zeros((N, 3))
    
    i_idx = iu[0]
    j_idx = iu[1]
    diff_ij = c[i_idx] - c[j_idx]
    
    # Pair gradient factors: -p / d^(p+2)
    pair_factors = -p / (safe_d ** (p + 2))
    np.add.at(grad, i_idx, pair_factors[:, np.newaxis] * diff_ij)
    np.add.at(grad, j_idx, -pair_factors[:, np.newaxis] * diff_ij)
    
    # Face gradient
    signs = np.sign(c)
    signs[signs == 0] = 1.0
    face_factors = p / (safe_face ** (p + 1))
    grad += face_factors * signs
    
    return value, grad.flatten()


def _gradient_flow(c_init, p=4.0, n_steps=2000, lr_init=0.005, lr_decay=0.999, eps=1e-6, deadline=None):
    """
    Damped gradient flow with adaptive step size.
    """
    c = c_init.copy()
    lr = lr_init
    best_c = c.copy()
    best_score = _score(c)
    
    for step in range(n_steps):
        if deadline is not None and time.time() > deadline:
            break
        
        val, grad = _potential_obj_and_grad(c.flatten(), p=p, eps=eps)
        g = grad.reshape(N, 3)
        
        # Adaptive step: scale by inverse of gradient norm
        g_norm = np.linalg.norm(g)
        if g_norm < 1e-12:
            break
        effective_lr = lr / (1.0 + g_norm)
        
        # Update
        c_new = c - effective_lr * g
        c_new = np.clip(c_new, -0.4999, 0.4999)
        
        # Check if improved
        s_new = _score(c_new)
        if s_new > best_score:
            best_score = s_new
            best_c = c_new.copy()
            c = c_new
        else:
            # Reject: reduce learning rate
            lr *= 0.9
            if lr < 1e-8:
                break
        
        lr *= lr_decay
    
    return best_c


def _generate_bcc_init(n_points):
    """
    Generate initial points on a BCC (body-centered cubic) lattice.
    BCC has 2 points per unit cell: corners + center.
    For a cube of side 1, we tile with small BCC cells.
    """
    # We want approximately n_points points.
    # BCC: in a cubic cell of side a, we have 2 atoms.
    # Total points ~ (1/a)^3 * 2
    # So a ~ (2/n_points)^(1/3)
    
    # Try different scales and pick the one closest to n_points
    best_points = None
    best_diff = float('inf')
    
    for scale in np.arange(0.5, 1.5, 0.01):
        a = scale * (2.0 / n_points) ** (1.0/3.0)
        n_cells = int(np.floor(1.0 / a))
        if n_cells < 1:
            continue
        
        points = []
        for ix in range(n_cells):
            for iy in range(n_cells):
                for iz in range(n_cells):
                    # Corner
                    cx = ix * a + 0.5
                    cy = iy * a + 0.5
                    cz = iz * a + 0.5
                    points.append([cx, cy, cz])
                    # Body center
                    points.append([cx + a/2, cy + a/2, cz + a/2])
        
        pts = np.array(points)
        # Keep only points inside [-0.5, 0.5]^3
        mask = np.all(pts >= -0.5 + 1e-6, axis=1) & np.all(pts <= 0.5 - 1e-6, axis=1)
        pts = pts[mask]
        
        if len(pts) >= n_points:
            # Take first n_points
            pts = pts[:n_points]
            diff = abs(len(pts) - n_points)
            if diff < best_diff:
                best_diff = diff
                best_points = pts.copy()
        elif len(pts) > 0:
            # Pad with random points if needed
            remaining = n_points - len(pts)
            rng = np.random.default_rng(42)
            extra = rng.uniform(-0.45, 0.45, size=(remaining, 3))
            pts = np.vstack([pts, extra])
            if best_points is None:
                best_points = pts.copy()
                best_diff = remaining
    
    if best_points is None:
        # Fallback: simple grid
        m = int(np.ceil(n_points ** (1.0/3.0)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_points = grid[:n_points].copy()
    
    return best_points


def _optimize_from_initial(c_init, beta, maxiter=300, ftol=1e-12, gtol=1e-10):
    """Run L-BFGS-B optimization with analytic gradient."""
    try:
        res = minimize(
            _soft_min_obj_and_grad,
            c_init.flatten(),
            args=(beta,),
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol, 'maxls': 20}
        )
        return res.x.reshape(N, 3)
    except Exception:
        return c_init


def _local_polish(c, n_iter=20, step=0.001):
    """
    Coordinate-wise local polishing: for each point and each coordinate,
    try small perturbations and keep if score improves.
    """
    c = c.copy()
    current_score = _score(c)
    
    for it in range(n_iter):
        improved = False
        for i in range(N):
            for k in range(3):
                best_val = c[i, k]
                best_s = current_score
                # Try several step sizes
                for s in [step, step*0.5, step*2, step*0.25]:
                    for direction in [-1, 1]:
                        trial = c.copy()
                        trial[i, k] = c[i, k] + direction * s
                        if trial[i, k] < -0.5 or trial[i, k] > 0.5:
                            continue
                        s_trial = _score(trial)
                        if s_trial > best_s + 1e-14:
                            best_s = s_trial
                            best_val = trial[i, k]
                if abs(best_val - c[i, k]) > 1e-12:
                    c[i, k] = best_val
                    current_score = best_s
                    improved = True
        if not improved:
            break
        # Halve step size each iteration
        step *= 0.5
    
    return c


def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin (150s limit)
    
    # BCC lattice initialization
    c0 = _generate_bcc_init(N)
    c0 = np.clip(c0, -0.49, 0.49)
    
    best_c = c0.copy()
    best_score = _score(best_c)
    
    rng = np.random.default_rng(42)
    
    # Phase 1: Potential field gradient flow (smooth landscape, avoids sharp minima)
    # Use multiple power exponents for different landscape characteristics
    for p in [2.0, 4.0, 6.0]:
        if time.time() > deadline - 60.0:
            break
        c_pf = _gradient_flow(best_c, p=p, n_steps=1500, lr_init=0.01, lr_decay=0.9995, deadline=deadline - 50.0)
        s_pf = _score(c_pf)
        if s_pf > best_score:
            best_score = s_pf
            best_c = c_pf.copy()
    
    # Phase 2: Soft-min continuation with beta annealing
    beta_schedule = [10.0, 30.0, 80.0, 200.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0, 15000.0]
    
    for beta in beta_schedule:
        if time.time() > deadline - 20.0:
            break
        
        # Use current best as starting point
        c_opt = _optimize_from_initial(best_c, beta, maxiter=400, ftol=1e-13, gtol=1e-11)
        c_opt = np.clip(c_opt, -0.4999, 0.4999)
        score = _score(c_opt)
        if score > best_score:
            best_score = score
            best_c = c_opt.copy()
        
        # Also try a perturbed version
        if time.time() < deadline - 20.0:
            c_pert = best_c + rng.normal(0, 0.01, size=(N, 3))
            c_pert = np.clip(c_pert, -0.49, 0.49)
            c_opt2 = _optimize_from_initial(c_pert, beta, maxiter=250, ftol=1e-13, gtol=1e-11)
            c_opt2 = np.clip(c_opt2, -0.4999, 0.4999)
            score2 = _score(c_opt2)
            if score2 > best_score:
                best_score = score2
                best_c = c_opt2.copy()
    
    # Phase 3: Final high-precision polish
    for polish_beta in [20000.0, 50000.0]:
        if time.time() > deadline - 10.0:
            break
        c_polished = _optimize_from_initial(best_c, polish_beta, maxiter=800, ftol=1e-14, gtol=1e-12)
        c_polished = np.clip(c_polished, -0.49999, 0.49999)
        score = _score(c_polished)
        if score > best_score:
            best_score = score
            best_c = c_polished.copy()
    
    # Phase 4: Coordinate-wise local polishing
    if time.time() < deadline - 3.0:
        c_final = _local_polish(best_c, n_iter=40, step=0.002)
        score_final = _score(c_final)
        if score_final > best_score:
            best_score = score_final
            best_c = c_final
    
    return best_c
# EVOLVE-BLOCK-END
