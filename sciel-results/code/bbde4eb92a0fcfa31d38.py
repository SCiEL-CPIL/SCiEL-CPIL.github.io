import time
import numpy as np
from scipy.optimize import minimize

N = 50
_START_TIME = time.time()
_DEADLINE = _START_TIME + 142.0  # 8s safety margin before 150s kill

def _time_left():
    return _DEADLINE - time.time()

def _score(pts):
    """Compute the valid packing radius (or 0 if invalid)."""
    if pts.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    dists_origin = np.linalg.norm(pts, axis=1)
    if np.any(dists_origin > 1.0 + 1e-9):
        return 0.0
    r_cont = 1.0 - np.max(dists_origin)
    if r_cont < 0:
        return 0.0
    if N < 2:
        return r_cont
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_mat = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dist_mat, np.inf)
    r_pair = 0.5 * np.min(dist_mat)
    return min(r_cont, r_pair)

def _soft_min_obj_and_grad(flat, beta):
    """
    Soft-min objective for L-BFGS-B.
    Objective: minimize f(x) = (1/beta) * ln(sum(exp(-beta * x_k)))
    where x_k are the critical distances (pairwise/2 and boundary).
    Minimizing this is equivalent to maximizing the soft-min of x_k.
    """
    pts = flat.reshape(N, 2)
    
    # Calculate boundary distances: 1 - |c_i|
    dists_origin = np.linalg.norm(pts, axis=1)
    boundary_dists = 1.0 - dists_origin
    
    # Calculate pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_mat = np.sqrt(np.sum(diff ** 2, axis=-1))
    
    iu = np.triu_indices(N, k=1)
    pair_dists = dist_mat[iu]
    
    # Combine all critical distances
    all_dists = np.concatenate([boundary_dists, pair_dists / 2.0])
    
    # Soft min calculation with numerical stability
    y = -beta * all_dists
    max_y = np.max(y)
    y_shifted = y - max_y
    sum_exp = np.sum(np.exp(y_shifted))
    log_sum_exp = max_y + np.log(sum_exp)
    obj_val = log_sum_exp / beta
    
    # Gradient
    weights = np.exp(y_shifted) / sum_exp
    
    grad_pts = np.zeros_like(pts)
    
    # 1. Boundary terms
    w_boundary = weights[0:N]
    safe_dists = np.where(dists_origin < 1e-8, 1.0, dists_origin)
    boundary_grad = (w_boundary / safe_dists)[:, np.newaxis] * pts
    grad_pts += boundary_grad
    
    # 2. Pairwise terms
    w_pairs = weights[N:]
    i_idx, m_idx = iu
    d_pairs = dist_mat[i_idx, m_idx]
    safe_d_pairs = np.where(d_pairs < 1e-8, 1.0, d_pairs)
    vecs = pts[i_idx] - pts[m_idx]
    unit_vecs = vecs / safe_d_pairs[:, np.newaxis]
    
    grad_contrib_i = -0.5 * w_pairs[:, np.newaxis] * unit_vecs
    grad_contrib_m = 0.5 * w_pairs[:, np.newaxis] * unit_vecs
    
    np.add.at(grad_pts, i_idx, grad_contrib_i)
    np.add.at(grad_pts, m_idx, grad_contrib_m)
    
    grad_flat = grad_pts.flatten()
    
    return obj_val, grad_flat

def _force_directed_relax(pts, n_steps=200, target_r=0.12):
    """
    Force-directed relaxation to improve packing.
    Pushes overlapping circles apart and pulls them toward the center.
    """
    pts = pts.copy()
    target_d = 2.0 * target_r
    
    for step in range(n_steps):
        if _time_left() < 5.0:
            break
        
        # Compute forces
        forces = np.zeros_like(pts)
        
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_mat = np.sqrt(np.sum(diff ** 2, axis=-1))
        np.fill_diagonal(dist_mat, np.inf)
        
        # Only consider pairs that are too close
        for i in range(N):
            for j in range(i + 1, N):
                d = dist_mat[i, j]
                if d < target_d and d > 1e-10:
                    overlap = target_d - d
                    direction = (pts[i] - pts[j]) / d
                    force_mag = overlap * 0.5
                    forces[i] += force_mag * direction
                    forces[j] -= force_mag * direction
        
        # Boundary constraint: push circles inward if they're outside
        dists_origin = np.linalg.norm(pts, axis=1)
        for i in range(N):
            if dists_origin[i] > 1.0 - target_r:
                direction = -pts[i] / dists_origin[i]
                excess = dists_origin[i] - (1.0 - target_r)
                forces[i] += excess * direction
        
        # Update positions with adaptive learning rate
        lr = 0.01 * (1.0 - step / max(n_steps, 1)) + 0.001
        pts += lr * forces
        
        # Project to unit disk
        dists = np.linalg.norm(pts, axis=1)
        mask = dists > 1.0
        if np.any(mask):
            pts[mask] = pts[mask] / dists[mask, np.newaxis]
    
    return pts

def _hex_lattice_init(r_guess=0.12):
    """Generate a hexagonal lattice packing as initial guess."""
    spacing = 2.0 * r_guess
    pts = []
    k = int(1.0 / spacing) + 3
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            if x * x + y * y <= (1.0 - r_guess) ** 2:
                pts.append([x, y])
    pts = np.array(pts)
    if len(pts) >= N:
        radii = np.hypot(pts[:, 0], pts[:, 1])
        idx = np.argsort(radii, kind='stable')[:N]
        return pts[idx]
    else:
        return _ring_init([1, 6, 12, 18, 13], [0.0, 0.25, 0.50, 0.75, 0.95])

def _ring_init(ring_sizes, ring_radii, angle_offsets=None):
    """Generate points from concentric rings."""
    points = np.zeros((N, 2))
    idx = 0
    if angle_offsets is None:
        angle_offsets = [0.0] * len(ring_sizes)
    for ring_idx, (size, radius) in enumerate(zip(ring_sizes, ring_radii)):
        offset = angle_offsets[ring_idx] if ring_idx < len(angle_sizes) else 0.0
        for j in range(size):
            angle = 2 * np.pi * j / size + offset
            points[idx] = [radius * np.cos(angle), radius * np.sin(angle)]
            idx += 1
    return points

def _random_init():
    angles = np.random.uniform(0, 2 * np.pi, N)
    radii = np.sqrt(np.random.uniform(0, 0.95, N))
    return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])

def _soft_min_optimize(pts_init, beta_schedule, max_iter_per_stage=500):
    """
    Multi-stage soft-min optimization with beta annealing.
    """
    current_pts = pts_init.copy()
    
    for stage, beta in enumerate(beta_schedule):
        if _time_left() < 10.0:
            break
        
        def objective(x):
            return _soft_min_obj_and_grad(x, beta)
        
        x0 = current_pts.flatten()
        result = minimize(objective, x0, method='L-BFGS-B', jac=True,
                          options={'maxiter': max_iter_per_stage, 'ftol': 1e-14,
                                   'gtol': 1e-10, 'maxls': 50})
        current_pts = result.x.reshape(N, 2)
        
        # Project to ensure validity
        dists = np.linalg.norm(current_pts, axis=1)
        mask = dists > 1.0
        if np.any(mask):
            current_pts[mask] = current_pts[mask] / dists[mask, np.newaxis]
    
    return current_pts

def _identify_contacts(pts, r, tol=1e-4):
    """Identify the contact graph: pairs at distance ~2r and boundary contacts at ~1-r."""
    dists = np.linalg.norm(pts, axis=1)
    boundary_contacts = dists > (1.0 - r - tol)
    
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_mat = np.linalg.norm(diff, axis=-1)
    np.fill_diagonal(dist_mat, np.inf)
    pair_contacts = dist_mat < (2.0 * r + tol)
    # Get unique pairs
    iu = np.triu_indices(N, k=1)
    pair_contact_mask = pair_contacts[iu]
    contact_pairs = list(zip(iu[0][pair_contact_mask], iu[1][pair_contact_mask]))
    
    return contact_pairs, boundary_contacts

def _newton_solve(pts, contact_pairs, boundary_contacts, max_iter=200, tol=1e-12):
    """
    Solve the KKT system using Newton-Raphson.
    Unknowns: positions (2N) and radius r (1) -> 2N+1 unknowns.
    Equations:
      For each contact pair (i,j): |c_i - c_j| - 2r = 0
      For each boundary contact i: |c_i| - (1-r) = 0
    We need at least 2N+1 equations. If we have fewer, we add
    "free" equations (e.g., fix some coordinates or use additional
    near-contact pairs).
    """
    x = np.concatenate([pts.flatten(), [0.0]])  # last element is r
    r_init = _score(pts)
    x[-1] = r_init
    
    n_pairs = len(contact_pairs)
    n_bound = np.sum(boundary_contacts)
    n_eq = n_pairs + n_bound
    n_var = 2 * N + 1
    
    if n_eq < n_var:
        # Need more equations: add near-contact pairs
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_mat = np.linalg.norm(diff, axis=-1)
        np.fill_diagonal(dist_mat, np.inf)
        iu = np.triu_indices(N, k=1)
        pair_dists = dist_mat[iu]
        # Sort by distance, add closest pairs until we have enough
        order = np.argsort(pair_dists)
        existing_pairs = set(contact_pairs)
        extra_pairs = []
        for idx in order:
            i, j = iu[0][idx], iu[1][idx]
            if (i, j) not in existing_pairs and (j, i) not in existing_pairs:
                extra_pairs.append((i, j))
                existing_pairs.add((i, j))
            if n_eq + len(extra_pairs) >= n_var:
                break
        contact_pairs = contact_pairs + extra_pairs
        n_pairs = len(contact_pairs)
        n_eq = n_pairs + n_bound
    
    # If still not enough, we'll use a least-squares approach
    use_least_squares = (n_eq < n_var)
    
    def _residual_and_jacobian(x):
        pts_x = x[:2*N].reshape(N, 2)
        r = x[-1]
        
        residuals = []
        jac_rows = []
        
        # Pair constraints: |c_i - c_j| - 2r = 0
        for (i, j) in contact_pairs:
            d = pts_x[i] - pts_x[j]
            dist = np.linalg.norm(d)
            if dist < 1e-10:
                residuals.append(-2.0 * r)
                jac_rows.append(np.zeros(2*N+1))
                continue
            residuals.append(dist - 2.0 * r)
            row = np.zeros(2*N+1)
            unit = d / dist
            row[2*i:2*i+2] = unit
            row[2*j:2*j+2] = -unit
            row[-1] = -2.0
            jac_rows.append(row)
        
        # Boundary constraints: |c_i| - (1-r) = 0
        for i in range(N):
            if boundary_contacts[i]:
                dist = np.linalg.norm(pts_x[i])
                if dist < 1e-10:
                    residuals.append(-(1.0 - r))
                    jac_rows.append(np.zeros(2*N+1))
                    continue
                residuals.append(dist - (1.0 - r))
                row = np.zeros(2*N+1)
                unit = pts_x[i] / dist
                row[2*i:2*i+2] = unit
                row[-1] = 1.0
                jac_rows.append(row)
        
        res = np.array(residuals)
        J = np.array(jac_rows)
        return res, J
    
    for it in range(max_iter):
        if _time_left() < 5.0:
            break
        res, J = _residual_and_jacobian(x)
        
        if use_least_squares:
            # Least squares solve
            try:
                dx, _, _, _ = np.linalg.lstsq(J, -res, rcond=None)
            except np.linalg.LinAlgError:
                break
        else:
            # Direct solve
            try:
                dx = np.linalg.solve(J, -res)
            except np.linalg.LinAlgError:
                try:
                    dx, _, _, _ = np.linalg.lstsq(J, -res, rcond=None)
                except np.linalg.LinAlgError:
                    break
        
        # Damping
        step = 1.0
        for _ in range(20):
            x_new = x + step * dx
            # Keep points in reasonable range
            pts_new = x_new[:2*N].reshape(N, 2)
            if np.any(np.linalg.norm(pts_new, axis=1) > 1.5):
                step *= 0.5
                continue
            r_new = x_new[-1]
            if r_new < 0 or r_new > 0.2:
                step *= 0.5
                continue
            res_new, _ = _residual_and_jacobian(x_new)
            if np.linalg.norm(res_new) < np.linalg.norm(res) * (1.0 + 1e-4):
                break
            step *= 0.5
        
        x = x + step * dx
        r_current = x[-1]
        pts_current = x[:2*N].reshape(N, 2)
        
        # Check convergence
        if np.linalg.norm(res) < tol:
            break
        if np.linalg.norm(dx) < 1e-14:
            break
    
    pts_result = x[:2*N].reshape(N, 2)
    r_result = x[-1]
    
    # Project to valid disk
    dists = np.linalg.norm(pts_result, axis=1)
    too_far = dists > 1.0
    if np.any(too_far):
        pts_result[too_far] = pts_result[too_far] / dists[too_far, np.newaxis]
    
    return pts_result, r_result

def _full_optimize(pts, time_budget=20.0):
    """Full optimization pipeline: force-directed -> Newton -> soft-min annealing -> Newton polish."""
    start = time.time()
    
    # Step 1: Force-directed relaxation
    if _time_left() > time_budget * 0.25 + 5.0:
        pts = _force_directed_relax(pts, n_steps=150)
    
    # Step 2: Newton-Raphson with active-set
    if _time_left() > time_budget * 0.5 + 5.0:
        r_est = _score(pts)
        contact_pairs, boundary_contacts = _identify_contacts(pts, r_est, tol=1e-3)
        pts, r_newton = _newton_solve(pts, contact_pairs, boundary_contacts, max_iter=100)
        s_newton = _score(pts)
    
    # Step 3: Multi-stage soft-min optimization with beta annealing
    if _time_left() > time_budget * 0.75 + 5.0:
        beta_schedule = [100, 500, 1000, 5000]
        pts = _soft_min_optimize(pts, beta_schedule, max_iter_per_stage=200)
    
    # Step 4: Another Newton pass on refined solution
    if _time_left() > time_budget * 0.9 + 5.0:
        r_est = _score(pts)
        contact_pairs, boundary_contacts = _identify_contacts(pts, r_est, tol=5e-4)
        pts, r_newton2 = _newton_solve(pts, contact_pairs, boundary_contacts, max_iter=100)
        s2 = _score(pts)
    
    return pts

def construct():
    global _START_TIME
    _START_TIME = time.time()
    _DEADLINE = _START_TIME + 142.0
    
    # Generate multiple initial configurations
    inits = []
    inits.append(_hex_lattice_init())
    
    # Ring configurations
    ring_configs = [
        ([1, 6, 12, 18, 13], [0.0, 0.25, 0.50, 0.75, 0.95]),
        ([1, 6, 12, 20, 11], [0.0, 0.25, 0.50, 0.75, 0.95]),
        ([1, 6, 14, 18, 11], [0.0, 0.25, 0.50, 0.75, 0.95]),
        ([1, 8, 12, 18, 11], [0.0, 0.25, 0.50, 0.75, 0.95]),
    ]
    for sizes, radii in ring_configs:
        inits.append(_ring_init(sizes, radii))
    
    for _ in range(3):
        inits.append(_random_init())
    
    best_pts = inits[0].copy()
    best_score = _score(best_pts)
    
    # Try each initialization with full optimization
    for init_pts in inits:
        if _time_left() < 15.0:
            break
        
        # Time budget per initialization
        elapsed = time.time() - _START_TIME
        remaining = _DEADLINE - 15.0 - elapsed
        if remaining <= 0:
            break
        time_budget = min(remaining / max(1, len(inits)), 25.0)
        
        try:
            optimized = _full_optimize(init_pts.copy(), time_budget=time_budget)
            s = _score(optimized)
            if s > best_score:
                best_score = s
                best_pts = optimized.copy()
        except Exception:
            continue
    
    # Final polish on the best solution
    if _time_left() > 10.0:
        try:
            # Try multi-stage soft-min on the best solution
            beta_schedule = [500, 1000, 5000]
            pts_polished = _soft_min_optimize(best_pts, beta_schedule, max_iter_per_stage=300)
            s_polished = _score(pts_polished)
            if s_polished > best_score:
                best_score = s_polished
                best_pts = pts_polished
        except Exception:
            pass
    
    if _time_left() > 5.0:
        try:
            # Final Newton polish
            r_est = _score(best_pts)
            contact_pairs, boundary_contacts = _identify_contacts(best_pts, r_est, tol=3e-4)
            final_pts, _ = _newton_solve(best_pts, contact_pairs, boundary_contacts, max_iter=50)
            s_final2 = _score(final_pts)
            if s_final2 > best_score:
                best_score = s_final2
                best_pts = final_pts
        except Exception:
            pass
    
    return best_pts
