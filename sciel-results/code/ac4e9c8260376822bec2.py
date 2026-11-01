import time
import numpy as np
from scipy.optimize import minimize

N = 50
_DEADLINE = time.time() + 142.0  # 8s safety margin before 150s kill

def _score(pts):
    """Compute the valid packing radius (or 0 if invalid)."""
    if len(pts) != N:
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
    dist_mat = np.linalg.norm(diff, axis=-1)
    np.fill_diagonal(dist_mat, np.inf)
    r_pair = 0.5 * np.min(dist_mat)
    return min(r_cont, r_pair)

def _force_directed(pts, n_iters=200, k_boundary=50.0, k_pair=100.0):
    """Force-directed relaxation to find a good contact structure."""
    pts = pts.copy().astype(float)
    for it in range(n_iters):
        forces = np.zeros_like(pts)
        # Boundary repulsion
        dists = np.linalg.norm(pts, axis=1)
        safe_dists = np.maximum(dists, 1e-8)
        # Push inward if too close to boundary
        boundary_force = k_boundary * np.maximum(0.0, dists - 0.85)
        forces -= (boundary_force / safe_dists)[:, np.newaxis] * pts
        # Pairwise repulsion (inverse square for short range)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_mat = np.linalg.norm(diff, axis=-1)
        np.fill_diagonal(dist_mat, np.inf)
        # Repulsive force ~ 1/d^2 for d < threshold
        threshold = 0.35
        mask = dist_mat < threshold
        inv_d2 = np.where(mask, 1.0 / np.maximum(dist_mat, 1e-8)**2, 0.0)
        force_mat = inv_d2 * k_pair
        np.fill_diagonal(force_mat, 0.0)
        # Force on i from j: direction (p_i - p_j) / d * strength
        # = (p_i - p_j) / d^3 * k  (since strength ~ 1/d^2, direction ~ 1/d)
        dist_safe = np.maximum(dist_mat, 1e-8)
        force_vec = force_mat[:, :, np.newaxis] * diff / dist_safe[:, :, np.newaxis]
        forces += np.sum(force_vec, axis=1)
        # Integrate with damping
        lr = 0.01 * (1.0 - it / n_iters)
        pts += lr * forces
        # Project to unit disk
        new_dists = np.linalg.norm(pts, axis=1)
        too_far = new_dists > 1.0
        if np.any(too_far):
            pts[too_far] = pts[too_far] / new_dists[too_far, np.newaxis]
    return pts

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
        if time.time() > _DEADLINE - 5.0:
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

def _hex_lattice_init():
    """Generate a hexagonal lattice packing as initial guess."""
    best_pts = None
    best_score = -1.0
    
    for spacing in [0.24, 0.25, 0.26, 0.27, 0.28]:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        
        mask = radius <= 1.0
        points = points[mask]
        radius = radius[mask]
        
        if len(points) >= N:
            idx = np.argsort(radius, kind="stable")[:N]
            pts = points[idx]
            s = _score(pts)
            if s > best_score:
                best_score = s
                best_pts = pts
    
    if best_pts is not None:
        return best_pts
    
    angles = np.random.uniform(0, 2 * np.pi, N)
    radii = np.sqrt(np.random.uniform(0, 0.9, N))
    return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])

def _ring_init():
    """Ring-based initialization: one center + rings of circles."""
    # Try different ring configurations
    best_pts = None
    best_score = -1.0
    
    configs = [
        (1, 6, 12, 18, 13),  # 1+6+12+18+13 = 50
        (1, 6, 12, 20, 11),
        (1, 6, 14, 18, 11),
        (1, 8, 12, 18, 11),
        (1, 6, 12, 16, 15),
        (1, 6, 14, 16, 13),
    ]
    
    for config in configs:
        pts = []
        pts.append([0.0, 0.0])
        ring_idx = 0
        for ring_size in config[1:]:
            ring_idx += 1
            # Radius of this ring
            r_ring = 0.13 * ring_idx
            for k in range(ring_size):
                angle = 2.0 * np.pi * k / ring_size + (0.3 if ring_idx % 2 == 0 else 0.0)
                x = r_ring * np.cos(angle)
                y = r_ring * np.sin(angle)
                pts.append([x, y])
        
        pts = np.array(pts[:N])
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_pts = pts
    
    if best_pts is not None:
        return best_pts
    return _hex_lattice_init()

def _random_init():
    angles = np.random.uniform(0, 2 * np.pi, N)
    radii = np.sqrt(np.random.uniform(0, 0.95, N))
    return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])

def _soft_min_obj_and_grad(flat, beta):
    """Soft-min objective for L-BFGS-B fallback."""
    pts = flat.reshape(N, 2)
    dists_origin = np.linalg.norm(pts, axis=1)
    boundary_dists = 1.0 - dists_origin
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_mat = np.linalg.norm(diff, axis=-1)
    iu = np.triu_indices(N, k=1)
    pair_dists = dist_mat[iu]
    all_dists = np.concatenate([boundary_dists, pair_dists / 2.0])
    y = -beta * all_dists
    max_y = np.max(y)
    y_shifted = y - max_y
    sum_exp = np.sum(np.exp(y_shifted))
    log_sum_exp = max_y + np.log(sum_exp)
    obj_val = log_sum_exp / beta
    weights = np.exp(y_shifted) / sum_exp
    grad_pts = np.zeros_like(pts)
    w_boundary = weights[0:N]
    safe_dists = np.where(dists_origin < 1e-8, 1.0, dists_origin)
    boundary_grad = (w_boundary / safe_dists)[:, np.newaxis] * pts
    grad_pts += boundary_grad
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
    return obj_val, grad_pts.flatten()

def _lbfgs_refine(pts, beta=500, maxiter=300):
    """L-BFGS-B refinement as fallback."""
    try:
        res = minimize(
            _soft_min_obj_and_grad,
            pts.flatten(),
            args=(beta,),
            method='L-BFGS-B',
            jac=True,
            bounds=[(-1.0, 1.0)] * (2 * N),
            options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-10, 'disp': False}
        )
        new_pts = res.x.reshape(N, 2)
        dists = np.linalg.norm(new_pts, axis=1)
        mask = dists > 1.0
        if np.any(mask):
            new_pts[mask] = new_pts[mask] / dists[mask, np.newaxis]
        return new_pts
    except Exception:
        return pts

def _full_optimize(pts, time_budget=20.0):
    """Full optimization pipeline: force-directed -> Newton -> L-BFGS polish."""
    start = time.time()
    
    # Step 1: Force-directed relaxation
    if time.time() - start < time_budget * 0.3:
        pts = _force_directed(pts, n_iters=150)
    
    # Step 2: Newton-Raphson with active-set
    if time.time() - start < time_budget * 0.6:
        r_est = _score(pts)
        contact_pairs, boundary_contacts = _identify_contacts(pts, r_est, tol=1e-3)
        pts, r_newton = _newton_solve(pts, contact_pairs, boundary_contacts, max_iter=100)
        s_newton = _score(pts)
    
    # Step 3: L-BFGS polish
    if time.time() - start < time_budget * 0.8:
        pts = _lbfgs_refine(pts, beta=1000, maxiter=200)
    
    # Step 4: Another Newton pass on refined solution
    if time.time() - start < time_budget * 0.95:
        r_est = _score(pts)
        contact_pairs, boundary_contacts = _identify_contacts(pts, r_est, tol=5e-4)
        pts, r_newton2 = _newton_solve(pts, contact_pairs, boundary_contacts, max_iter=100)
        s2 = _score(pts)
    
    return pts

def construct():
    start_time = time.time()
    
    # Generate multiple initial configurations
    inits = []
    inits.append(_hex_lattice_init())
    inits.append(_ring_init())
    for _ in range(3):
        inits.append(_random_init())
    
    best_pts = inits[0].copy()
    best_score = _score(best_pts)
    
    # Try each initialization with full optimization
    for init_pts in inits:
        if time.time() > _DEADLINE - 15.0:
            break
        
        # Time budget per initialization
        elapsed = time.time() - start_time
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
    if time.time() < _DEADLINE - 5.0:
        try:
            final = _lbfgs_refine(best_pts, beta=5000, maxiter=500)
            s_final = _score(final)
            if s_final > best_score:
                best_score = s_final
                best_pts = final
        except Exception:
            pass
    
    # One more Newton pass if time allows
    if time.time() < _DEADLINE - 3.0:
        try:
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

# EVOLVE-BLOCK-END
