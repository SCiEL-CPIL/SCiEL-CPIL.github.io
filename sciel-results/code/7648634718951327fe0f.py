import numpy as np
import time
from scipy.optimize import minimize

N = 50
START_TIME = time.time()
TIME_LIMIT = 140.0  # Leave 10s buffer for safety


def _calc_score(points):
    """Calculate the minimum radius for a given configuration."""
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair_dist = np.min(dist)
    
    radii = np.hypot(points[:, 0], points[:, 1])
    min_boundary_dist = np.min(1.0 - radii)
    
    return min(min_pair_dist / 2.0, min_boundary_dist)


def _ring_init(r_est=0.125):
    """Generate ring-based initial configuration matching known optimal structure.
    
    Structure: 1 center + 5 ring1 + 13 ring2 + 31 ring3 = 50
    """
    centers = np.zeros((N, 2))
    
    # Center point
    centers[0] = [0.0, 0.0]
    
    # Ring 1: 5 points
    R1 = r_est / np.sin(np.pi / 5)
    for i in range(5):
        angle = 2 * np.pi * i / 5 + np.pi / 5
        centers[1 + i] = [R1 * np.cos(angle), R1 * np.sin(angle)]
    
    # Ring 2: 13 points
    R2 = R1 + 1.5 * r_est
    for i in range(13):
        angle = 2 * np.pi * i / 13
        centers[6 + i] = [R2 * np.cos(angle), R2 * np.sin(angle)]
    
    # Ring 3: 31 points
    R3 = r_est / np.sin(np.pi / 31)
    R3 = min(R3, 1.0 - r_est)
    for i in range(31):
        angle = 2 * np.pi * i / 31 + np.pi / 31
        centers[19 + i] = [R3 * np.cos(angle), R3 * np.sin(angle)]
    
    return centers


def _hex_lattice_init(spacing, offset_x=0.0, offset_y=0.0):
    """Generate points on a hexagonal lattice inside the unit circle."""
    points = []
    k = int(1.0 / spacing) + 3
    for i in range(-k, k + 1):
        for j in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing + offset_x
            y = j * spacing * np.sqrt(3.0) / 2.0 + offset_y
            if x*x + y*y <= 1.0:
                points.append([x, y])
    points = np.array(points)
    if len(points) < N:
        return None
    radii = np.hypot(points[:, 0], points[:, 1])
    indices = np.argsort(radii)[:N]
    return points[indices]


def _softmin_obj_grad(x, beta):
    """
    Objective and gradient for maximizing the soft-min of constraints.
    """
    C = x.reshape(N, 2)
    
    norms = np.linalg.norm(C, axis=1)
    safe_norms = np.where(norms > 1e-10, norms, 1.0)
    boundary_vals = 1.0 - norms
    
    diff = C[:, np.newaxis, :] - C[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    mask = ~np.eye(N, dtype=bool)
    pair_vals = dists[mask] / 2.0
    
    all_vals = np.concatenate([boundary_vals, pair_vals])
    
    min_val = np.min(all_vals)
    exp_args = -beta * (all_vals - min_val)
    exp_vals = np.exp(exp_args)
    sum_exp = np.sum(exp_vals)
    
    soft_min = min_val - np.log(sum_exp) / beta
    obj = -soft_min
    
    weights = exp_vals / sum_exp
    
    grad = np.zeros_like(C)
    
    w_boundary = weights[:N]
    boundary_grad = -C / safe_norms[:, np.newaxis]
    grad -= w_boundary[:, np.newaxis] * boundary_grad
    
    idx_i, idx_j = np.where(mask)
    w_pair_vals = weights[N:]
    
    W = np.zeros((N, N))
    W[idx_i, idx_j] = w_pair_vals
    
    safe_dists = np.where(dists > 1e-10, dists, 1.0)
    unit_vecs = diff / safe_dists[:, :, np.newaxis]
    
    pair_grad = -np.sum(W[:, :, np.newaxis] * unit_vecs, axis=1)
    
    grad += pair_grad
    
    return obj, grad.flatten()


def _physics_relax(C, iterations=300, dt=0.015):
    """
    Physics-based relaxation using repulsive forces.
    """
    pts = C.copy()
    
    for _ in range(iterations):
        if time.time() - START_TIME > TIME_LIMIT - 15:
            break
            
        forces = np.zeros_like(pts)
        
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, 1.0)
        
        mask = (dists < 0.5) & (dists > 1e-6)
        
        safe_dists = np.where(dists > 1e-6, dists, 1.0)
        inv_dist_sq = np.zeros_like(dists)
        inv_dist_sq[mask] = 1.0 / (dists[mask] ** 2)
        
        forces += np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
        
        norms = np.linalg.norm(pts, axis=1)
        boundary_mask = norms > 0.7
        if np.any(boundary_mask):
            k = 3.0
            boundary_forces = -k * (norms - 0.7)[:, np.newaxis] * pts / norms[:, np.newaxis]
            forces[boundary_mask] += boundary_forces[boundary_mask]
        
        pts += dt * forces
        
        norms = np.linalg.norm(pts, axis=1)
        out_mask = norms > 1.0
        if np.any(out_mask):
            pts[out_mask] = pts[out_mask] / norms[out_mask, np.newaxis]
        
        pts -= np.mean(pts, axis=0)
        
        norms = np.linalg.norm(pts, axis=1)
        out_mask = norms > 1.0
        if np.any(out_mask):
            pts[out_mask] = pts[out_mask] / norms[out_mask, np.newaxis]
            
    return pts


def _optimize(points, max_steps=3000, initial_step=0.01):
    """Gradient ascent with random restarts to escape local minima."""
    step = initial_step
    best_score = _calc_score(points)
    best_points = points.copy()
    no_improve_count = 0
    
    for s in range(max_steps):
        if time.time() - START_TIME > TIME_LIMIT - 10:
            break
            
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dist, np.inf)
        
        radii = np.hypot(points[:, 0], points[:, 1])
        boundary_dist = 1.0 - radii
        
        min_pair_dist = np.min(dist)
        min_boundary_dist = np.min(boundary_dist)
        
        current_score = min(min_pair_dist / 2.0, min_boundary_dist)
        
        if current_score > best_score + 1e-10:
            best_score = current_score
            best_points = points.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count > 150:
                break
        
        if min_pair_dist / 2.0 < min_boundary_dist:
            idx_i, idx_j = np.unravel_index(np.argmin(dist), dist.shape)
            if idx_i > idx_j:
                idx_i, idx_j = idx_j, idx_i
            
            vec = points[idx_i] - points[idx_j]
            norm = np.linalg.norm(vec)
            if norm > 1e-9:
                vec /= norm
            else:
                vec = np.array([1.0, 0.0])
            
            points[idx_i] += step * vec
            points[idx_j] -= step * vec
        else:
            idx_i = np.argmin(boundary_dist)
            vec = -points[idx_i]
            norm = np.linalg.norm(vec)
            if norm > 1e-9:
                vec /= norm
            else:
                vec = np.array([1.0, 0.0])
            
            points[idx_i] += step * vec
        
        if s % 40 == 0 and s > 0:
            points += np.random.randn(N, 2) * step * 0.1
        
        step *= 0.995
        
    return best_points


def _scipy_optimize(C, beta_seq=None):
    """Run scipy minimize with increasing beta for soft-min optimization."""
    if beta_seq is None:
        beta_seq = [20, 50, 100, 200, 500]
    
    current_C = C.copy()
    
    for beta in beta_seq:
        if time.time() - START_TIME > TIME_LIMIT - 10:
            break
        
        def fun(x):
            return _softmin_obj_grad(x, beta)[0]
        
        def jac(x):
            return _softmin_obj_grad(x, beta)[1]
        
        try:
            res = minimize(fun, current_C.flatten(), jac=jac, method='L-BFGS-B',
                          options={'maxiter': 200, 'gtol': 1e-8})
            if res.success:
                new_C = res.x.reshape(N, 2)
                if _calc_score(new_C) > _calc_score(current_C):
                    current_C = new_C
        except Exception:
            pass
    
    return current_C


def construct():
    best_points = None
    best_score = -1
    
    # Strategy 1: Ring-based initialization (matches known optimal structure)
    for r_est in [0.125, 0.126, 0.124, 0.127, 0.123]:
        if time.time() - START_TIME > TIME_LIMIT - 20:
            break
            
        points = _ring_init(r_est)
        score_init = _calc_score(points)
        
        if score_init > best_score:
            best_score = score_init
            best_points = points.copy()
        
        # Physics relaxation first
        if time.time() - START_TIME < TIME_LIMIT - 25:
            relaxed = _physics_relax(points.copy(), iterations=200, dt=0.015)
            score_relax = _calc_score(relaxed)
            if score_relax > best_score:
                best_score = score_relax
                best_points = relaxed.copy()
                points = relaxed
        
        # Gradient optimization
        if time.time() - START_TIME < TIME_LIMIT - 15:
            optimized = _optimize(points.copy(), max_steps=2000, initial_step=0.01)
            score_opt = _calc_score(optimized)
            if score_opt > best_score:
                best_score = score_opt
                best_points = optimized.copy()
        
        # Scipy soft-min refinement
        if time.time() - START_TIME < TIME_LIMIT - 10:
            refined = _scipy_optimize(best_points.copy(), beta_seq=[50, 100, 200])
            score_refined = _calc_score(refined)
            if score_refined > best_score:
                best_score = score_refined
                best_points = refined.copy()
    
    # Strategy 2: Hex lattice initializations
    spacings = [0.24, 0.25, 0.26, 0.23]
    offsets = [(0, 0), (0.01, 0.01), (-0.01, 0.01)]
    
    for s in spacings:
        for ox, oy in offsets:
            if time.time() - START_TIME > TIME_LIMIT - 15:
                break
                
            points = _hex_lattice_init(s, ox, oy)
            if points is None or len(points) != N:
                continue
            
            score_init = _calc_score(points)
            if score_init > best_score:
                best_score = score_init
                best_points = points.copy()
            
            if time.time() - START_TIME < TIME_LIMIT - 10:
                relaxed = _physics_relax(points.copy(), iterations=150, dt=0.012)
                score_relax = _calc_score(relaxed)
                if score_relax > best_score:
                    best_score = score_relax
                    best_points = relaxed.copy()
                    points = relaxed
            
            if time.time() - START_TIME < TIME_LIMIT - 8:
                optimized = _optimize(points.copy(), max_steps=1500, initial_step=0.008)
                score_opt = _calc_score(optimized)
                if score_opt > best_score:
                    best_score = score_opt
                    best_points = optimized.copy()
    
    # Final polish: small steps on best configuration
    if best_points is not None and time.time() - START_TIME < TIME_LIMIT - 5:
        polished = _optimize(best_points.copy(), max_steps=800, initial_step=0.003)
        score = _calc_score(polished)
        if score > best_score:
            best_points = polished
    
    # Ensure validity: clip points to be strictly inside unit circle
    if best_points is not None:
        radii = np.hypot(best_points[:, 0], best_points[:, 1])
        for i in range(N):
            if radii[i] >= 1.0:
                scale = 0.999 / radii[i] if radii[i] > 0 else 1.0
                best_points[i] *= scale
    
    return best_points
