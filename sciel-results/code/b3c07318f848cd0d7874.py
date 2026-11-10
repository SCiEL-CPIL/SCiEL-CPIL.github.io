"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

START_TIME = time.time()
TIME_LIMIT = 140  # Safety margin from 150s limit

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

def _calc_score(C):
    """Calculate the effective radius for a given set of centers."""
    C = np.asarray(C).reshape(-1, 2)
    if C.shape[0] != N:
        return 0.0
    if not np.all(np.isfinite(C)):
        return 0.0
    
    norms = np.linalg.norm(C, axis=1)
    r_bound = 1.0 - np.max(norms)
    
    diff = C[:, np.newaxis, :] - C[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    r_pair = min_dist / 2.0
    
    return min(r_bound, r_pair)

def _softmin_obj_grad(x, beta):
    """
    Objective and gradient for maximizing the soft-min of constraints.
    Constraints:
    1. Boundary: 1 - ||c_i||
    2. Pairwise: ||c_i - c_j|| / 2
    
    We minimize -softmin(constraints).
    """
    C = x.reshape(N, 2)
    
    # 1. Boundary constraints
    norms = np.linalg.norm(C, axis=1)
    safe_norms = np.where(norms > 1e-10, norms, 1.0)
    boundary_vals = 1.0 - norms
    
    # 2. Pairwise constraints
    diff = C[:, np.newaxis, :] - C[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    mask = ~np.eye(N, dtype=bool)
    pair_vals = dists[mask] / 2.0
    
    all_vals = np.concatenate([boundary_vals, pair_vals])
    
    # Soft min calculation
    min_val = np.min(all_vals)
    exp_args = -beta * (all_vals - min_val)
    exp_vals = np.exp(exp_args)
    sum_exp = np.sum(exp_vals)
    
    soft_min = min_val - np.log(sum_exp) / beta
    obj = -soft_min
    
    # Gradient
    weights = exp_vals / sum_exp
    
    grad = np.zeros_like(C)
    
    # Boundary part
    w_boundary = weights[:N]
    boundary_grad = -C / safe_norms[:, np.newaxis]
    grad -= w_boundary[:, np.newaxis] * boundary_grad
    
    # Pairwise part
    idx_i, idx_j = np.where(mask)
    w_pair_vals = weights[N:]
    
    W = np.zeros((N, N))
    W[idx_i, idx_j] = w_pair_vals
    
    safe_dists = np.where(dists > 1e-10, dists, 1.0)
    unit_vecs = diff / safe_dists[:, :, np.newaxis]
    
    pair_grad = -np.sum(W[:, :, np.newaxis] * unit_vecs, axis=1)
    
    grad += pair_grad
    
    return obj, grad.flatten()

def _physics_relax(C, iterations=500, dt=0.02):
    """
    Physics-based relaxation using repulsive forces.
    Helps escape local minima by pushing overlapping or close circles apart.
    """
    pts = C.copy()
    
    for _ in range(iterations):
        if time.time() - START_TIME > TIME_LIMIT - 10:
            break
            
        forces = np.zeros_like(pts)
        
        # Pairwise repulsion
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, 1.0)  # Avoid division by zero
        
        # Only apply force if distance is less than some threshold
        # Use a soft repulsion: F = 1 / dist^2
        # But we want to maximize the minimum distance, so we push apart
        # any pair that is closer than the current minimum + epsilon
        
        # Vectorized force calculation
        # F_ij = (p_i - p_j) / dist^3  (inverse square law direction)
        # We want to push them apart, so force on i is in direction (p_i - p_j)
        
        # Only consider pairs with dist < 1.0 to save computation
        mask = (dists < 1.0) & (dists > 1e-6)
        
        # Compute forces
        # f = diff / dist^3
        safe_dists = np.where(dists > 1e-6, dists, 1.0)
        inv_dist_sq = np.zeros_like(dists)
        inv_dist_sq[mask] = 1.0 / (dists[mask] ** 2)
        
        # Force on i from j is in direction (p_i - p_j) * inv_dist^2
        # F_i = sum_j (p_i - p_j) * inv_dist_ij^2
        # Vectorized: sum over j of diff[i,j] * inv_dist_sq[i,j]
        
        forces += np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
        
        # Boundary force: push points inward if they are close to boundary
        norms = np.linalg.norm(pts, axis=1)
        boundary_mask = norms > 0.8
        if np.any(boundary_mask):
            # Force magnitude proportional to how close to boundary
            # F = -k * (norm - 0.8) * (p / norm)
            k = 5.0
            boundary_forces = -k * (norms - 0.8)[:, np.newaxis] * pts / norms[:, np.newaxis]
            forces[boundary_mask] += boundary_forces[boundary_mask]
        
        # Update positions
        pts += dt * forces
        
        # Enforce hard boundary
        norms = np.linalg.norm(pts, axis=1)
        out_mask = norms > 1.0
        if np.any(out_mask):
            pts[out_mask] = pts[out_mask] / norms[out_mask, np.newaxis]
            
        # Re-center to prevent drift
        pts -= np.mean(pts, axis=0)
        
        # Re-enforce boundary after re-centering
        norms = np.linalg.norm(pts, axis=1)
        out_mask = norms > 1.0
        if np.any(out_mask):
            pts[out_mask] = pts[out_mask] / norms[out_mask, np.newaxis]
            
    return pts

def construct():
    if time.time() - START_TIME > TIME_LIMIT:
        return _hex_lattice_init(0.25)
    
    best_C = None
    best_score = 0.0
    
    # Generate diverse initializations
    inits = []
    spacings = [0.24, 0.25, 0.26, 0.23, 0.27]
    offsets = [(0,0), (0.01, 0.01), (-0.01, 0.01), (0.01, -0.01), (-0.01, -0.01)]
    
    for s in spacings:
        for ox, oy in offsets:
            pts = _hex_lattice_init(s, ox, oy)
            if pts is not None and len(pts) == N:
                inits.append(pts)
    
    # Add some random initializations
    np.random.seed(42)
    for _ in range(3):
        pts = np.random.uniform(-0.8, 0.8, (N, 2))
        norms = np.linalg.norm(pts, axis=1)
        mask = norms > 0.9
        if np.any(mask):
            pts[mask] /= norms[mask, np.newaxis] * 0.95
        inits.append(pts)

    # Optimization Strategy:
    # For each init, try a sequence of betas.
    # Low beta: smooth, finds basin.
    # High beta: sharp, refines.
    
    beta_sequence = [20, 50, 100, 200, 500]
    
    for init in inits:
        if time.time() - START_TIME > TIME_LIMIT - 15:
            break
            
        C = init.copy()
        current_score = _calc_score(C)
        
        if current_score > best_score:
            best_score = current_score
            best_C = C.copy()
            
        # Run physics relaxation first to get a better starting point
        if time.time() - START_TIME < TIME_LIMIT - 20:
            C_phys = _physics_relax(C, iterations=200, dt=0.02)
            score_phys = _calc_score(C_phys)
            if score_phys > current_score:
                C = C_phys
                current_score = score_phys
                if current_score > best_score:
                    best_score = current_score
                    best_C = C.copy()
        
        # Run optimization chain
        for beta in beta_sequence:
            if time.time() - START_TIME > TIME_LIMIT - 10:
                break
                
            def fun(x):
                return _softmin_obj_grad(x, beta)[0]
            
            def jac(x):
                return _softmin_obj_grad(x, beta)[1]
            
            try:
                res = minimize(
                    fun, 
                    C.flatten(), 
                    jac=jac, 
                    method='L-BFGS-B', 
                    options={'maxiter': 1500, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 20}
                )
                
                C_new = res.x.reshape(N, 2)
                score_new = _calc_score(C_new)
                
                if score_new > current_score:
                    C = C_new
                    current_score = score_new
                    
                    if current_score > best_score:
                        best_score = current_score
                        best_C = C.copy()
                else:
                    pass
                    
            except Exception:
                pass

    # Final polish on the best result found
    if best_C is not None and time.time() - START_TIME < TIME_LIMIT - 10:
        C = best_C.copy()
        
        # Try physics relaxation again on the best result
        if time.time() - START_TIME < TIME_LIMIT - 15:
            C_phys = _physics_relax(C, iterations=300, dt=0.01)
            score_phys = _calc_score(C_phys)
            if score_phys > best_score:
                C = C_phys
                best_score = score_phys
                best_C = C.copy()
        
        # Very high beta for final precision
        for beta in [1000, 2000]:
            if time.time() - START_TIME > TIME_LIMIT - 5:
                break
            def fun(x):
                return _softmin_obj_grad(x, beta)[0]
            def jac(x):
                return _softmin_obj_grad(x, beta)[1]
            try:
                res = minimize(
                    fun, 
                    C.flatten(), 
                    jac=jac, 
                    method='L-BFGS-B', 
                    options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-10}
                )
                C_new = res.x.reshape(N, 2)
                score_new = _calc_score(C_new)
                if score_new > best_score:
                    best_score = score_new
                    best_C = C_new.copy()
            except Exception:
                pass

    if best_C is None:
        best_C = _hex_lattice_init(0.25)
        
    return best_C

# EVOLVE-BLOCK-END
