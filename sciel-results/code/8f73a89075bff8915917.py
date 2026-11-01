"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 32
import numpy as np
from scipy.optimize import minimize
import time

# EVOLVE-BLOCK-START

def _get_packing_from_positions(p):
    """
    Given center positions p (N,2), compute maximum valid radii.
    r_i = min(dist_to_boundary_i, min_{j!=i} 0.5 * dist(i,j))
    """
    p = np.asarray(p, dtype=np.float64).reshape(N, 2).copy()
    p[:, 0] = np.clip(p[:, 0], 0.0, 1.0)
    p[:, 1] = np.clip(p[:, 1], 0.0, 1.0)
    
    diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dist, np.inf)
    
    min_dist_to_others = np.min(dist, axis=1)
    dist_to_box = np.min(np.stack([p[:, 0], 1.0 - p[:, 0], p[:, 1], 1.0 - p[:, 1]], axis=1), axis=1)
    
    radii = np.minimum(dist_to_box, 0.5 * min_dist_to_others)
    radii = np.maximum(radii, 0.0)
    
    # Safety factor for strict non-overlap
    radii *= 0.99999
    
    # Ensure centers are within bounds considering radius
    p[:, 0] = np.clip(p[:, 0], radii, 1.0 - radii)
    p[:, 1] = np.clip(p[:, 1], radii, 1.0 - radii)
    
    return p, radii

def _sum_radii(pos):
    """Objective: maximize sum of radii (minimize negative)."""
    p = pos.reshape(N, 2)
    diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dist, np.inf)
    
    min_dist_to_others = np.min(dist, axis=1)
    dist_to_box = np.min(np.stack([p[:, 0], 1.0 - p[:, 0], p[:, 1], 1.0 - p[:, 1]], axis=1), axis=1)
    
    radii = np.minimum(dist_to_box, 0.5 * min_dist_to_others)
    radii = np.maximum(radii, 0.0)
    
    return -np.sum(radii)

def _grad_sum_radii(pos):
    """
    Analytical gradient of -sum(radii) with respect to positions.
    Uses subgradient of min: gradient flows from the minimizing constraint(s).
    """
    p = pos.reshape(N, 2).copy()
    grad = np.zeros((N, 2), dtype=np.float64)
    
    diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=-1) + 1e-20)
    np.fill_diagonal(dist, np.inf)
    
    min_dist_to_others = np.min(dist, axis=1)
    dist_to_box = np.min(np.stack([p[:, 0], 1.0 - p[:, 0], p[:, 1], 1.0 - p[:, 1]], axis=1), axis=1)
    
    # For each circle, find which constraint is active (minimizing)
    for i in range(N):
        # Determine active constraint
        d_box = dist_to_box[i]
        d_others = 0.5 * min_dist_to_others[i]
        
        # Subgradient: average over all nearly-active constraints
        eps = 1e-8
        active_box = abs(d_box - d_others) < eps
        active_others = abs(d_others - d_box) < eps
        
        if active_box and active_others:
            # Both active: average gradients
            w_box = 0.5
            w_others = 0.5
        elif d_box <= d_others:
            w_box = 1.0
            w_others = 0.0
        else:
            w_box = 0.0
            w_others = 1.0
        
        # Gradient contribution from box constraint (if active)
        if w_box > 0:
            # r_i = min(x_i, 1-x_i, y_i, 1-y_i)
            # grad of x_i wrt p[i,0] = 1, wrt p[i,1] = 0
            # grad of (1-x_i) wrt p[i,0] = -1
            # etc.
            if p[i, 0] < 0.5 and p[i, 1] < 0.5:
                grad[i, 0] += w_box * 1.0
                grad[i, 1] += w_box * 1.0
            elif p[i, 0] < 0.5:
                grad[i, 0] += w_box * 1.0
                grad[i, 1] += w_box * (-1.0)
            elif p[i, 1] < 0.5:
                grad[i, 0] += w_box * (-1.0)
                grad[i, 1] += w_box * 1.0
            else:
                grad[i, 0] += w_box * (-1.0)
                grad[i, 1] += w_box * (-1.0)
        
        # Gradient contribution from neighbor constraint (if active)
        if w_others > 0:
            # r_i = 0.5 * min_j dist(i,j)
            # Find all j that are nearly at minimum distance
            min_d = min_dist_to_others[i]
            for j in range(N):
                if i == j:
                    continue
                if abs(dist[i, j] - min_d) < eps:
                    # grad of 0.5*dist(i,j) wrt p[i] = 0.5 * (p[i]-p[j]) / dist
                    d = p[i] - p[j]
                    grad[i] += w_others * 0.5 * d / dist[i, j]
                    # Also affects r_j through the same constraint
                    # But we're computing grad of -sum(r_k), so:
                    # d(-r_j)/dp[i] includes the effect on r_j via dist(i,j)
                    # r_j = 0.5 * min_k dist(j,k), if k=i is active for j
                    # This creates coupling. For simplicity, use only direct gradient.
    
    return grad.flatten()

def _hex_initialization():
    """Generate a hexagonal grid initialization."""
    pts = []
    for i in range(6):
        for j in range(6):
            x = (i + 0.5) / 6.0
            y = (j + 0.5) / 6.0
            if i % 2 == 0:
                y += 0.5 / 6.0 * 0.8
            if 0.0 < x < 1.0 and 0.0 < y < 1.0:
                pts.append([x, y])
    
    if len(pts) < N:
        while len(pts) < N:
            pts.append([np.random.rand(), np.random.rand()])
            
    return np.array(pts[:N]).flatten()

def _grid_initialization(spacing=1.0/6.0):
    """Generate a regular grid initialization."""
    pts = []
    n_per_side = int(1.0 / spacing) + 1
    for i in range(n_per_side):
        for j in range(n_per_side):
            x = (i + 0.5) * spacing
            y = (j + 0.5) * spacing
            if 0.0 < x < 1.0 and 0.0 < y < 1.0:
                pts.append([x, y])
    
    # Shuffle and take N
    np.random.shuffle(pts)
    if len(pts) < N:
        while len(pts) < N:
            pts.append([np.random.rand(), np.random.rand()])
    return np.array(pts[:N]).flatten()

def _physics_preprocess(pos, n_iters=80):
    """
    Physics-based push-apart to spread out initial points.
    """
    p = pos.reshape(N, 2).copy()
    target_dist = 1.0 / np.sqrt(N) * 0.9
    for _ in range(n_iters):
        forces = np.zeros((N, 2), dtype=np.float64)
        for i in range(N):
            for j in range(i + 1, N):
                d = p[i] - p[j]
                dist = np.sqrt(np.sum(d ** 2))
                if dist < target_dist:
                    if dist > 1e-10:
                        f = 0.3 * (target_dist - dist) * d / dist
                    else:
                        f = np.array([0.01, 0.01])
                    forces[i] += f
                    forces[j] -= f
        p += 0.1 * forces
        p[:, 0] = np.clip(p[:, 0], 0.01, 0.99)
        p[:, 1] = np.clip(p[:, 1], 0.01, 0.99)
    return p.flatten()

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_score = -np.inf
    best_result = None
    
    # Generate diverse initializations
    inits = []
    inits.append(_hex_initialization())
    inits.append(_grid_initialization(1.0/6.0))
    inits.append(_grid_initialization(1.0/5.0))
    for _ in range(4):
        inits.append(np.random.rand(2 * N))
    
    # Preprocess with physics
    for idx in range(len(inits)):
        if time.time() > deadline - 60.0:
            break
        inits[idx] = _physics_preprocess(inits[idx], n_iters=50)
    
    # Try SLSQP with explicit constraints for better results
    def _slsqp_objective(pos):
        return _sum_radii(pos)
    
    def _slsqp_gradient(pos):
        return _grad_sum_radii(pos)
    
    # Bounds for SLSQP
    bounds = [(0.0, 1.0)] * (2 * N)
    
    # Constraints: ensure circles don't overlap and stay in box
    # We'll use inequality constraints: dist(i,j) >= r_i + r_j
    # But since r depends on positions, this is complex.
    # Instead, use a penalty approach or rely on the objective.
    
    for init_pos in inits:
        if time.time() > deadline - 30.0:
            break
        
        # Try L-BFGS-B with analytical gradient
        try:
            res = minimize(_sum_radii, init_pos, jac=_grad_sum_radii,
                          method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-10, 'maxls': 50})
            
            p, radii = _get_packing_from_positions(res.x)
            score = np.sum(radii)
            
            if score > best_score:
                best_score = score
                best_result = np.column_stack((p, radii))
                
        except Exception:
            pass
        
        # Try SLSQP
        if time.time() > deadline - 20.0:
            continue
        
        try:
            res_slsqp = minimize(_slsqp_objective, init_pos, jac=_slsqp_gradient,
                                 method='SLSQP', bounds=bounds,
                                 options={'maxiter': 300, 'ftol': 1e-12})
            
            p, radii = _get_packing_from_positions(res_slsqp.x)
            score = np.sum(radii)
            
            if score > best_score:
                best_score = score
                best_result = np.column_stack((p, radii))
                
        except Exception:
            pass
    
    # Final high-precision polish
    if best_result is not None and time.time() < deadline - 5.0:
        try:
            start_pos = best_result[:, :2].flatten()
            res_final = minimize(_sum_radii, start_pos, jac=_grad_sum_radii,
                                method='L-BFGS-B', bounds=bounds,
                                options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-12, 'maxls': 100})
            
            p_final, radii_final = _get_packing_from_positions(res_final.x)
            score_final = np.sum(radii_final)
            
            if score_final > best_score:
                best_score = score_final
                best_result = np.column_stack((p_final, radii_final))
                
        except Exception:
            pass
    
    # Fallback
    if best_result is None:
        p, radii = _get_packing_from_positions(inits[0])
        best_result = np.column_stack((p, radii))
        
    return best_result

# EVOLVE-BLOCK-END
