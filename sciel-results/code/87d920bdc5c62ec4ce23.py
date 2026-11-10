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
    # Estimate range
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
    # Select the N points closest to the origin to form a compact cluster
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
    # Avoid division by zero in gradient later
    safe_norms = np.where(norms > 1e-10, norms, 1.0)
    boundary_vals = 1.0 - norms
    
    # 2. Pairwise constraints
    diff = C[:, np.newaxis, :] - C[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    # Mask diagonal
    mask = ~np.eye(N, dtype=bool)
    pair_vals = dists[mask] / 2.0
    
    all_vals = np.concatenate([boundary_vals, pair_vals])
    
    # Soft min calculation: softmin(v) = min(v) - log(sum(exp(-beta*(v - min(v)))))/beta
    # This form is numerically stable.
    min_val = np.min(all_vals)
    exp_args = -beta * (all_vals - min_val)
    # Clip to avoid underflow/overflow issues if beta is very large, though shifted it should be ok
    exp_vals = np.exp(exp_args)
    sum_exp = np.sum(exp_vals)
    
    soft_min = min_val - np.log(sum_exp) / beta
    obj = -soft_min
    
    # Gradient
    # d(softmin)/dv_k = exp(-beta*(v_k - min_val)) / sum_exp
    weights = exp_vals / sum_exp
    
    grad = np.zeros_like(C)
    
    # Boundary part
    # d(boundary_val_i)/d(c_i) = -c_i / ||c_i||
    # Contribution to grad_obj (which is -softmin):
    # grad_obj += -weight_i * d(softmin)/d(boundary_val_i) * d(boundary_val_i)/d(c_i)
    # Wait, d(softmin)/d(v_k) IS the weight w_k.
    # So d(softmin)/d(c_i) = sum_k w_k * d(v_k)/d(c_i).
    # d(obj)/d(c_i) = - d(softmin)/d(c_i) = - sum_k w_k * d(v_k)/d(c_i).
    
    w_boundary = weights[:N]
    # d(boundary_val_i)/d(c_i) = -c_i / ||c_i||
    boundary_grad = -C / safe_norms[:, np.newaxis]
    
    # grad_obj += - (w_boundary * boundary_grad) summed appropriately?
    # Actually, grad_obj[c_i] += - w_boundary[i] * boundary_grad[i]
    grad -= w_boundary[:, np.newaxis] * boundary_grad
    
    # Pairwise part
    # We need to map weights back to the (i, j) pairs.
    # mask is (N, N). pair_vals is flattened from mask.
    idx_i, idx_j = np.where(mask)
    w_pair_vals = weights[N:]
    
    # Create a weight matrix W where W[i, j] = w_pair for pair (i, j)
    # Note: dists is symmetric, so pair_vals is symmetric. Weights are symmetric.
    # W[i, j] = w_{ij}
    W = np.zeros((N, N))
    W[idx_i, idx_j] = w_pair_vals
    
    # d(pair_val_ij)/d(c_i) = (c_i - c_j) / ||c_i - c_j||
    # d(pair_val_ij)/d(c_j) = (c_j - c_i) / ||c_i - c_j||
    
    # We want: grad_obj[c_i] += - sum_j W[i, j] * d(pair_val_ij)/d(c_i)
    
    # Vectorized:
    # diff is (N, N, 2)
    # dists is (N, N)
    safe_dists = np.where(dists > 1e-10, dists, 1.0)
    # unit_vec[i, j] = (c_i - c_j) / dist[i, j]
    unit_vecs = diff / safe_dists[:, :, np.newaxis]
    
    # grad_pair[i] = - sum_j W[i, j] * unit_vec[i, j]
    # This is a weighted sum of vectors.
    # W[:, :, np.newaxis] * unit_vecs -> (N, N, 2)
    # Sum over axis 1 (j) -> (N, 2)
    pair_grad = -np.sum(W[:, :, np.newaxis] * unit_vecs, axis=1)
    
    grad += pair_grad
    
    return obj, grad.flatten()

def construct():
    if time.time() - START_TIME > TIME_LIMIT:
        # Fallback if somehow called late
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
    
    # If not enough, add random
    if len(inits) < 3:
        for _ in range(5):
            pts = np.random.uniform(-0.8, 0.8, (N, 2))
            # Project to unit circle roughly
            norms = np.linalg.norm(pts, axis=1)
            mask = norms > 0.9
            pts[mask] /= norms[mask, np.newaxis] * 0.95
            inits.append(pts)

    # Optimization Strategy:
    # For each init, try a sequence of betas.
    # Low beta: smooth, finds basin.
    # High beta: sharp, refines.
    
    beta_sequence = [20, 50, 100, 200, 500]
    
    for init in inits:
        if time.time() - START_TIME > TIME_LIMIT - 10:
            break
            
        C = init.copy()
        current_score = _calc_score(C)
        
        if current_score > best_score:
            best_score = current_score
            best_C = C.copy()
            
        # Run optimization chain
        for beta in beta_sequence:
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
                    options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 20}
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
                    # If no improvement, break early for this init?
                    # Or continue to see if a higher beta helps from the same spot?
                    # Usually if low beta doesn't improve, higher beta won't either unless stuck.
                    # But sometimes it helps escape flat regions.
                    # Let's allow one more step if we haven't tried the highest beta yet.
                    pass
                    
            except Exception:
                pass

    # Final polish on the best result found
    if best_C is not None and time.time() - START_TIME < TIME_LIMIT - 5:
        C = best_C.copy()
        # Very high beta for final precision
        for beta in [1000, 2000]:
            if time.time() - START_TIME > TIME_LIMIT - 2:
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
        # Absolute fallback
        best_C = _hex_lattice_init(0.25)
        
    return best_C

# EVOLVE-BLOCK-END
