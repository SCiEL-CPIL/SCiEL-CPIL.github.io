"""
Circle packing in perimeter-4 rectangle.
Combines multi-start L-BFGS optimization with simultaneous aspect ratio adjustment.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 21

def _check_valid(circles, W, H, tol=1e-7):
    """
    Check if circles are valid: disjoint, inside [0,W]x[0,H].
    circles: (N, 3) array of [x, y, r]
    """
    if circles.shape[0] != N:
        return False
    xs, ys, rs = circles[:, 0], circles[:, 1], circles[:, 2]
    
    # Check radii
    if np.any(rs < -1e-9):
        return False
    rs = np.maximum(rs, 0)
    
    # Check boundaries
    if np.any(xs - rs < -tol) or np.any(xs + rs > W + tol) or \
       np.any(ys - rs < -tol) or np.any(ys + rs > H + tol):
        return False
        
    # Check pairwise separation
    for i in range(N):
        for j in range(i + 1, N):
            d = np.sqrt((xs[i] - xs[j])**2 + (ys[i] - ys[j])**2)
            if d < rs[i] + rs[j] - tol:
                return False
    return True

def _evaluate_state(p, penalty_weight=1000.0):
    """
    Evaluate the state vector p.
    p[0] = t (aspect ratio parameter, W=2t, H=2(1-t))
    p[1:1+3N] = flattened [x0, y0, r0, x1, y1, r1, ...]
    
    Returns:
        objective: -sum(r) + penalties (to be minimized)
        gradient: gradient of the objective
    """
    t = p[0]
    # Clip t to ensure valid rectangle dimensions
    t = np.clip(t, 0.1, 0.9)
    W = 2.0 * t
    H = 2.0 * (1.0 - t)
    
    # Unpack circles
    coords = p[1:1+3*N].reshape(N, 3)
    xs, ys, rs = coords[:, 0], coords[:, 1], coords[:, 2]
    
    # Ensure radii are positive for calculation
    rs_eff = np.maximum(rs, 1e-6)
    
    obj = -np.sum(rs_eff)
    grad = np.zeros_like(p)
    
    # Gradient for t: d(-sum(r))/dt = 0 (radii don't depend directly on t in this formulation, 
    # but constraints do. The penalty gradients will handle the t dependence)
    grad[0] = 0.0 
    
    # Gradient for radii: d(-sum(r))/dr = -1
    grad[3:3+3*N:3] -= 1.0
    
    # --- Boundary Penalties ---
    # Violations:
    # Left:   x_i - r_i < 0  -> viol = r_i - x_i
    # Right:  x_i + r_i > W  -> viol = x_i + r_i - W
    # Bottom: y_i - r_i < 0  -> viol = r_i - y_i
    # Top:    y_i + r_i > H  -> viol = y_i + r_i - H
    
    eps = 1e-4
    def softplus(v):
        return eps * np.log1p(np.exp(v / eps))
    def sigmoid(v):
        return 1.0 / (1.0 + np.exp(-v / eps))
        
    # Left
    v_l = rs_eff - xs
    sp_l = softplus(v_l)
    obj += penalty_weight * np.sum(sp_l)
    sig_l = sigmoid(v_l)
    # d/dx_i = -penalty_weight * sig_l
    # d/dr_i = penalty_weight * sig_l
    grad[1:1+3*N:3] -= penalty_weight * sig_l
    grad[3:3+3*N:3] += penalty_weight * sig_l
    
    # Right
    v_r = xs + rs_eff - W
    sp_r = softplus(v_r)
    obj += penalty_weight * np.sum(sp_r)
    sig_r = sigmoid(v_r)
    # d/dx_i = penalty_weight * sig_r
    # d/dr_i = penalty_weight * sig_r
    # d/dt: W = 2t, dW/dt = 2. v_r = x + r - 2t. dv_r/dt = -2.
    # d(obj)/dt += penalty_weight * sum(sig_r * (-2))
    grad[1:1+3*N:3] += penalty_weight * sig_r
    grad[3:3+3*N:3] += penalty_weight * sig_r
    grad[0] += penalty_weight * np.sum(sig_r * (-2.0))
    
    # Bottom
    v_b = rs_eff - ys
    sp_b = softplus(v_b)
    obj += penalty_weight * np.sum(sp_b)
    sig_b = sigmoid(v_b)
    grad[2:2+3*N:3] -= penalty_weight * sig_b
    grad[3:3+3*N:3] += penalty_weight * sig_b
    
    # Top
    v_t = ys + rs_eff - H
    sp_t = softplus(v_t)
    obj += penalty_weight * np.sum(sp_t)
    sig_t = sigmoid(v_t)
    # H = 2(1-t) = 2 - 2t. dH/dt = -2. v_t = y + r - H. dv_t/dt = -(-2) = 2.
    grad[2:2+3*N:3] += penalty_weight * sig_t
    grad[3:3+3*N:3] += penalty_weight * sig_t
    grad[0] += penalty_weight * np.sum(sig_t * 2.0)
    
    # --- Overlap Penalties ---
    # For pairs (i, j) with i < j
    idx_i, idx_j = np.triu_indices(N, k=1)
    dx = xs[idx_i] - xs[idx_j]
    dy = ys[idx_i] - ys[idx_j]
    dist_sq = dx*dx + dy*dy
    min_dist = rs_eff[idx_i] + rs_eff[idx_j]
    
    # Only calculate penalty where there is potential overlap
    # Use a small epsilon in distance to avoid division by zero
    dist = np.sqrt(dist_sq + 1e-10)
    viol = min_dist - dist
    
    # Apply softplus to violation
    sp_o = softplus(viol)
    obj += penalty_weight * 2.0 * np.sum(sp_o) # Factor 2 for symmetry/weighting
    
    sig_o = sigmoid(viol)
    
    # Gradients for overlaps
    # d(sp_o)/d(viol) = sig_o
    # d(viol)/d(dist) = -1
    # d(dist)/d(dx) = dx/dist
    # d(viol)/dx_i = -1 * (dx/dist)
    # d(viol)/dx_j = -1 * (-dx/dist) = dx/dist
    # d(viol)/dr_i = 1
    # d(viol)/dr_j = 1
    
    # Accumulate gradients
    # We need to scatter the gradients back to the original indices
    # grad_x_i += penalty_weight * 2.0 * sig_o * (-dx/dist)
    # grad_x_j += penalty_weight * 2.0 * sig_o * (dx/dist)
    
    # Using np.add.at for accumulation
    # Indices in the flat array p:
    # x_i index: 1 + 3*idx_i
    # y_i index: 2 + 3*idx_i
    # r_i index: 3 + 3*idx_i
    
    gx = -penalty_weight * 2.0 * sig_o * (dx / dist)
    gy = -penalty_weight * 2.0 * sig_o * (dy / dist)
    gr = penalty_weight * 2.0 * sig_o
    
    np.add.at(grad, 1 + 3*idx_i, gx)
    np.add.at(grad, 2 + 3*idx_i, gy)
    np.add.at(grad, 3 + 3*idx_i, gr)
    
    np.add.at(grad, 1 + 3*idx_j, -gx)
    np.add.at(grad, 2 + 3*idx_j, -gy)
    np.add.at(grad, 3 + 3*idx_j, gr)
    
    return obj, grad

def _refine_state(p_init, deadline, penalty_weight=1000.0):
    """
    Refine a single initial state using L-BFGS-B.
    """
    # Define bounds
    # t in [0.1, 0.9]
    # x, y in [0, 2] (max dimension is 2 when t=0.5, but actually max W or H is 2)
    # r in [0.001, 1.0] (max radius is limited by box size)
    bounds = [(0.1, 0.9)]
    for i in range(N):
        bounds.append((0.0, 2.0)) # x
        bounds.append((0.0, 2.0)) # y
        bounds.append((0.001, 1.0)) # r
        
    def objective(p):
        return _evaluate_state(p, penalty_weight)[0]
        
    def gradient(p):
        return _evaluate_state(p, penalty_weight)[1]
        
    try:
        res = minimize(
            objective, 
            p_init, 
            jac=gradient, 
            method='L-BFGS-B', 
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        if res.success or res.fun < objective(p_init):
            return res.x
    except Exception:
        pass
    return p_init

def _generate_initial_state(rng, mode):
    """
    Generate an initial state vector.
    """
    p = np.zeros(1 + 3*N)
    
    # Random aspect ratio
    if mode == 'random':
        p[0] = rng.uniform(0.3, 0.7)
    else:
        p[0] = 0.5 # Start with square
        
    t = p[0]
    W = 2.0 * t
    H = 2.0 * (1.0 - t)
    
    if mode == 'grid':
        # 5x5 grid, fill 21
        side = 5
        idx = 0
        for i in range(side):
            for j in range(side):
                if idx >= N:
                    break
                x = (i + 0.5) / side * W
                y = (j + 0.5) / side * H
                r = 0.5 / side * min(W, H) * 0.9
                p[1 + 3*idx] = x
                p[2 + 3*idx] = y
                p[3 + 3*idx] = r
                idx += 1
    elif mode == 'hex':
        # Hexagonal packing
        # Approximate number of rows
        n_rows = 4
        n_cols = 6
        idx = 0
        dx = W / n_cols
        dy = H / n_rows
        for r_idx in range(n_rows):
            for c_idx in range(n_cols):
                if idx >= N:
                    break
                x = (c_idx + 0.5) * dx
                if r_idx % 2 == 1:
                    x += dx / 2
                y = (r_idx + 0.5) * dy
                r = 0.5 * min(dx, dy * 1.732) * 0.95 # Hex packing factor
                # Clamp x to stay in bounds
                x = np.clip(x, r, W - r)
                y = np.clip(y, r, H - r)
                p[1 + 3*idx] = x
                p[2 + 3*idx] = y
                p[3 + 3*idx] = r
                idx += 1
        # Fill remaining
        while idx < N:
            p[1 + 3*idx] = W/2
            p[2 + 3*idx] = H/2
            p[3 + 3*idx] = 0.01
            idx += 1
    else: # random
        for i in range(N):
            p[1 + 3*i] = rng.uniform(0.1, W - 0.1)
            p[2 + 3*i] = rng.uniform(0.1, H - 0.1)
            p[3 + 3*i] = rng.uniform(0.05, 0.2)
            
    return p

def construct():
    start_time = time.time()
    deadline = start_time + 140 # Safety margin
    
    best_score = -np.inf
    best_p = None
    
    rng = np.random.RandomState(42)
    
    # List of initial states to try
    initial_states = []
    
    # 1. Grid starts with varying aspect ratios
    for t_val in [0.4, 0.5, 0.6, 0.45, 0.55]:
        p_init = _generate_initial_state(rng, 'grid')
        p_init[0] = t_val
        initial_states.append(p_init)
        
    # 2. Hex starts
    for t_val in [0.45, 0.5, 0.55]:
        p_init = _generate_initial_state(rng, 'hex')
        p_init[0] = t_val
        initial_states.append(p_init)
        
    # 3. Random starts
    for _ in range(5):
        p_init = _generate_initial_state(rng, 'random')
        initial_states.append(p_init)
        
    # Refine each state
    for i, p_init in enumerate(initial_states):
        if time.time() > deadline - 10:
            break
            
        # Refine
        p_refined = _refine_state(p_init, deadline)
        
        # Evaluate
        obj, _ = _evaluate_state(p_refined)
        
        # Extract circles and check validity
        t = p_refined[0]
        W = 2.0 * t
        H = 2.0 * (1.0 - t)
        circles = p_refined[1:1+3*N].reshape(N, 3)
        
        # Check validity strictly
        if _check_valid(circles, W, H):
            sum_r = np.sum(circles[:, 2])
            # The objective is -sum_r + penalty. If valid, penalty is ~0.
            # So a lower objective (more negative) is better.
            if -obj > best_score:
                best_score = -obj
                best_p = p_refined.copy()
                
    # If we have a best solution, do a final high-precision polish
    if best_p is not None and time.time() < deadline - 5:
        # Run L-BFGS again with tighter tolerances
        p_final = _refine_state(best_p, deadline, penalty_weight=5000.0)
        
        t = p_final[0]
        W = 2.0 * t
        H = 2.0 * (1.0 - t)
        circles = p_final[1:1+3*N].reshape(N, 3)
        
        if _check_valid(circles, W, H):
            obj, _ = _evaluate_state(p_final)
            if -obj > best_score:
                best_p = p_final
                best_score = -obj
    
    if best_p is not None:
        circles = best_p[1:1+3*N].reshape(N, 3)
        # Ensure radii are non-negative
        circles[:, 2] = np.maximum(circles[:, 2], 0.0)
        return circles
    
    # Fallback: Simple grid in square
    side = 5
    w, h = 1.0, 1.0
    circles = np.zeros((N, 3))
    idx = 0
    for i in range(side):
        for j in range(side):
            if idx >= N:
                break
            circles[idx, 0] = (i + 0.5) / side * w
            circles[idx, 1] = (j + 0.5) / side * h
            circles[idx, 2] = 0.5 / side * 0.9
            idx += 1
    return circles

if __name__ == "__main__":
    res = construct()
    print(f"Result shape: {res.shape}")
    print(f"Sum of radii: {np.sum(res[:, 2])}")
