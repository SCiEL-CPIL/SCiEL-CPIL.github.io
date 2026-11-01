import numpy as np
from scipy.optimize import minimize
import time

N = 32

def _hexagonal_init(n):
    """Generate an initial hexagonal packing configuration."""
    # Estimate side length for hex packing
    # Area per circle ~ pi * r^2. Hex packing density ~ 0.9069.
    # We want to fit n circles in [0,1]^2.
    # Approx radius r ~ 1 / sqrt(n / 0.9)
    r_est = 1.0 / np.sqrt(n / 0.9)
    
    # Generate grid points
    pts = []
    # Rows
    rows = int(np.sqrt(n * 2 / np.sqrt(3))) + 2
    for j in range(rows):
        for i in range(rows):
            x = i * r_est * np.sqrt(3)
            y = j * r_est * 1.5
            if j % 2 == 1:
                x += r_est * np.sqrt(3) / 2
            pts.append([x, y])
            
    pts = np.array(pts)
    # Center and scale to fit in [0,1]
    # Find bounding box
    min_x, min_y = pts.min(axis=0)
    max_x, max_y = pts.max(axis=0)
    width = max_x - min_x
    height = max_y - min_y
    
    # Scale to fit within [r_est, 1-r_est] roughly
    scale = min(1.0 / width, 1.0 / height)
    # Actually, let's just place them in a grid and let the optimizer fix it.
    # A simple grid is often sufficient for L-BFGS-B to find a good local opt.
    
    side = int(np.ceil(np.sqrt(n)))
    cells = []
    for i in range(side):
        for j in range(side):
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            # Initial radius small to avoid immediate huge penalties
            r = 0.5 / side - 0.05 
            cells.append([x, y, r])
            
    return np.array(cells[:n])

def _objective_and_grad(v, mu):
    """
    v: [x1, y1, r1, ..., xN, yN, rN]
    mu: penalty weight
    Returns objective value and gradient.
    Objective: -sum(r_i) + mu * sum(penalties)
    """
    n = N
    x = v[0:n]
    y = v[n:2*n]
    r = v[2*n:3*n]
    
    # Objective part
    obj = -np.sum(r)
    grad_obj = np.zeros(3*n)
    grad_obj[2*n:3*n] = -1.0
    
    # Penalty part
    # Pairwise constraints: (xi-xj)^2 + (yi-yj)^2 >= (ri+rj)^2
    # Penalty: max(0, ri+rj - dist)^2
    
    grad_pen = np.zeros(3*n)
    pen_val = 0.0
    
    # Vectorized pairwise calculation
    # dist matrix
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d = np.sqrt(dx**2 + dy**2)
    
    # Avoid division by zero
    d[d == 0] = 1e-8
    
    # Overlap amount
    overlap = r[:, None] + r[None, :] - d
    
    # Only consider pairs i < j to avoid double counting and self
    # Create mask for i < j
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    
    # Get values for valid pairs
    o = overlap[mask]
    d_valid = d[mask]
    dx_valid = dx[mask]
    dy_valid = dy[mask]
    
    # active overlaps
    active = o > 0
    if np.any(active):
        o_act = o[active]
        d_act = d_valid[active]
        dx_act = dx_valid[active]
        dy_act = dy_valid[active]
        
        # Indices
        i_idx, j_idx = np.where(mask)
        i_act = i_idx[active]
        j_act = j_idx[active]
        
        # Penalty value
        pen_val += np.sum(o_act**2)
        
        # Gradient with respect to x_i, y_i, r_i for pair (i, j)
        # P = o^2, o = ri + rj - d
        # dP/do = 2*o
        # do/dxi = -dx/d = -dx_i/d (where dx_i = xi - xj)
        # do/dri = 1
        
        # Accumulate gradients
        # For each active pair (i, j):
        # grad_x[i] += 2*o * (-dx_act/d_act)
        # grad_y[i] += 2*o * (-dy_act/d_act)
        # grad_r[i] += 2*o
        # grad_x[j] += 2*o * ( dx_act/d_act)
        # grad_y[j] += 2*o * ( dy_act/d_act)
        # grad_r[j] += 2*o
        
        factor = 2 * o_act
        
        # Use np.add.at for accumulation
        np.add.at(grad_pen, i_act, -factor * dx_act / d_act) # x component for i
        np.add.at(grad_pen, n + i_act, -factor * dy_act / d_act) # y component for i
        np.add.at(grad_pen, 2*n + i_act, factor) # r component for i
        
        np.add.at(grad_pen, j_act, factor * dx_act / d_act) # x component for j
        np.add.at(grad_pen, n + j_act, factor * dy_act / d_act) # y component for j
        np.add.at(grad_pen, 2*n + j_act, factor) # r component for j
        
    # Wall constraints
    # xi >= ri  => max(0, ri - xi)^2
    # xi <= 1-ri => max(0, ri - (1-xi))^2 = max(0, ri + xi - 1)^2
    # Similarly for y
    
    # Left wall: max(0, ri - xi)^2
    o_left = r - x
    active_l = o_left > 0
    if np.any(active_l):
        o_l = o_left[active_l]
        idx = np.where(active_l)[0]
        pen_val += np.sum(o_l**2)
        f = 2 * o_l
        # grad_r[i] += f, grad_x[i] -= f
        np.add.at(grad_pen, 2*n + idx, f)
        np.add.at(grad_pen, idx, -f)
        
    # Right wall: max(0, ri + xi - 1)^2
    o_right = r + x - 1
    active_r = o_right > 0
    if np.any(active_r):
        o_r = o_right[active_r]
        idx = np.where(active_r)[0]
        pen_val += np.sum(o_r**2)
        f = 2 * o_r
        # grad_r[i] += f, grad_x[i] += f
        np.add.at(grad_pen, 2*n + idx, f)
        np.add.at(grad_pen, idx, f)
        
    # Bottom wall: max(0, ri - yi)^2
    o_bot = r - y
    active_b = o_bot > 0
    if np.any(active_b):
        o_b = o_bot[active_b]
        idx = np.where(active_b)[0]
        pen_val += np.sum(o_b**2)
        f = 2 * o_b
        np.add.at(grad_pen, 2*n + idx, f)
        np.add.at(grad_pen, n + idx, -f)
        
    # Top wall: max(0, ri + yi - 1)^2
    o_top = r + y - 1
    active_t = o_top > 0
    if np.any(active_t):
        o_t = o_top[active_t]
        idx = np.where(active_t)[0]
        pen_val += np.sum(o_t**2)
        f = 2 * o_t
        np.add.at(grad_pen, 2*n + idx, f)
        np.add.at(grad_pen, n + idx, f)
        
    total_obj = obj + mu * pen_val
    total_grad = grad_obj + mu * grad_pen
    
    return total_obj, total_grad

def _is_valid(v):
    """Check if the configuration is valid (no overlaps, inside square)."""
    n = N
    x = v[0:n]
    y = v[n:2*n]
    r = v[2*n:3*n]
    
    # Check bounds
    if np.any(x < r - 1e-6) or np.any(x > 1 - r + 1e-6):
        return False
    if np.any(y < r - 1e-6) or np.any(y > 1 - r + 1e-6):
        return False
    if np.any(r < 0):
        return False
        
    # Check overlaps
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d = np.sqrt(dx**2 + dy**2)
    
    # Required distance
    req = r[:, None] + r[None, :]
    
    # Check if any d < req - tol
    # Use a small tolerance
    if np.any(d < req - 1e-5):
        return False
        
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    best_score = 0.0
    best_v = None
    
    # Initial configuration
    v0 = _hexagonal_init(N).flatten()
    
    # Multi-start
    num_restarts = 50 # Adjust based on time
    
    # Penalty weight
    mu = 100.0
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Add jitter for restarts > 0
        if restart == 0:
            v_init = v0.copy()
        else:
            v_init = v0 + np.random.normal(0, 0.02, v0.shape)
            # Ensure initial radii are small enough to be valid or close to valid
            # Just clip radii to be small
            v_init[2*N:3*N] = np.clip(v_init[2*N:3*N], 0.01, 0.1)
            v_init[0:N] = np.clip(v_init[0:N], 0.0, 1.0)
            v_init[N:2*N] = np.clip(v_init[N:2*N], 0.0, 1.0)
            
        # Optimization
        try:
            res = minimize(
                _objective_and_grad, 
                v_init, 
                args=(mu,), 
                method='L-BFGS-B', 
                jac=True,
                options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6}
            )
            v_opt = res.x
            
            # Check validity
            if _is_valid(v_opt):
                score = np.sum(v_opt[2*N:3*N])
                if score > best_score:
                    best_score = score
                    best_v = v_opt.copy()
            else:
                # If invalid, try to shrink radii to make it valid
                # Simple heuristic: scale radii down
                x = v_opt[0:N]
                y = v_opt[N:2*N]
                r = v_opt[2*N:3*N]
                
                # Find max overlap violation
                # This is expensive, so maybe just scale down by 0.95 iteratively?
                # Or just skip this restart if it fails.
                # Given L-BFGS-B with high mu, it should be close to valid.
                pass
                
        except Exception as e:
            pass
            
        # If we have a good score, we might reduce restarts or increase precision
        # But for now, keep running until deadline.
        
    if best_v is None:
        # Fallback to seed
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        best_v = np.array(cells[:N]).flatten()
        
    # Reshape to (N, 3)
    result = best_v.reshape(N, 3)
    
    # Final safety check and fix
    # Ensure radii are positive
    result[:, 2] = np.maximum(result[:, 2], 1e-6)
    
    # If still invalid, shrink radii
    for _ in range(10):
        if _is_valid(result.flatten()):
            break
        result[:, 2] *= 0.95
        
    return result

# EVOLVE-BLOCK-END
