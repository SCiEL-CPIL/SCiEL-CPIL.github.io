import numpy as np
import time
from scipy.optimize import minimize

N = 26

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin

    # Precompute indices for pairs
    i_idx, j_idx = np.triu_indices(N, k=1)
    
    # Objective function for scipy: minimize -sum(radii)
    # We use a penalty method to handle constraints.
    def objective(params):
        # params is (N*3,)
        x = params[0:N]
        y = params[N:2*N]
        r = params[2*N:3*N]
        
        # Penalty for boundary violations
        # x - r >= 0, x + r <= 1  => r <= min(x, 1-x)
        # y - r >= 0, y + r <= 1  => r <= min(y, 1-y)
        max_r_bound = np.minimum(np.minimum(x, 1-x), np.minimum(y, 1-y))
        bound_viol = np.sum(np.maximum(0, r - max_r_bound))
        
        # Penalty for overlap violations
        # dist(i, j) >= r_i + r_j
        dx = x[i_idx] - x[j_idx]
        dy = y[i_idx] - y[j_idx]
        dist = np.sqrt(dx*dx + dy*dy)
        sum_r = r[i_idx] + r[j_idx]
        pair_viol = np.sum(np.maximum(0, sum_r - dist))
        
        total_viol = bound_viol + pair_viol
        
        # Strong penalty to encourage feasibility
        penalty_weight = 50.0
        return -np.sum(r) + penalty_weight * total_viol

    # Gradient (optional, but L-BFGS-B works well with numerical or analytical)
    # For simplicity and robustness, we'll let L-BFGS-B estimate gradients if not provided,
    # or we can provide them. Given the complexity, numerical is safer for a first try,
    # but analytical is faster. Let's try without first, or use finite differences.
    # Actually, providing gradients significantly speeds up convergence.
    
    def gradient(params):
        x = params[0:N].copy()
        y = params[N:2*N].copy()
        r = params[2*N:3*N].copy()
        
        grad = np.zeros_like(params)
        
        # Objective part: -sum(r)
        grad[2*N:3*N] = -1.0
        
        # Penalty part: 50 * total_viol
        # bound_viol = sum(max(0, r_i - min(x_i, 1-x_i, y_i, 1-y_i)))
        # This is non-smooth at boundaries, but subgradient is fine.
        
        # Calculate bounds
        max_r_bound = np.minimum(np.minimum(x, 1-x), np.minimum(y, 1-y))
        
        # Gradient of bound_viol w.r.t x, y, r
        # If r_i > max_r_bound_i:
        #   d(r_i - max_r_bound_i)/dr_i = 1
        #   d(r_i - max_r_bound_i)/dx_i = -d(max_r_bound_i)/dx_i
        #   d(r_i - max_r_bound_i)/dy_i = -d(max_r_bound_i)/dy_i
        
        active_bound = r > max_r_bound
        
        # d(max_r_bound)/dx:
        # max_r_bound = min(x, 1-x, y, 1-y)
        # If x is the min, d/dx = 1. If 1-x is min, d/dx = -1. If y or 1-y is min, d/dx = 0.
        
        # To simplify, let's just use numerical gradients for robustness.
        # L-BFGS-B with finite differences is acceptable for N=26 (78 vars) in 100s.
        # However, analytical is better. Let's implement a simple subgradient.
        
        # For each circle i:
        for i in range(N):
            # Determine which constraint is active for max_r_bound
            # candidates: x, 1-x, y, 1-y
            vals = [x[i], 1-x[i], y[i], 1-y[i]]
            min_val = min(vals)
            
            # If r_i > min_val, we have a violation.
            if r[i] > min_val:
                # Add gradient contributions
                # d/d r_i: +1 (from r_i - min_val)
                grad[2*N + i] += 50.0 * 1.0
                
                # d/d x_i: - d(min_val)/dx_i
                # If min_val is x[i], d/dx_i = 1 -> grad -= 50
                # If min_val is 1-x[i], d/dx_i = -1 -> grad += 50
                # Else 0
                if np.isclose(min_val, x[i]):
                    grad[i] += 50.0 * (-1.0)
                elif np.isclose(min_val, 1-x[i]):
                    grad[i] += 50.0 * (1.0)
                
                # d/d y_i: - d(min_val)/dy_i
                if np.isclose(min_val, y[i]):
                    grad[N + i] += 50.0 * (-1.0)
                elif np.isclose(min_val, 1-y[i]):
                    grad[N + i] += 50.0 * (1.0)

        # Pairwise violations: sum(max(0, r_i + r_j - dist_ij))
        # dist_ij = sqrt((x_i-x_j)^2 + (y_i-y_j)^2)
        # If r_i + r_j > dist_ij:
        #   d/d r_i = 1
        #   d/d r_j = 1
        #   d/d x_i = -(x_i-x_j)/dist_ij
        #   d/d x_j = (x_i-x_j)/dist_ij
        #   d/d y_i = -(y_i-y_j)/dist_ij
        #   d/d y_j = (y_i-y_j)/dist_ij
        
        dx = x[i_idx] - x[j_idx]
        dy = y[i_idx] - y[j_idx]
        dist = np.sqrt(dx*dx + dy*dy)
        # Avoid division by zero
        dist_safe = np.where(dist < 1e-8, 1e-8, dist)
        
        sum_r = r[i_idx] + r[j_idx]
        active_pair = sum_r > dist
        
        if np.any(active_pair):
            idx_active = np.where(active_pair)[0]
            
            for k in idx_active:
                ii = i_idx[k]
                jj = j_idx[k]
                
                # Radius gradients
                grad[2*N + ii] += 50.0 * 1.0
                grad[2*N + jj] += 50.0 * 1.0
                
                # Position gradients
                dx_val = dx[k]
                dy_val = dy[k]
                dist_val = dist_safe[k]
                
                # d(dist)/dx_i = -dx/dist
                # d(r_i + r_j - dist)/dx_i = -(-dx/dist) = dx/dist ? 
                # Let's check: f = r_i + r_j - dist.
                # df/dx_i = - d(dist)/dx_i = - ( (x_i-x_j)/dist ) = - dx_val / dist_val
                # Wait, dist = |p_i - p_j|. d/dx_i (sqrt((x_i-x_j)^2...)) = (x_i-x_j)/dist.
                # So df/dx_i = - (x_i-x_j)/dist.
                # My previous derivation: d/dx_i = -(x_i-x_j)/dist_ij.
                # So grad[x_i] += 50 * (-(x_i-x_j)/dist)
                
                grad[ii] += 50.0 * (-dx_val / dist_val)
                grad[jj] += 50.0 * (dx_val / dist_val)
                
                grad[N + ii] += 50.0 * (-dy_val / dist_val)
                grad[N + jj] += 50.0 * (dy_val / dist_val)

        return grad

    # Multi-start strategy
    best_params = None
    best_score = -np.inf

    # Generate diverse initializations
    # 1. Grid-based
    # 2. Random
    # 3. Radial
    
    initializations = []
    
    # Init 1: Grid
    grid_x, grid_y = np.meshgrid(np.linspace(0.1, 0.9, 5), np.linspace(0.1, 0.9, 5))
    grid_pts = np.column_stack([grid_x.ravel(), grid_y.ravel()])[:N]
    grid_r = np.full(N, 0.05)
    initializations.append(np.hstack([grid_pts.T, grid_r[:, None]]).ravel())
    
    # Init 2: Random
    np.random.seed(42)
    rand_pts = np.random.uniform(0.1, 0.9, (N, 2))
    rand_r = np.full(N, 0.05)
    initializations.append(np.hstack([rand_pts.T, rand_r[:, None]]).ravel())
    
    # Init 3: Radial (similar to current but flattened)
    pts = np.zeros((N, 2))
    pts[0] = [0.5, 0.5]
    r1 = 0.2
    for i in range(8):
        angle = i * np.pi / 4
        pts[i+1] = [0.5 + r1*np.cos(angle), 0.5 + r1*np.sin(angle)]
    r2 = 0.42
    angles2 = np.linspace(0, 2*np.pi, 17, endpoint=False)
    for i in range(17):
        pts[i+9] = [0.5 + r2*np.cos(angles2[i]), 0.5 + r2*np.sin(angles2[i])]
    rad_r = np.full(N, 0.05)
    initializations.append(np.hstack([pts.T, rad_r[:, None]]).ravel())
    
    # Add a few random restarts if time permits
    for restart in range(5):
        r_pts = np.random.uniform(0.1, 0.9, (N, 2))
        r_r = np.full(N, 0.05)
        initializations.append(np.hstack([r_pts.T, r_r[:, None]]).ravel())

    for init in initializations:
        if time.time() > deadline:
            break
            
        try:
            res = minimize(objective, init, method='L-BFGS-B', jac=gradient,
                           options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8})
            
            if res.success or res.fun < objective(init):
                score = -res.fun
                # Check validity
                x = res.x[0:N]
                y = res.x[N:2*N]
                r = res.x[2*N:3*N]
                
                max_r_bound = np.minimum(np.minimum(x, 1-x), np.minimum(y, 1-y))
                bound_viol = np.sum(np.maximum(0, r - max_r_bound))
                
                dx = x[i_idx] - x[j_idx]
                dy = y[i_idx] - y[j_idx]
                dist = np.sqrt(dx*dx + dy*dy)
                sum_r = r[i_idx] + r[j_idx]
                pair_viol = np.sum(np.maximum(0, sum_r - dist))
                
                total_viol = bound_viol + pair_viol
                
                if total_viol < 1e-5:
                    # Valid solution
                    if np.sum(r) > best_score:
                        best_score = np.sum(r)
                        best_params = res.x.copy()
                else:
                    # Invalid, but maybe we can project?
                    # For now, only accept valid ones for safety.
                    # Or accept if penalty is low? No, must be valid.
                    pass
        except Exception as e:
            pass

    if best_params is None:
        # Fallback: return a simple valid configuration
        # Grid of small circles
        x, y = np.meshgrid(np.linspace(0.1, 0.9, 5), np.linspace(0.1, 0.9, 5))
        pts = np.column_stack([x.ravel(), y.ravel()])[:N]
        r = np.full(N, 0.08)
        return np.hstack([pts.T, r[:, None]])
    
    # Final projection/safety check
    x = best_params[0:N]
    y = best_params[N:2*N]
    r = best_params[2*N:3*N]
    
    # Ensure strict validity
    scale = 1.0
    for _ in range(20):
        r_check = r * scale
        max_r_bound = np.minimum(np.minimum(x, 1-x), np.minimum(y, 1-y))
        if np.any(r_check > max_r_bound + 1e-9):
            scale *= 0.95
            continue
        
        dx = x[i_idx] - x[j_idx]
        dy = y[i_idx] - y[j_idx]
        dist = np.sqrt(dx*dx + dy*dy)
        sum_r = r_check[i_idx] + r_check[j_idx]
        if np.any(sum_r > dist + 1e-9):
            scale *= 0.95
            continue
        
        break
    
    r_final = r * scale
    result = np.hstack([x[:, None], y[:, None], r_final[:, None]])
    
    return result

# EVOLVE-BLOCK-END
