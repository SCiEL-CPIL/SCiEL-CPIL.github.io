"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, LinearConstraint
import itertools

def _pairwise_indices(n):
    """Return arrays of indices for all unique pairs (i,j) with i<j."""
    idx_i = []
    idx_j = []
    for i in range(n):
        for j in range(i + 1, n):
            idx_i.append(i)
            idx_j.append(j)
    return np.array(idx_i, dtype=int), np.array(idx_j, dtype=int)

def _compute_sum_radii(x):
    """Compute sum of radii from parameter vector x."""
    # x contains 3*N parameters: (x0,y0,r0, x1,y1,r1, ...)
    radii = x[2::3]
    return np.sum(radii)

def construct():
    n = N
    
    # Initial guess: grid placement
    side = int(np.ceil(np.sqrt(n)))
    initial_positions = []
    for i in range(side):
        for j in range(side):
            if len(initial_positions) >= n:
                break
            initial_positions.append(((i + 0.5) / side, (j + 0.5) / side))
    
    # Ensure we have exactly n positions
    while len(initial_positions) < n:
        initial_positions.append((0.5, 0.5))
    initial_positions = initial_positions[:n]
    
    # Initial radii: small equal radii
    initial_r = 0.5 / side - 0.05
    initial_x = []
    for (px, py) in initial_positions:
        initial_x.extend([px, py, initial_r])
    initial_x = np.array(initial_x)
    
    # Define constraints
    # 1. Circles must be inside the unit square:
    #    x_i >= r_i, x_i <= 1 - r_i, y_i >= r_i, y_i <= 1 - r_i
    #    => x_i - r_i >= 0, x_i + r_i <= 1, y_i - r_i >= 0, y_i + r_i <= 1
    #    => r_i - x_i <= 0, r_i - y_i <= 0, x_i + r_i <= 1, y_i + r_i <= 1
    #    In linear form: A_ub @ vars <= b_ub
    
    # For each circle i:
    #   -x_i + r_i <= 0  => x_i >= r_i
    #   -y_i + r_i <= 0  => y_i >= r_i
    #   x_i + r_i <= 1
    #   y_i + r_i <= 1
    
    # 2. Non-overlapping: (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2
    #    This is non-linear. We'll handle it via penalty or by using a different approach.
    #    Actually, let's use a penalty method for the non-linear constraints and keep
    #    the linear ones as proper constraints.
    
    # Alternative: Use a penalty-based approach with scipy.optimize.minimize
    # Minimize: -sum(r_i) + penalty * violation
    
    idx_i, idx_j = _pairwise_indices(n)
    num_pairs = len(idx_i)
    
    def objective_and_penalty(params):
        """Objective: -sum(radii) + penalty for constraint violations."""
        # Extract positions and radii
        xs = params[0::3]
        ys = params[1::3]
        rs = params[2::3]
        
        # Ensure radii are positive (clip to small positive)
        rs_clipped = np.maximum(rs, 1e-6)
        
        # Linear constraint violations (boundary)
        # x_i >= r_i => r_i - x_i <= 0
        # y_i >= r_i => r_i - y_i <= 0
        # x_i <= 1 - r_i => x_i + r_i <= 1
        # y_i <= 1 - r_i => y_i + r_i <= 1
        boundary_viol = 0.0
        for i in range(n):
            boundary_viol += max(0, rs_clipped[i] - xs[i])
            boundary_viol += max(0, rs_clipped[i] - ys[i])
            boundary_viol += max(0, xs[i] + rs_clipped[i] - 1)
            boundary_viol += max(0, ys[i] + rs_clipped[i] - 1)
        
        # Non-overlapping constraint violations
        overlap_viol = 0.0
        for k in range(num_pairs):
            i = idx_i[k]
            j = idx_j[k]
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = np.sqrt(dx*dx + dy*dy)
            min_dist = rs_clipped[i] + rs_clipped[j]
            if dist < min_dist:
                overlap_viol += (min_dist - dist)**2
        
        penalty = 100.0 * (boundary_viol + overlap_viol)
        return -np.sum(rs_clipped) + penalty
    
    # Bounds for parameters
    # x_i in [0, 1], y_i in [0, 1], r_i in [0, 0.5]
    bounds = []
    for i in range(n):
        bounds.append((0.0, 1.0))  # x_i
        bounds.append((0.0, 1.0))  # y_i
        bounds.append((0.0, 0.5))  # r_i
    
    # Use multiple restarts with different initial conditions
    best_result = None
    best_value = -np.inf
    
    np.random.seed(42)
    
    for restart in range(10):
        if restart == 0:
            x0 = initial_x.copy()
        else:
            # Random initial positions
            x0 = np.random.uniform(0.1, 0.9, size=2*n)
            x0[2::3] = np.random.uniform(0.02, 0.1, size=n)
        
        try:
            result = minimize(
                objective_and_penalty,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if result.success or result.fun < best_value:
                # Check if the solution is valid
                xs = result.x[0::3]
                ys = result.x[1::3]
                rs = result.x[2::3]
                rs = np.maximum(rs, 0)
                
                # Verify constraints
                valid = True
                for i in range(n):
                    if xs[i] < rs[i] - 1e-6 or xs[i] > 1 - rs[i] + 1e-6:
                        valid = False
                        break
                    if ys[i] < rs[i] - 1e-6 or ys[i] > 1 - rs[i] + 1e-6:
                        valid = False
                        break
                
                if valid:
                    for k in range(num_pairs):
                        i = idx_i[k]
                        j = idx_j[k]
                        dx = xs[i] - xs[j]
                        dy = ys[i] - ys[j]
                        dist = np.sqrt(dx*dx + dy*dy)
                        if dist < rs[i] + rs[j] - 1e-6:
                            valid = False
                            break
                
                if valid:
                    value = np.sum(rs)
                    if value > best_value:
                        best_value = value
                        best_result = np.column_stack([xs, ys, rs])
        except Exception:
            continue
    
    if best_result is not None:
        return best_result
    
    # Fallback: return the initial grid
    side = int(np.ceil(np.sqrt(n)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
    return np.array(cells[:n])

# EVOLVE-BLOCK-END
