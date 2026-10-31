import numpy as np
import time
from scipy.optimize import minimize
import itertools

N = 21

def _hexagonal_init(w, h, n):
    """
    Generate a hexagonal-like packing of n circles in a w x h rectangle.
    Returns x, y, r arrays.
    """
    # Estimate radius based on area
    # Area per circle ~ 1.15 * pi * r^2 for hex packing efficiency
    # Total area w*h ~ n * 1.15 * pi * r^2
    r_est = np.sqrt((w * h) / (n * 1.15 * np.pi))
    
    # We want to place n circles.
    # Let's try a grid of columns.
    # In hex packing, row spacing is r*sqrt(3), col spacing is 2r.
    # Let's just place them in a tight grid and let optimizer fix it.
    
    # Simple grid init is often enough if optimizer is good, but hex is better.
    # Let's create a list of centers.
    centers = []
    r = r_est * 0.9 # Start slightly smaller to allow expansion
    
    # Try to fit as many as possible in a hex lattice
    # Width of one column is 2r. Height of one row is r*sqrt(3).
    # Let's determine how many columns and rows we can fit.
    
    # Iterate over possible number of columns
    best_centers = None
    best_score = -1
    
    for cols in range(1, int(w / (2*r)) + 2):
        if cols == 0: continue
        # Calculate height needed for 'n' circles in 'cols' columns
        rows = np.ceil(n / cols)
        # Hex height: first row at r, next at r + r*sqrt(3), etc.
        # Actually, y-coords: r, r + r*sqrt(3), ...
        # Total height needed: r + (rows-1)*r*sqrt(3) + r = 2r + (rows-1)*r*sqrt(3)
        # If this height > h, this cols is invalid.
        h_needed = 2*r + (rows-1)*r*np.sqrt(3)
        if h_needed > h:
            continue
            
        # Generate centers
        c_list = []
        for i in range(n):
            col = i % cols
            row = i // cols
            # x position
            x = r + col * 2 * r
            # y position
            # Alternate rows are shifted by r
            if row % 2 == 0:
                y = r + row * r * np.sqrt(3)
            else:
                y = r + row * r * np.sqrt(3)
                # Shift x by r for odd rows? 
                # Standard hex: (0,0), (2r, 0), (r, r*sqrt(3)), (3r, r*sqrt(3))
                # So if row is odd, x += r
                x += r
            
            # Clip to bounds to ensure validity for init
            x = np.clip(x, r, w - r)
            y = np.clip(y, r, h - r)
            
            c_list.append((x, y))
        
        if len(c_list) == n:
            # Check for overlaps in this init (should be minimal if r_est is good)
            # Just use the first valid one
            xs = np.array([c[0] for c in c_list])
            ys = np.array([c[1] for c in c_list])
            rs = np.full(n, r)
            
            # Score: sum of radii (all same here)
            # We prefer tighter packing, so smaller h_needed is better? 
            # Or just take the first valid one.
            return xs, ys, rs
            
    # Fallback: simple grid
    side = int(np.ceil(np.sqrt(n)))
    xs = np.array([((i % side) + 0.5) * (w / side) for i in range(n)])
    ys = np.array([((i // side) + 0.5) * (h / side) for i in range(n)])
    rs = np.full(n, np.min(w, h) / (2 * side) * 0.9)
    return xs, ys, rs

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 110s safety margin out of 120s
    
    # Variables: [w, h, x_1...x_N, y_1...y_N, r_1...r_N]
    # Total vars: 2 + 3*N = 65
    
    def unflatten(v):
        w = v[0]
        h = v[1]
        x = v[2:2+N]
        y = v[2+N:2+2*N]
        r = v[2+2*N:2+3*N]
        return w, h, x, y, r

    def flatten(w, h, x, y, r):
        return np.concatenate([[w, h], x, y, r])

    # Penalty weights
    # We want to minimize -sum(r) + penalties
    # Penalties should be large enough to enforce constraints but not dominate the objective initially
    lambda_overlap = 100.0
    lambda_wall = 100.0
    lambda_perim = 100.0

    def objective_penalty(v):
        w, h, x, y, r = unflatten(v)
        
        # Objective: -sum(r)
        val = -np.sum(r)
        
        # Perimeter constraint: w + h <= 2
        # If w+h > 2, penalize
        perim_viol = max(0, w + h - 2.0)
        val += lambda_perim * perim_viol**2
        
        # Wall constraints
        # x_i - r_i >= 0  =>  r_i - x_i <= 0
        # w - x_i - r_i >= 0 => x_i + r_i - w <= 0
        # y_i - r_i >= 0
        # h - y_i - r_i >= 0
        
        # Vectorized wall violations
        # Left: r - x
        viol_left = np.maximum(0, r - x)
        # Right: x + r - w
        viol_right = np.maximum(0, x + r - w)
        # Bottom: r - y
        viol_bottom = np.maximum(0, r - y)
        # Top: y + r - h
        viol_top = np.maximum(0, y + r - h)
        
        wall_viol = np.sum(viol_left**2) + np.sum(viol_right**2) + np.sum(viol_bottom**2) + np.sum(viol_top**2)
        val += lambda_wall * wall_viol
        
        # Circle-circle overlaps
        # (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2
        # Violation: (r_i + r_j) - dist_ij > 0
        
        # Compute pairwise distances and sums of radii
        # To be efficient, we can compute all pairs, but N=21 is small.
        # dist2[i, j] = (x_i - x_j)^2 + (y_i - y_j)^2
        # r_sum[i, j] = r_i + r_j
        
        # Using broadcasting
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist2 = dx**2 + dy**2
        
        r_sum = r[:, None] + r[None, :]
        
        # We only care about i < j to avoid double counting and self-interaction
        # dist2 is symmetric. r_sum is symmetric.
        # Violation is max(0, r_sum - sqrt(dist2))
        # To avoid sqrt, we can use squared violation? No, that changes gradient.
        # But sqrt is differentiable everywhere except 0.
        # dist2 is 0 on diagonal.
        
        # Create mask for i < j
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        
        # Get relevant parts
        d2_masked = dist2[mask]
        rs_masked = r_sum[mask]
        
        # Calculate distances
        dists = np.sqrt(d2_masked)
        
        # Violations
        overlaps = np.maximum(0, rs_masked - dists)
        
        val += lambda_overlap * np.sum(overlaps**2)
        
        return val

    def compute_gradient(v):
        # Numerical gradient is slow but robust for this size if used sparingly.
        # However, L-BFGS-B can use analytical or numerical.
        # Let's let L-BFGS-B use numerical gradient initially, or implement analytical.
        # Given time constraints, numerical might be too slow for 65 vars * many iterations.
        # Let's try to provide an analytical gradient for the penalty terms.
        
        w, h, x, y, r = unflatten(v)
        
        grad = np.zeros_like(v)
        
        # Gradient of -sum(r)
        grad[2+2*N:2+3*N] -= 1.0
        
        # Perimeter
        if w + h > 2.0:
            grad[0] += 2 * lambda_perim * (w + h - 2.0)
            grad[1] += 2 * lambda_perim * (w + h - 2.0)
            
        # Walls
        # Left: max(0, r-x)^2
        # d/dr = 2(r-x) if r>x else 0
        # d/dx = -2(r-x) if r>x else 0
        viol_left = np.maximum(0, r - x)
        grad[2+2*N:2+3*N] += 2 * lambda_wall * viol_left
        grad[2:2+N] -= 2 * lambda_wall * viol_left
        
        # Right: max(0, x+r-w)^2
        # d/dx = 2(x+r-w)
        # d/dr = 2(x+r-w)
        viol_right = np.maximum(0, x + r - w)
        grad[2:2+N] += 2 * lambda_wall * viol_right
        grad[2+2*N:2+3*N] += 2 * lambda_wall * viol_right
        grad[0] -= 2 * lambda_wall * np.sum(viol_right)
        
        # Bottom: max(0, r-y)^2
        viol_bottom = np.maximum(0, r - y)
        grad[2+2*N:2+3*N] += 2 * lambda_wall * viol_bottom
        grad[2+N:2+2*N] -= 2 * lambda_wall * viol_bottom
        
        # Top: max(0, y+r-h)^2
        viol_top = np.maximum(0, y + r - h)
        grad[2+N:2+2*N] += 2 * lambda_wall * viol_top
        grad[2+2*N:2+3*N] += 2 * lambda_wall * viol_top
        grad[1] -= 2 * lambda_wall * np.sum(viol_top)
        
        # Overlaps
        # L = sum max(0, r_i+r_j - dist)^2
        # d/dx_i: sum_{j!=i} 2 * overlap_ij * d(dist)/d(x_i) * (-1) ? No.
        # f = (r_i+r_j - dist)^2
        # df/dx_i = 2(r_i+r_j - dist) * (-1) * d(dist)/dx_i
        # d(dist)/dx_i = (x_i - x_j) / dist
        
        # We need to accumulate gradients for each i from all j.
        
        # Re-calculate masks and values
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist2 = dx**2 + dy**2
        dists = np.sqrt(dist2)
        
        # Avoid division by zero
        dists_safe = np.where(dists > 1e-8, dists, 1e-8)
        
        r_sum = r[:, None] + r[None, :]
        overlaps = np.maximum(0, r_sum - dists)
        
        # Factor: 2 * overlap
        # d(dist)/dx_i = (x_i - x_j) / dist
        # Contribution to grad[x_i] from pair (i,j) is:
        # -2 * overlap_ij * (x_i - x_j) / dist_ij
        
        # Sum over j for each i
        # Note: overlap is 0 if not overlapping, so we can just compute the term.
        
        # Term_ij = 2 * overlap_ij * (x_i - x_j) / dist_ij
        # We want sum_j Term_ij for grad[x_i]
        
        # Using broadcasting
        # x_diff = x[:, None] - x[None, :]
        # term_x = 2 * overlaps * (x_diff / dists_safe)
        # grad_x = sum(term_x, axis=1)
        
        # But we must be careful: the gradient of the penalty for pair (i,j) contributes to both i and j.
        # The penalty is sum_{i<j} P_{ij}.
        # P_{ij} = (r_i+r_j - d_{ij})^2
        # dP_{ij}/dx_i = 2(r_i+r_j - d_{ij}) * (x_i - x_j)/d_{ij}
        # dP_{ij}/dx_j = 2(r_i+r_j - d_{ij}) * (x_j - x_i)/d_{ij} = - dP_{ij}/dx_i
        
        # So for each pair (i,j), we add G to grad[x_i] and -G to grad[x_j].
        # In the matrix formulation:
        # grad[x_i] = sum_{j != i} 2 * overlap_{ij} * (x_i - x_j) / d_{ij}
        
        # Let's compute this.
        # overlaps is N x N, symmetric, 0 on diagonal.
        # x_diff is N x N.
        # factor = 2 * overlaps / dists_safe
        # grad_x_contrib = sum(factor * x_diff, axis=1)
        
        factor = 2 * overlaps / dists_safe
        # Set diagonal to 0 to be safe (overlap is 0 anyway)
        np.fill_diagonal(factor, 0)
        
        grad_x = np.sum(factor * dx, axis=1)
        grad_y = np.sum(factor * dy, axis=1)
        
        grad[2:2+N] += lambda_overlap * grad_x
        grad[2+N:2+2*N] += lambda_overlap * grad_y
        
        # Gradient w.r.t radii
        # dP_{ij}/dr_i = 2(r_i+r_j - d_{ij}) * 1
        # dP_{ij}/dr_j = 2(r_i+r_j - d_{ij}) * 1
        # grad[r_i] = sum_{j != i} 2 * overlap_{ij}
        
        grad_r = np.sum(2 * overlaps, axis=1)
        grad[2+2*N:2+3*N] += lambda_overlap * grad_r
        
        return grad

    # Bounds
    bounds = [(0.2, 1.8), (0.2, 1.8)]
    for i in range(N):
        bounds.append((0.0, 2.0)) # x_i
    for i in range(N):
        bounds.append((0.0, 2.0)) # y_i
    for i in range(N):
        bounds.append((0.0, 1.0)) # r_i
        
    best_value = -np.inf
    best_v = None
    
    np.random.seed(42)
    
    # List of initializations
    inits = []
    
    # Aspect ratios
    aspects = [1.0, 1.2, 0.8, 1.5, 0.67, 1.3, 0.77]
    
    for ar in aspects:
        w_init = np.sqrt(ar)
        h_init = 1.0 / np.sqrt(ar)
        
        # Hex init
        xs, ys, rs = _hexagonal_init(w_init, h_init, N)
        v0 = np.concatenate([[w_init, h_init], xs, ys, rs])
        inits.append(v0)
        
        # Random init
        xs_r = np.random.uniform(0.1, w_init - 0.1, N)
        ys_r = np.random.uniform(0.1, h_init - 0.1, N)
        rs_r = np.full(N, 0.1)
        v0_r = np.concatenate([[w_init, h_init], xs_r, ys_r, rs_r])
        inits.append(v0_r)
        
    # Shuffle
    np.random.shuffle(inits)
    
    for i, v0 in enumerate(inits):
        if time.time() > deadline - 30: # Leave 30s for polish
            break
            
        v0_clipped = v0.copy()
        v0_clipped[0] = np.clip(v0_clipped[0], 0.2, 1.8)
        v0_clipped[1] = np.clip(v0_clipped[1], 0.2, 1.8)
        for k in range(N):
            v0_clipped[2+k] = np.clip(v0_clipped[2+k], 0.01, 1.99)
            v0_clipped[2+N+k] = np.clip(v0_clipped[2+N+k], 0.01, 1.99)
            v0_clipped[2+2*N+k] = np.clip(v0_clipped[2+2*N+k], 0.01, 0.99)
            
        try:
            # Run L-BFGS-B
            result = minimize(
                objective_penalty, v0_clipped, jac=compute_gradient, method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False}
            )
            
            # Check if the result is feasible
            w, h, x, y, r = unflatten(result.x)
            feasible = True
            if w + h > 2.0 + 1e-4:
                feasible = False
            for k in range(N):
                if x[k] - r[k] < -1e-4 or w - x[k] - r[k] < -1e-4 or \
                   y[k] - r[k] < -1e-4 or h - y[k] - r[k] < -1e-4 or r[k] < -1e-4:
                    feasible = False
                    break
                for j in range(k + 1, N):
                    dist2 = (x[k] - x[j])**2 + (y[k] - y[j])**2
                    if dist2 < (r[k] + r[j])**2 - 1e-4:
                        feasible = False
                        break
                if not feasible:
                    break
            
            if feasible:
                value = np.sum(r)
                if value > best_value:
                    best_value = value
                    best_v = result.x.copy()
                    
        except Exception:
            continue

    # Polish phase: Use SLSQP on the best feasible solution found
    if best_v is not None and time.time() < deadline - 5:
        # Define SLSQP objective and constraints
        def slsqp_obj(v):
            w, h, x, y, r = unflatten(v)
            return -np.sum(r)

        def slsqp_cons(v):
            w, h, x, y, r = unflatten(v)
            cons = []
            # Perimeter
            cons.append({'type': 'ineq', 'fun': lambda v: 2.0 - v[0] - v[1]})
            
            # Walls
            for i in range(N):
                cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[2+i] - v[2+2*N+i]})
                cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[0] - v[2+i] - v[2+2*N+i]})
                cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[2+N+i] - v[2+2*N+i]})
                cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[1] - v[2+N+i] - v[2+2*N+i]})
                
            # Circles
            for i in range(N):
                for j in range(i + 1, N):
                    cons.append({
                        'type': 'ineq',
                        'fun': lambda v, i=i, j=j: (v[2+i] - v[2+j])**2 + (v[2+N+i] - v[2+N+j])**2 - (v[2+2*N+i] + v[2+2*N+j])**2
                    })
            return cons

        try:
            # Use a tighter bound for SLSQP to stay close to feasible region
            # We can run SLSQP for a short time
            result_polish = minimize(
                slsqp_obj, best_v, method='SLSQP',
                bounds=bounds,
                constraints=slsqp_cons(best_v),
                options={'maxiter': 200, 'ftol': 1e-10, 'disp': False}
            )
            
            if result_polish.success:
                w, h, x, y, r = unflatten(result_polish.x)
                # Verify again
                feasible = True
                if w + h > 2.0 + 1e-6:
                    feasible = False
                for k in range(N):
                    if x[k] - r[k] < -1e-5 or w - x[k] - r[k] < -1e-5 or \
                       y[k] - r[k] < -1e-5 or h - y[k] - r[k] < -1e-5 or r[k] < -1e-5:
                        feasible = False
                        break
                    for j in range(k + 1, N):
                        dist2 = (x[k] - x[j])**2 + (y[k] - y[j])**2
                        if dist2 < (r[k] + r[j])**2 - 1e-5:
                            feasible = False
                            break
                    if not feasible:
                        break
                            
                if feasible:
                    value = np.sum(r)
                    if value > best_value:
                        best_value = value
                        best_v = result_polish.x.copy()
                        
        except Exception:
            pass

    if best_v is None:
        # Fallback
        side = int(np.ceil(np.sqrt(N)))
        w = 1.0
        h = 1.0
        xs = np.array([((i % side) + 0.5) / side for i in range(N)])
        ys = np.array([((i // side) + 0.5) / side for i in range(N)])
        rs = np.full(N, 0.5 / side - 0.01)
        return np.column_stack([xs, ys, rs])
    else:
        w, h, x, y, r = unflatten(best_v)
        return np.column_stack([x, y, r])
