"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def construct():
    """
    Pack 21 circles in a rectangle of perimeter at most 4.
    Uses optimization to maximize sum of radii subject to non-overlap constraints.
    """
    # Perimeter = 2(w+h) = 4 => w+h = 2
    # We will try a few aspect ratios to find the best one.
    # Square (1x1) is often good, but rectangles can sometimes pack better.
    best_val = -1
    best_result = None
    
    # Define aspect ratios to try
    # w + h = 2. w can range from 0.5 to 1.5 (assuming w<=h or just symmetric)
    # Let's try w in [0.8, 1.2]
    aspect_ratios = np.linspace(0.7, 1.3, 11)
    
    for w in aspect_ratios:
        h = 2.0 - w
        
        # Initial guess: Grid layout
        cols = int(np.ceil(np.sqrt(N)))
        rows = int(np.ceil(N / cols))
        
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.zeros(N)
        
        for i in range(N):
            col = i % cols
            row = i // cols
            x[i] = (col + 0.5) * w / cols
            y[i] = (row + 0.5) * h / rows
            # Initial radius: half the cell size, slightly smaller
            r[i] = min(w / cols, h / rows) * 0.45
        
        # Flatten variables: [x1, y1, r1, ..., x21, y21, r21]
        v0 = np.concatenate([x, y, r])
        
        # Define objective function
        # We want to maximize sum(r). So minimize -sum(r).
        # Add penalty for overlaps to keep them valid.
        # Penalty: Sum of squared overlaps for circle-circle and circle-wall.
        
        def objective(v):
            x_v = v[0:N]
            y_v = v[N:2*N]
            r_v = v[2*N:3*N]
            
            # 1. Maximize radii (minimize negative sum)
            val = -np.sum(r_v)
            
            # 2. Circle-Circle Overlaps
            # dist_ij = sqrt((xi-xj)^2 + (yi-yj)^2)
            # overlap = max(0, ri + rj - dist_ij)
            # penalty = sum(overlap^2)
            
            # Vectorized calculation for efficiency
            # Create NxN distance matrix
            dx = x_v[:, np.newaxis] - x_v[np.newaxis, :]
            dy = y_v[:, np.newaxis] - y_v[np.newaxis, :]
            dist = np.sqrt(dx**2 + dy**2 + 1e-12) # avoid div by 0
            
            # Sum of radii matrix
            r_sum = r_v[:, np.newaxis] + r_v[np.newaxis, :]
            
            # Overlap matrix
            # We only care about i < j, but symmetric matrix is fine if we divide by 2
            overlap_cc = np.maximum(0, r_sum - dist)
            # Zero out diagonal (self-overlap is not a constraint, dist=0, r_sum=2ri, overlap=2ri? No, we want 0)
            np.fill_diagonal(overlap_cc, 0)
            
            penalty_cc = np.sum(overlap_cc**2) / 2.0 # /2 because symmetric
            
            # 3. Wall Overlaps
            # Left: max(0, ri - xi)
            # Right: max(0, xi - (w - ri))
            # Bottom: max(0, ri - yi)
            # Top: max(0, yi - (h - ri))
            
            over_left = np.maximum(0, r_v - x_v)
            over_right = np.maximum(0, x_v - (w - r_v))
            over_bottom = np.maximum(0, r_v - y_v)
            over_top = np.maximum(0, y_v - (h - r_v))
            
            penalty_walls = np.sum(over_left**2) + np.sum(over_right**2) + np.sum(over_bottom**2) + np.sum(over_top**2)
            
            # Total objective
            # Scale penalty heavily to ensure validity
            return val + 10.0 * (penalty_cc + penalty_walls)
        
        def gradient(v):
            # Numerical gradient is safer and simpler for this small N
            # But L-BFGS-B can use numerical gradient. Let's provide it for speed if possible.
            # Actually, for N=21, 63 variables, numerical gradient is fine.
            # Let's rely on scipy's numerical gradient to avoid coding errors.
            return None # Let scipy handle it? No, minimize needs grad or uses finite diff.
            # Default method 'BFGS' uses finite diff. 'L-BFGS-B' also can.
            # Let's just return None and let scipy use finite diff, or implement a simple one.
            # Finite diff on 63 vars x 1000 iters is fast enough.
        
        # Use Nelder-Mead or L-BFGS-B
        # L-BFGS-B is better for unconstrained (with soft constraints)
        # Bounds:
        # x in [0, w], y in [0, h], r in [0, min(w,h)/2]
        bounds = []
        for i in range(N):
            bounds.append((0, w)) # x
            bounds.append((0, h)) # y
            bounds.append((0, min(w, h) / 2.0)) # r
        
        # Run optimization
        # We run multiple restarts with different initial conditions if needed, 
        # but grid init is usually good enough for N=21.
        
        res = minimize(objective, v0, method='L-BFGS-B', bounds=bounds, 
                       options={'maxiter': 2000, 'ftol': 1e-10, 'gtol': 1e-8})
        
        if res.success or res.fun < 1e-5: # Check if penalty is low
            x_opt = res.x[0:N]
            y_opt = res.x[N:2*N]
            r_opt = res.x[2*N:3*N]
            
            # Validate strictly
            if _validate_packing(x_opt, y_opt, r_opt, w, h):
                val = np.sum(r_opt)
                if val > best_val:
                    best_val = val
                    best_result = np.column_stack([x_opt, y_opt, r_opt])
        
        # Try a second pass with Nelder-Mead if L-BFGS-B got stuck
        # Or just refine with Nelder-Mead
        res_nm = minimize(objective, res.x, method='Nelder-Mead', 
                          options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8})
        
        if res_nm.success:
            x_opt = res_nm.x[0:N]
            y_opt = res_nm.x[N:2*N]
            r_opt = res_nm.x[2*N:3*N]
            
            if _validate_packing(x_opt, y_opt, r_opt, w, h):
                val = np.sum(r_opt)
                if val > best_val:
                    best_val = val
                    best_result = np.column_stack([x_opt, y_opt, r_opt])

    if best_result is not None:
        return best_result
    
    # Fallback: simple grid
    side = int(np.ceil(np.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
    return np.array(cells[:N])


def _validate_packing(x, y, r, w, h, tol=1e-5):
    """Check if packing is valid."""
    for i in range(len(x)):
        if r[i] < -tol:
            return False
        if x[i] - r[i] < -tol or x[i] + r[i] > w + tol:
            return False
        if y[i] - r[i] < -tol or y[i] + r[i] > h + tol:
            return False
    
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist_sq = dx * dx + dy * dy
            min_dist = r[i] + r[j]
            if dist_sq < min_dist * min_dist - tol:
                return False
    
    return True

# EVOLVE-BLOCK-END
