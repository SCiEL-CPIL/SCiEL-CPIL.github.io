"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 32

# EVOLVE-BLOCK-START
import numpy as np
import time

def _hex_init(n):
    """Generate a hexagonal close-packed initial configuration for n circles in unit square."""
    # For hexagonal packing, rows alternate offset
    # We want to fit as many as possible, then trim/adjust for n
    # Start with a rough grid, then refine positions
    
    # Hexagonal lattice: horizontal spacing = 2r, vertical spacing = sqrt(3)*r
    # For a square container, we need to choose r such that we can fit n circles
    
    # Let's try different aspect ratios and find a good starting point
    # Use a simple approach: create a hexagonal grid, then optimize positions
    
    # Generate hexagonal points
    # For n=32, a good arrangement might be 6x6 grid with some removed, or 5x7, etc.
    # Let's try a few configurations
    
    best_points = None
    best_score = -1
    
    for rows in range(4, 9):
        for cols in range(4, 9):
            if rows * cols < n:
                continue
            # Create hexagonal grid
            points = []
            for i in range(rows):
                for j in range(cols):
                    # Hexagonal offset
                    x = (j + (i % 2) * 0.5) * 2.0 / (cols - 0.5) if cols > 1 else 0.5
                    y = (i + 0.5) * 2.0 / (rows + 0.5 * (rows % 2)) if rows > 1 else 0.5
                    points.append([x, y])
            
            if len(points) < n:
                continue
            
            # Take first n points (or best n)
            pts = np.array(points[:n])
            
            # Score: minimum distance to boundary and between circles
            # Boundary distances
            min_boundary = np.min(np.min(pts, axis=0), np.min(1 - pts, axis=0))
            
            # Pairwise distances
            min_pair = np.inf
            for i in range(len(pts)):
                for j in range(i+1, len(pts)):
                    d = np.linalg.norm(pts[i] - pts[j])
                    min_pair = min(min_pair, d)
            
            score = min(min_boundary, min_pair / 2)
            if score > best_score:
                best_score = score
                best_points = pts
    
    if best_points is None:
        # Fallback to grid
        side = int(np.ceil(np.sqrt(n)))
        points = []
        for i in range(side):
            for j in range(side):
                points.append([(i + 0.5) / side, (j + 0.5) / side])
        best_points = np.array(points[:n])
    
    return best_points

def construct():
    """Construct 32 non-overlapping circles in unit square maximizing sum of radii."""
    start_time = time.time()
    deadline = start_time + 115  # Leave 5s safety margin
    
    # Initialize with hexagonal packing
    pts = _hex_init(N)
    
    # Initial radius: small enough to guarantee no overlap
    # Find minimum distance to boundary and between points
    min_boundary = np.min(np.min(pts, axis=0), np.min(1 - pts, axis=0))
    min_pair = np.inf
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(pts[i] - pts[j])
            min_pair = min(min_pair, d)
    
    r0 = min(min_boundary, min_pair / 2) * 0.9
    radii = np.full(N, r0)
    
    # Use gradient ascent with penalty method
    # Objective: maximize sum(radii) - lambda * penalty(constraints)
    # Constraints: for each pair i,j: dist(i,j) >= r_i + r_j
    #              for each i: r_i <= min(x_i, y_i, 1-x_i, 1-y_i)
    
    # We'll use a simple iterative approach:
    # 1. Try to increase all radii by small delta
    # 2. If any constraint violated, reduce radii of violating circles and adjust positions
    
    # Better: use L-BFGS-B with smooth penalty
    from scipy.optimize import minimize
    
    def _objective_and_grad(params):
        """params: [x_1, y_1, r_1, x_2, y_2, r_2, ...]
        Returns (objective, gradient) for maximization via negation."""
        x = params[0::3]
        y = params[1::3]
        r = params[2::3]
        
        # Objective: -sum(r) + penalty
        obj = -np.sum(r)
        grad_x = np.zeros_like(x)
        grad_y = np.zeros_like(y)
        grad_r = -np.ones_like(r)
        
        # Penalty for pair constraints: dist(i,j) < r_i + r_j
        # Use soft penalty: max(0, r_i + r_j - dist)^2
        mu = 10.0  # penalty weight
        
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx*dx + dy*dy)
                if d < 1e-10:
                    d = 1e-10
                violation = r[i] + r[j] - d
                if violation > 0:
                    # Penalty: mu * violation^2
                    obj += mu * violation * violation
                    # Gradient
                    grad_factor = 2 * mu * violation
                    grad_x[i] += grad_factor * dx / d
                    grad_y[i] += grad_factor * dy / d
                    grad_x[j] -= grad_factor * dx / d
                    grad_y[j] -= grad_factor * dy / d
                    grad_r[i] += grad_factor
                    grad_r[j] += grad_factor
        
        # Penalty for boundary constraints: r_i > min(x_i, y_i, 1-x_i, 1-y_i)
        for i in range(N):
            for coord, val in [('x', x[i]), ('y', y[i])]:
                # Left/bottom: r > val
                viol = r[i] - val
                if viol > 0:
                    obj += mu * viol * viol
                    grad_factor = 2 * mu * viol
                    if coord == 'x':
                        grad_x[i] -= grad_factor
                    else:
                        grad_y[i] -= grad_factor
                    grad_r[i] += grad_factor
                # Right/top: r > 1 - val
                viol = r[i] - (1 - val)
                if viol > 0:
                    obj += mu * viol * viol
                    grad_factor = 2 * mu * viol
                    if coord == 'x':
                        grad_x[i] += grad_factor
                    else:
                        grad_y[i] += grad_factor
                    grad_r[i] += grad_factor
        
        grad = np.concatenate([grad_x, grad_y, grad_r])
        return obj, grad
    
    # Initial parameters
    params0 = np.concatenate([pts[:, 0], pts[:, 1], radii])
    
    # Run optimization
    best_params = params0.copy()
    best_obj = _objective_and_grad(params0)[0]
    
    # Use L-BFGS-B
    result = minimize(
        _objective_and_grad,
        params0,
        method='L-BFGS-B',
        jac=True,
        options={'maxiter': 200, 'maxfun': 5000, 'ftol': 1e-12, 'gtol': 1e-10}
    )
    
    if time.time() < deadline:
        if result.fun < best_obj:
            best_params = result.x
            best_obj = result.fun
    
    # Extract best solution
    x_best = best_params[0::3]
    y_best = best_params[1::3]
    r_best = best_params[2::3]
    
    # Ensure validity: clip radii and positions
    for i in range(N):
        # Clip position to [r_i, 1-r_i]
        r_best[i] = max(0.01, min(r_best[i], 0.49))
        x_best[i] = max(r_best[i], min(1 - r_best[i], x_best[i]))
        y_best[i] = max(r_best[i], min(1 - r_best[i], y_best[i]))
    
    # Final check and fix overlaps
    for iteration in range(50):
        if time.time() > deadline:
            break
        changed = False
        for i in range(N):
            for j in range(i+1, N):
                dx = x_best[i] - x_best[j]
                dy = y_best[i] - y_best[j]
                d = np.sqrt(dx*dx + dy*dy)
                required = r_best[i] + r_best[j]
                if d < required:
                    # Reduce radii to fix overlap
                    overlap = required - d
                    r_best[i] -= overlap * 0.5
                    r_best[j] -= overlap * 0.5
                    changed = True
        if not changed:
            break
    
    # Ensure minimum radius
    r_best = np.maximum(r_best, 0.01)
    
    # Re-clip positions after radius adjustments
    for i in range(N):
        x_best[i] = max(r_best[i], min(1 - r_best[i], x_best[i]))
        y_best[i] = max(r_best[i], min(1 - r_best[i], y_best[i]))
    
    # Final validity check
    for i in range(N):
        for j in range(i+1, N):
            d = np.sqrt((x_best[i]-x_best[j])**2 + (y_best[i]-y_best[j])**2)
            if d < r_best[i] + r_best[j]:
                # Still overlapping, reduce radii more aggressively
                overlap = (r_best[i] + r_best[j]) - d
                r_best[i] -= overlap
                r_best[j] -= overlap
                r_best[i] = max(0.01, r_best[i])
                r_best[j] = max(0.01, r_best[j])
    
    result_arr = np.column_stack([x_best, y_best, r_best])
    return result_arr

# EVOLVE-BLOCK-END
