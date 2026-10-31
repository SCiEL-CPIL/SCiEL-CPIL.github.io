import numpy as np
import time
from scipy.optimize import minimize
import sys

N = 21

def construct():
    start_time = time.time()
    timeout = 110.0  # Safety margin for 120s limit
    
    best_score = 0.0
    best_solution = None
    
    # Objective function to minimize
    def objective(v):
        W = v[0]
        H = 2.0 - W
        if W <= 0 or H <= 0:
            return 1e6
        
        # Extract circle parameters
        # v[1:] is flattened [x1, y1, r1, x2, y2, r2, ...]
        params = v[1:].reshape(N, 3)
        xs = params[:, 0]
        ys = params[:, 1]
        rs = params[:, 2]
        
        # Sum of radii (we want to maximize, so minimize negative)
        sum_r = np.sum(rs)
        
        # Penalty for overlaps
        overlap_pen = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                dist = np.sqrt((xs[i] - xs[j])**2 + (ys[i] - ys[j])**2)
                overlap = rs[i] + rs[j] - dist
                if overlap > 0:
                    overlap_pen += overlap * overlap
        
        # Penalty for boundary violations
        boundary_pen = 0.0
        for i in range(N):
            # Left wall
            if xs[i] < rs[i]:
                boundary_pen += (rs[i] - xs[i])**2
            # Right wall
            if xs[i] + rs[i] > W:
                boundary_pen += (xs[i] + rs[i] - W)**2
            # Bottom wall
            if ys[i] < rs[i]:
                boundary_pen += (rs[i] - ys[i])**2
            # Top wall
            if ys[i] + rs[i] > H:
                boundary_pen += (ys[i] + rs[i] - H)**2
        
        # Weights
        # If penalty is non-zero, we want to push it down.
        # If penalty is 0, we want to maximize sum_r.
        # A common trick is: Objective = -sum_r * weight_sum + penalty * weight_pen
        # To ensure validity, weight_pen should be high initially or dynamically adjusted.
        # For simplicity in a fixed budget, a high weight for penalty is safer.
        weight_pen = 10.0
        weight_sum = 1.0
        
        return -weight_sum * sum_r + weight_pen * (overlap_pen + boundary_pen)

    # Gradient approximation is slow, L-BFGS-B uses numerical diff if not provided.
    # For N=21, dimensions = 1 + 21*3 = 64. Numerical diff is 65 evaluations per step.
    # This might be slow. Let's try to provide an analytical gradient or use COBYLA?
    # COBYLA is derivative-free and often good for small N.
    # Let's use COBYLA or Nelder-Mead?
    # L-BFGS-B with numerical grad: ~65 evals/step. 1000 steps = 65000 evals.
    # Each eval is O(N^2) ~ 400 ops. 26M ops. Should be fast in Python?
    # Python loops are slow. Vectorization is key.
    
    def objective_vec(v):
        W = v[0]
        H = 2.0 - W
        if W <= 0.01 or H <= 0.01:
            return 1e6
            
        params = v[1:].reshape(N, 3)
        xs = params[:, 0]
        ys = params[:, 1]
        rs = params[:, 2]
        
        sum_r = np.sum(rs)
        
        # Vectorized overlap calculation
        # Create distance matrix
        dx = xs[:, np.newaxis] - xs[np.newaxis, :]
        dy = ys[:, np.newaxis] - ys[np.newaxis, :]
        dist = np.sqrt(dx**2 + dy**2)
        
        # Required distance
        req = rs[:, np.newaxis] + rs[np.newaxis, :]
        
        # Overlap is max(0, req - dist)
        # We only care about i < j, but matrix is symmetric.
        # Sum over upper triangle
        iu = np.triu_indices(N, k=1)
        overlaps = req[iu] - dist[iu]
        overlaps = np.maximum(0, overlaps)
        overlap_pen = np.sum(overlaps**2)
        
        # Boundary penalties
        # Left: max(0, r - x)
        pen_left = np.maximum(0, rs - xs)
        # Right: max(0, x + r - W)
        pen_right = np.maximum(0, xs + rs - W)
        # Bottom: max(0, r - y)
        pen_bottom = np.maximum(0, rs - ys)
        # Top: max(0, y + r - H)
        pen_top = np.maximum(0, ys + rs - H)
        
        boundary_pen = np.sum(pen_left**2) + np.sum(pen_right**2) + np.sum(pen_bottom**2) + np.sum(pen_top**2)
        
        weight_pen = 10.0
        weight_sum = 1.0
        
        return -weight_sum * sum_r + weight_pen * (overlap_pen + boundary_pen)

    def get_solution(v):
        W = v[0]
        H = 2.0 - W
        params = v[1:].reshape(N, 3)
        # Return array of shape (N, 3) as required by interface
        # The interface expects (x, y, r)
        return params

    def is_valid(params, W, H):
        # Check if circles are disjoint and inside
        # Tolerance 1e-5
        tol = 1e-5
        xs = params[:, 0]
        ys = params[:, 1]
        rs = params[:, 2]
        
        # Boundary
        if np.any(xs < -tol) or np.any(xs + rs > W + tol):
            return False
        if np.any(ys < -tol) or np.any(ys + rs > H + tol):
            return False
        
        # Overlaps
        for i in range(N):
            for j in range(i+1, N):
                dist = np.sqrt((xs[i]-xs[j])**2 + (ys[i]-ys[j])**2)
                if dist < rs[i] + rs[j] - tol:
                    return False
        return True

    # Initializations
    def init_grid(W, H):
        # Grid packing
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        # Scale to fit
        # Simple grid
        w_step = W / side
        h_step = H / side
        for i in range(side):
            for j in range(side):
                if len(cells) >= N:
                    break
                x = (i + 0.5) * w_step
                y = (j + 0.5) * h_step
                r = 0.5 * min(w_step, h_step) * 0.9 # 10% margin
                cells.append([x, y, r])
        # Fill remaining if N > side*side? No, side*side >= N usually.
        # If N < side*side, we just take the first N.
        # But we have exactly N circles.
        # This creates a grid of side x side.
        return np.array(cells[:N])

    def init_hex(W, H):
        # Hexagonal packing
        # Estimate radius
        # Area per circle ~ pi r^2. Total area W*H.
        # Packing density ~ 0.9.
        # N * pi r^2 * 0.9 <= W*H -> r ~ sqrt(W*H / (N*pi*0.9))
        r_est = np.sqrt(W * H / (N * np.pi * 0.9))
        
        # Hex spacing
        dx = 2 * r_est * 0.95
        dy = 2 * r_est * 0.95 * np.sqrt(3)/2
        
        circles = []
        row = 0
        y = r_est
        while len(circles) < N and y < H - r_est:
            x = r_est
            offset = 0 if row % 2 == 0 else dx/2
            while x < W - r_est and len(circles) < N:
                circles.append([x + offset, y, r_est])
                x += dx
            y += dy
            row += 1
        
        if len(circles) < N:
            # Fallback to grid
            return init_grid(W, H)
            
        return np.array(circles[:N])

    def init_random(W, H):
        # Random placement, small radii
        rs = np.random.uniform(0.05, 0.1, N)
        xs = np.random.uniform(rs, W-rs, N)
        ys = np.random.uniform(rs, H-rs, N)
        return np.column_stack((xs, ys, rs))

    # Main loop
    num_restarts = 50
    best_val = -np.inf
    
    for restart in range(num_restarts):
        if time.time() - start_time > timeout:
            break
            
        # Randomize W
        # Square is often good, but let's try a range
        W = np.random.uniform(0.8, 1.2)
        H = 2.0 - W
        
        # Choose initialization
        init_type = restart % 3
        if init_type == 0:
            params = init_grid(W, H)
        elif init_type == 1:
            params = init_hex(W, H)
        else:
            params = init_random(W, H)
        
        # Add small jitter to break symmetry
        params[:, 0] += np.random.uniform(-0.01, 0.01, N)
        params[:, 1] += np.random.uniform(-0.01, 0.01, N)
        
        # Ensure initial validity roughly
        # If not valid, shrink radii
        for i in range(N):
            # Shrink to fit boundaries
            max_r = min(params[i,0], W-params[i,0], params[i,1], H-params[i,1])
            if params[i,2] > max_r * 0.9:
                params[i,2] = max_r * 0.9
        
        v = np.concatenate(([W], params.flatten()))
        
        # Optimize
        # Use L-BFGS-B
        # Bounds:
        # W: [0.1, 1.9]
        # x, y, r: [0.0, 2.0] (loose)
        bounds = [(0.1, 1.9)] + [(0.0, 2.0)] * (3 * N)
        
        try:
            res = minimize(objective_vec, v, method='L-BFGS-B', bounds=bounds, options={'maxiter': 200, 'ftol': 1e-8})
            
            if res.success or res.fun < best_val:
                # Check validity of result
                W_res = res.x[0]
                H_res = 2.0 - W_res
                params_res = res.x[1:].reshape(N, 3)
                
                if is_valid(params_res, W_res, H_res):
                    val = np.sum(params_res[:, 2])
                    if val > best_val:
                        best_val = val
                        best_solution = params_res
                        
        except Exception:
            continue

    # If no valid solution found, return a fallback (grid)
    if best_solution is None:
        W = 1.0
        H = 1.0
        best_solution = init_grid(W, H)
        # Ensure it's valid
        # Grid with 5x5 for 21 circles
        # 5x5=25.
        # Side 1.0. Step 0.2. R=0.09.
        # 5*0.2=1.0.
        # Fits.
        
    return best_solution
