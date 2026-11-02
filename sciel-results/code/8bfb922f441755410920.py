import numpy as np
import time
from scipy.optimize import minimize

N = 32

def construct():
    start_time = time.time()
    time_limit = 110.0  # Safety margin
    
    # M is the weight for maximizing radii
    # If M is too high, it might prioritize invalid large radii over valid small ones.
    # If M is too low, it will find valid small radii.
    # We want to maximize sum(r) subject to validity.
    # The penalty for invalidity should dominate.
    M = 1000.0
    
    def cost_and_grad(z):
        """
        z: array of shape (3*N,) containing [x1, y1, r1, ..., xN, yN, rN]
        Returns: cost, gradient
        """
        x = z[0::3]
        y = z[1::3]
        r = z[2::3]
        
        cost = 0.0
        grad = np.zeros_like(z)
        
        # 1. Overlap Penalties
        # Sum over i < j of max(0, ri + rj - dist_ij)^2
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist_sq = dx*dx + dy*dy
                dist = np.sqrt(dist_sq) if dist_sq > 1e-12 else 1e-12
                
                overlap = r[i] + r[j] - dist
                if overlap > 0:
                    # Penalty term: overlap^2
                    cost += overlap * overlap
                    
                    # Gradient w.r.t ri: 2 * overlap
                    grad[2*i + 2] += 2.0 * overlap
                    # Gradient w.r.t rj: 2 * overlap
                    grad[2*j + 2] += 2.0 * overlap
                    
                    # Gradient w.r.t xi:
                    # d(overlap)/dxi = -d(dist)/dxi = -dx/dist
                    # d(overlap^2)/dxi = 2*overlap * (-dx/dist)
                    factor = 2.0 * overlap / dist
                    grad[2*i] += -factor * dx
                    grad[2*j] += factor * dx
                    
                    # Gradient w.r.t yi:
                    grad[2*i + 1] += -factor * dy
                    grad[2*j + 1] += factor * dy
        
        # 2. Boundary Penalties
        # xi must be in [ri, 1-ri]
        # yi must be in [ri, 1-ri]
        for i in range(N):
            # Left: xi >= ri  =>  ri - xi <= 0
            viol_l = r[i] - x[i]
            if viol_l > 0:
                cost += viol_l * viol_l
                grad[2*i] += -2.0 * viol_l  # d(viol^2)/dx = 2*viol*(-1)
                grad[2*i + 2] += 2.0 * viol_l # d(viol^2)/dr = 2*viol*(1)
                
            # Right: xi <= 1 - ri => 1 - ri - xi <= 0
            viol_r = 1.0 - r[i] - x[i]
            if viol_r > 0:
                cost += viol_r * viol_r
                grad[2*i] += -2.0 * viol_r
                grad[2*i + 2] += -2.0 * viol_r # d(1-r-x)/dr = -1, so 2*viol*(-1)
                
            # Bottom: yi >= ri
            viol_b = r[i] - y[i]
            if viol_b > 0:
                cost += viol_b * viol_b
                grad[2*i + 1] += -2.0 * viol_b
                grad[2*i + 2] += 2.0 * viol_b
                
            # Top: yi <= 1 - ri
            viol_t = 1.0 - r[i] - y[i]
            if viol_t > 0:
                cost += viol_t * viol_t
                grad[2*i + 1] += -2.0 * viol_t
                grad[2*i + 2] += -2.0 * viol_t
                
        # 3. Objective: Maximize sum(r) => Minimize -M * sum(r)
        cost -= M * np.sum(r)
        grad[2::3] -= M
        
        return cost, grad

    def is_valid(z, tol=1e-8):
        x = z[0::3]
        y = z[1::3]
        r = z[2::3]
        
        # Check boundaries
        if np.any(x < r - tol) or np.any(x > 1.0 - r + tol):
            return False
        if np.any(y < r - tol) or np.any(y > 1.0 - r + tol):
            return False
            
        # Check overlaps
        for i in range(N):
            for j in range(i + 1, N):
                dist = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                if dist < r[i] + r[j] - tol:
                    return False
        return True

    def get_score(z):
        r = z[2::3]
        return np.sum(r)

    best_score = 0.0
    best_z = None
    
    def try_optimize(init_z, max_iters=200):
        nonlocal best_score, best_z
        
        # Bounds for L-BFGS-B
        # x, y in [0, 1]
        # r in [0, 0.5]
        bounds = []
        for i in range(N):
            bounds.append((0.0, 1.0)) # x
            bounds.append((0.0, 1.0)) # y
            bounds.append((0.0, 0.5)) # r
            
        # Callback to check for improvement
        def callback(zk):
            nonlocal best_score, best_z
            if is_valid(zk):
                score = get_score(zk)
                if score > best_score:
                    best_score = score
                    best_z = zk.copy()
            return False # Continue optimization
        
        try:
            res = minimize(
                cost_and_grad, 
                init_z, 
                method='L-BFGS-B', 
                jac=True, # We provide gradient
                bounds=bounds, 
                options={'maxiter': max_iters, 'ftol': 1e-9, 'gtol': 1e-6, 'maxls': 50},
                callback=callback
            )
        except Exception:
            pass
            
        # Check final result
        if is_valid(res.x):
            score = get_score(res.x)
            if score > best_score:
                best_score = score
                best_z = res.x.copy()

    # Initialization Strategies
    np.random.seed(42)
    
    # Strategy 1: Random
    for restart in range(10):
        if time.time() - start_time > time_limit:
            break
        x = np.random.rand(N) * 0.8 + 0.1
        y = np.random.rand(N) * 0.8 + 0.1
        r = np.full(N, 0.05)
        z = np.zeros(3*N)
        z[0::3] = x
        z[1::3] = y
        z[2::3] = r
        try_optimize(z, max_iters=300)
        
    # Strategy 2: Grid
    side = 6
    grid_x = []
    grid_y = []
    for i in range(side):
        for j in range(side):
            grid_x.append((i + 0.5) / side)
            grid_y.append((j + 0.5) / side)
            
    for restart in range(5):
        if time.time() - start_time > time_limit:
            break
        # Jitter grid
        x = np.array(grid_x[:N]) + np.random.rand(N) * 0.05
        y = np.array(grid_y[:N]) + np.random.rand(N) * 0.05
        x = np.clip(x, 0.05, 0.95)
        y = np.clip(y, 0.05, 0.95)
        r = np.full(N, 0.05)
        z = np.zeros(3*N)
        z[0::3] = x
        z[1::3] = y
        z[2::3] = r
        try_optimize(z, max_iters=300)

    # Strategy 3: Hexagonal approximation
    # Create a hexagonal grid of points
    hex_points = []
    rows = 7
    cols = 6
    for i in range(rows):
        for j in range(cols):
            x = (j + (i % 2) * 0.5) * (1.0 / (cols - 0.5))
            y = i * (1.0 / (rows - 1)) * 0.866
            if x < 1.0 and y < 1.0:
                hex_points.append((x, y))
                
    # Pick N points
    if len(hex_points) >= N:
        # Shuffle and pick
        np.random.shuffle(hex_points)
        hex_points = hex_points[:N]
        
        for restart in range(5):
            if time.time() - start_time > time_limit:
                break
            x = np.array([p[0] for p in hex_points]) + np.random.rand(N) * 0.02
            y = np.array([p[1] for p in hex_points]) + np.random.rand(N) * 0.02
            x = np.clip(x, 0.05, 0.95)
            y = np.clip(y, 0.05, 0.95)
            r = np.full(N, 0.05)
            z = np.zeros(3*N)
            z[0::3] = x
            z[1::3] = y
            z[2::3] = r
            try_optimize(z, max_iters=300)

    if best_z is None:
        # Fallback: Simple grid
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) < N:
                    cells.append([(i + 0.5) / side, (j + 0.5) / side, 0.4 / side])
        best_z = np.array([v for c in cells for v in c]).reshape(N, 3).T.flatten()
        
    # Final cleanup: Ensure strict validity
    # If the best solution is slightly invalid due to floating point, shrink radii
    if best_z is not None:
        # Check if valid
        if not is_valid(best_z):
            # Try to fix by shrinking
            r = best_z[2::3]
            for scale in [0.99, 0.98, 0.95]:
                r_new = r * scale
                z_new = best_z.copy()
                z_new[2::3] = r_new
                if is_valid(z_new):
                    best_z = z_new
                    break
            
    # Format output
    result = best_z.reshape(N, 3).T
    return result
