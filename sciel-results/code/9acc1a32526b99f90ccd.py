import numpy as np
from scipy.optimize import minimize

N = 26

def construct():
    def grid_initialize(n):
        """Initialize circles on a grid pattern for better starting configuration."""
        # Use a grid-based initialization with some randomness
        # Try different grid sizes and pick the best
        best_circles = None
        best_sum = -1
        
        for grid_size in [3, 4, 5, 6]:
            # Create grid points
            points = []
            for i in range(grid_size):
                for j in range(grid_size):
                    x = (i + 0.5) / grid_size
                    y = (j + 0.5) / grid_size
                    points.append((x, y))
            
            # Shuffle and take n points
            np.random.shuffle(points)
            if len(points) < n:
                # Add more random points
                while len(points) < n:
                    points.append((np.random.uniform(0.1, 0.9), np.random.uniform(0.1, 0.9)))
            
            selected = points[:n]
            
            # Compute initial radii based on distance to nearest neighbors and boundaries
            circles = []
            for x, y in selected:
                r_bound = min(x, 1-x, y, 1-y)
                r_min = r_bound
                for x2, y2 in selected:
                    if x2 != x or y2 != y:
                        dist = np.sqrt((x-x2)**2 + (y-y2)**2)
                        r_min = min(r_min, dist * 0.4)
                r = max(0.01, min(r_min, 0.15))
                circles.append((x, y, r))
            
            # Check validity and compute sum
            valid = True
            for i in range(n):
                x, y, r = circles[i]
                if r > x or r > 1-x or r > y or r > 1-y:
                    valid = False
                    break
                for j in range(i+1, n):
                    x2, y2, r2 = circles[j]
                    dist = np.sqrt((x-x2)**2 + (y-y2)**2)
                    if dist < r + r2:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                s = sum(c[2] for c in circles)
                if s > best_sum:
                    best_sum = s
                    best_circles = circles
        
        if best_circles is None:
            # Fallback: simple grid
            circles = []
            for i in range(5):
                for j in range(5):
                    if len(circles) < n:
                        x = (i + 0.5) / 5
                        y = (j + 0.5) / 5
                        circles.append((x, y, 0.08))
            best_circles = circles[:n]
        
        return np.array(best_circles)
    
    def energy(params):
        x_ = params[0::3]
        y_ = params[1::3]
        r_ = params[2::3]
        
        # Boundary violations
        pen_bound = 0.0
        for i in range(N):
            pen_bound += max(0, r_[i] - x_[i])**2
            pen_bound += max(0, r_[i] - (1 - x_[i]))**2
            pen_bound += max(0, r_[i] - y_[i])**2
            pen_bound += max(0, r_[i] - (1 - y_[i]))**2
            
        # Overlap violations
        pen_overlap = 0.0
        for i in range(N):
            for j in range(i+1, N):
                dx = x_[i] - x_[j]
                dy = y_[i] - y_[j]
                dist_sq = dx*dx + dy*dy
                min_dist = r_[i] + r_[j]
                if dist_sq < min_dist**2:
                    overlap = min_dist - np.sqrt(dist_sq)
                    pen_overlap += overlap**2
                    
        lam = 200.0
        return -np.sum(r_) + lam * (pen_bound + pen_overlap)
    
    def gradient(params):
        """Compute analytical gradient of the energy function."""
        x_ = params[0::3]
        y_ = params[1::3]
        r_ = params[2::3]
        
        grad = np.zeros_like(params)
        lam = 200.0
        
        # Objective: -sum(r_i)
        for i in range(N):
            grad[2*i+2] = -1.0
        
        # Boundary penalties
        for i in range(N):
            # x-boundary
            if r_[i] > x_[i]:
                grad[3*i] += 2*lam*(r_[i] - x_[i])
                grad[3*i+2] += 2*lam*(r_[i] - x_[i])
            if r_[i] > 1-x_[i]:
                grad[3*i] -= 2*lam*(r_[i] - (1-x_[i]))
                grad[3*i+2] += 2*lam*(r_[i] - (1-x_[i]))
            # y-boundary
            if r_[i] > y_[i]:
                grad[3*i+1] += 2*lam*(r_[i] - y_[i])
                grad[3*i+2] += 2*lam*(r_[i] - y_[i])
            if r_[i] > 1-y_[i]:
                grad[3*i+1] -= 2*lam*(r_[i] - (1-y_[i]))
                grad[3*i+2] += 2*lam*(r_[i] - (1-y_[i]))
        
        # Overlap penalties
        for i in range(N):
            for j in range(i+1, N):
                dx = x_[i] - x_[j]
                dy = y_[i] - y_[j]
                dist_sq = dx*dx + dy*dy
                min_dist = r_[i] + r_[j]
                if dist_sq > 1e-12 and dist_sq < min_dist**2:
                    dist = np.sqrt(dist_sq)
                    overlap = min_dist - dist
                    # Gradient of overlap^2 w.r.t. x_i, y_i, r_i, x_j, y_j, r_j
                    # d(overlap^2)/d(x_i) = 2*overlap * (-dx/dist)
                    # d(overlap^2)/d(y_i) = 2*overlap * (-dy/dist)
                    # d(overlap^2)/d(r_i) = 2*overlap * 1
                    # d(overlap^2)/d(x_j) = 2*overlap * (dx/dist)
                    # d(overlap^2)/d(y_j) = 2*overlap * (dy/dist)
                    # d(overlap^2)/d(r_j) = 2*overlap * 1
                    factor = 2*lam*overlap/dist
                    grad[3*i] += factor * dx
                    grad[3*i+1] += factor * dy
                    grad[3*i+2] += 2*lam*overlap
                    grad[3*j] -= factor * dx
                    grad[3*j+1] -= factor * dy
                    grad[3*j+2] += 2*lam*overlap
        
        return grad
    
    # Try multiple initializations and pick the best
    best_result = None
    best_value = -1
    
    for attempt in range(3):
        init_circles = grid_initialize(N)
        
        x0 = init_circles[:, 0]
        y0 = init_circles[:, 1]
        r0 = init_circles[:, 2]
        
        params0 = np.zeros(3*N)
        params0[0::3] = x0
        params0[1::3] = y0
        params0[2::3] = r0
        
        bounds = []
        for i in range(N):
            bounds.append((0.01, 0.99))
            bounds.append((0.01, 0.99))
            bounds.append((0.005, 0.3))
        
        # Run optimization
        res = minimize(energy, params0, jac=gradient, method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-6})
        
        params_opt = res.x
        x_opt = params_opt[0::3]
        y_opt = params_opt[1::3]
        r_opt = params_opt[2::3]
        
        # Post-processing: Ensure strict validity
        for _ in range(20):
            changed = False
            # Clamp to boundaries
            for i in range(N):
                r_max_bound = min(x_opt[i], 1-x_opt[i], y_opt[i], 1-y_opt[i])
                if r_opt[i] > r_max_bound:
                    r_opt[i] = max(0.001, r_max_bound * 0.999)
                    changed = True
                r_opt[i] = max(0.001, r_opt[i])
            
            # Fix overlaps
            for i in range(N):
                for j in range(i+1, N):
                    dist = np.sqrt((x_opt[i]-x_opt[j])**2 + (y_opt[i]-y_opt[j])**2)
                    min_dist = r_opt[i] + r_opt[j]
                    if dist < min_dist:
                        # Scale both radii down
                        if min_dist > 0:
                            scale = dist / min_dist
                            r_opt[i] *= scale
                            r_opt[j] *= scale
                            changed = True
                        else:
                            r_opt[i] = max(0.001, r_opt[i] * 0.9)
                            r_opt[j] = max(0.001, r_opt[j] * 0.9)
                            changed = True
            
            if not changed:
                break
        
        # Final clamp
        for i in range(N):
            r_opt[i] = min(r_opt[i], x_opt[i]*0.999, (1-x_opt[i])*0.999, y_opt[i]*0.999, (1-y_opt[i])*0.999)
            r_opt[i] = max(0.001, r_opt[i])
        
        # Verify validity
        valid = True
        for i in range(N):
            x, y, r = x_opt[i], y_opt[i], r_opt[i]
            if r > x or r > 1-x or r > y or r > 1-y or r < 0.001:
                valid = False
                break
            for j in range(i+1, N):
                dist = np.sqrt((x_opt[i]-x_opt[j])**2 + (y_opt[i]-y_opt[j])**2)
                if dist < r_opt[i] + r_opt[j] - 1e-9:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            value = np.sum(r_opt)
            if value > best_value:
                best_value = value
                best_result = np.column_stack([x_opt, y_opt, r_opt])
    
    if best_result is None:
        # Fallback: simple grid
        result = np.zeros((N, 3))
        for i in range(N):
            row = i // 5
            col = i % 5
            result[i, 0] = (col + 0.5) / 5
            result[i, 1] = (row + 0.5) / 5
            result[i, 2] = 0.08
        return result
    
    return best_result
