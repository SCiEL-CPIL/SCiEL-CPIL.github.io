"""Circle packing in perimeter-4 rectangle: pack 21 circles, maximize sum of radii."""
N = 21
import numpy as np
from scipy.optimize import minimize
import time

# EVOLVE-BLOCK-START

def _pairwise_overlap(x, y, r):
    """Check if any pair of circles overlaps."""
    n = len(x)
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] - 1e-12:
                return True
    return False

def _in_bounds(x, y, r, W, H):
    """Check if all circles are within bounds."""
    for i in range(len(x)):
        if x[i] < r[i] - 1e-12 or x[i] > W - r[i] + 1e-12:
            return False
        if y[i] < r[i] - 1e-12 or y[i] > H - r[i] + 1e-12:
            return False
    return True

def _force_directed_relax(x, y, r, W, H, n_iter=2000, dt=0.001):
    """Relax positions using force-directed method with collision resolution."""
    x = np.array(x, dtype=float).copy()
    y = np.array(y, dtype=float).copy()
    n = len(x)
    
    for it in range(n_iter):
        # Compute forces
        fx = np.zeros(n)
        fy = np.zeros(n)
        
        # Repulsion between overlapping circles
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = r[i] + r[j]
                if dist < min_dist and dist > 1e-12:
                    # Strong repulsive force
                    overlap = min_dist - dist
                    force = 10.0 * overlap
                    fx[i] += force * dx / dist
                    fy[i] += force * dy / dist
                    fx[j] -= force * dx / dist
                    fy[j] -= force * dy / dist
                elif dist <= 1e-12:
                    # Perturb if exactly overlapping
                    fx[i] += 0.01 * (np.random.rand() - 0.5)
                    fy[i] += 0.01 * (np.random.rand() - 0.5)
                    fx[j] -= 0.01 * (np.random.rand() - 0.5)
                    fy[j] -= 0.01 * (np.random.rand() - 0.5)
        
        # Wall repulsion
        for i in range(n):
            # Left wall
            if x[i] < r[i]:
                overlap = r[i] - x[i]
                fx[i] += 10.0 * overlap
            # Right wall
            if x[i] > W - r[i]:
                overlap = x[i] - (W - r[i])
                fx[i] -= 10.0 * overlap
            # Bottom wall
            if y[i] < r[i]:
                overlap = r[i] - y[i]
                fy[i] += 10.0 * overlap
            # Top wall
            if y[i] > H - r[i]:
                overlap = y[i] - (H - r[i])
                fy[i] -= 10.0 * overlap
        
        # Update positions
        x += dt * fx
        y += dt * fy
        
        # Project back to bounds
        for i in range(n):
            x[i] = np.clip(x[i], r[i], W - r[i])
            y[i] = np.clip(y[i], r[i], H - r[i])
    
    return x, y

def _slsqp_optimize(x, y, r, W, H, max_iter=500):
    """Use SLSQP to optimize positions for given radii."""
    n = len(x)
    x0 = np.concatenate([x, y])
    
    def objective(v):
        xi = v[:n]
        yi = v[n:2*n]
        # Minimize negative sum of radii (we want to maximize, but radii are fixed here)
        # Actually, we optimize positions to minimize constraint violations
        # Use a penalty for overlaps and wall violations
        penalty = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                dx = xi[i] - xi[j]
                dy = yi[i] - yi[j]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = r[i] + r[j]
                if dist < min_dist:
                    penalty += (min_dist - dist) ** 2
            # Wall penalties
            if xi[i] < r[i]:
                penalty += (r[i] - xi[i]) ** 2
            if xi[i] > W - r[i]:
                penalty += (xi[i] - (W - r[i])) ** 2
            if yi[i] < r[i]:
                penalty += (r[i] - yi[i]) ** 2
            if yi[i] > H - r[i]:
                penalty += (yi[i] - (H - r[i])) ** 2
        return penalty
    
    # Constraints: keep circles within bounds
    bounds = []
    for i in range(n):
        bounds.append((r[i], W - r[i]))
        bounds.append((r[i], H - r[i]))
    
    result = minimize(objective, x0, method='SLSQP', bounds=bounds, 
                      options={'maxiter': max_iter, 'ftol': 1e-16})
    
    if result.success:
        xi = result.x[:n]
        yi = result.x[n:2*n]
        return xi, yi
    else:
        return x, y

def _expand_circles(x, y, r, W, H, n_iter=50, tol=1e-12):
    """Iteratively expand circles while maintaining feasibility."""
    r = np.array(r, dtype=float).copy()
    x = np.array(x, dtype=float).copy()
    y = np.array(y, dtype=float).copy()
    n = len(x)
    
    for it in range(n_iter):
        improved = False
        for i in range(n):
            # Try to expand circle i
            # Find maximum expansion possible
            max_exp = 0.0
            
            # Check walls
            max_exp = min(max_exp, W - x[i] - r[i])
            max_exp = min(max_exp, x[i] - r[i])
            max_exp = min(max_exp, H - y[i] - r[i])
            max_exp = min(max_exp, y[i] - r[i])
            
            # Check other circles
            for j in range(n):
                if i == j:
                    continue
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx * dx + dy * dy)
                # Maximum expansion such that dist >= (r[i]+dr) + r[j]
                # dr <= dist - r[i] - r[j]
                max_exp = min(max_exp, dist - r[i] - r[j])
            
            if max_exp > tol:
                # Expand by a fraction of max
                dr = 0.5 * max_exp
                r[i] += dr
                improved = True
        
        if not improved:
            break
    
    # Final SLSQP polish after expansion
    x, y = _slsqp_optimize(x, y, r, W, H, max_iter=200)
    
    return x, y, r

def _local_search_polish(x, y, r, W, H, n_steps=100):
    """Adaptive local search with geometric step decay."""
    x = np.array(x, dtype=float).copy()
    y = np.array(y, dtype=float).copy()
    n = len(x)
    
    step_size = 0.01
    directions = [(np.cos(2*np.pi*k/16), np.sin(2*np.pi*k/16)) for k in range(16)]
    
    for step in range(n_steps):
        step_size *= 0.95  # Geometric decay
        if step_size < 1e-6:
            break
        
        for i in range(n):
            best_x, best_y = x[i], y[i]
            best_score = -1.0
            
            # Compute current "score" = sum of distances to constraints
            for dx, dy in directions:
                nx = x[i] + step_size * dx
                ny = y[i] + step_size * dy
                
                # Check bounds
                if nx < r[i] or nx > W - r[i]:
                    continue
                if ny < r[i] or ny > H - r[i]:
                    continue
                
                # Check overlaps with all other circles
                valid = True
                min_dist = float('inf')
                for j in range(n):
                    if i == j:
                        continue
                    d = np.sqrt((nx - x[j])**2 + (ny - y[j])**2)
                    if d < r[i] + r[j] - 1e-12:
                        valid = False
                        break
                    min_dist = min(min_dist, d - r[i] - r[j])
                
                if valid:
                    score = min_dist
                    if score > best_score:
                        best_score = score
                        best_x, best_y = nx, ny
            
            x[i], y[i] = best_x, best_y
    
    return x, y

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Best solution found
    best_sum = 0.0
    best_x = None
    best_y = None
    best_r = None
    best_W = 1.0
    best_H = 1.0
    
    # Try different aspect ratios
    # Perimeter: 2(W+H) = 4 => W + H = 2
    # Try W from 0.5 to 1.5 (H = 2 - W)
    aspect_ratios = [1.0, 0.9, 1.1, 0.8, 1.2, 0.7, 1.3, 0.6, 1.4]
    
    for W in aspect_ratios:
        if time.time() > deadline:
            break
        H = 2.0 - W
        if H <= 0:
            continue
        
        # Initial placement: grid
        side = int(np.ceil(np.sqrt(N)))
        # For N=21, side=5 gives 25 cells, use 21
        xs_init = []
        ys_init = []
        for i in range(side):
            for j in range(side):
                if len(xs_init) >= N:
                    break
                xs_init.append((i + 0.5) * W / side)
                ys_init.append((j + 0.5) * H / side)
        
        x_init = np.array(xs_init[:N])
        y_init = np.array(ys_init[:N])
        
        # Initial radii: uniform based on grid spacing
        # Approximate radius from grid
        r_init = np.full(N, 0.5 * min(W, H) / side * 0.8)
        
        # Force-directed relaxation
        if time.time() > deadline:
            break
        x_fd, y_fd = _force_directed_relax(x_init, y_init, r_init, W, H, n_iter=3000)
        
        # SLSQP optimization
        if time.time() > deadline:
            break
        x_sl, y_sl = _slsqp_optimize(x_fd, y_fd, r_init, W, H, max_iter=300)
        
        # Expand circles
        if time.time() > deadline:
            break
        x_exp, y_exp, r_exp = _expand_circles(x_sl, y_sl, r_init, W, H, n_iter=30)
        
        # Local search polish
        if time.time() > deadline:
            break
        x_ls, y_ls = _local_search_polish(x_exp, y_exp, r_exp, W, H, n_steps=50)
        
        # Final SLSQP after local search
        if time.time() > deadline:
            break
        x_final, y_final = _slsqp_optimize(x_ls, y_ls, r_exp, W, H, max_iter=200)
        
        # One more expansion
        if time.time() > deadline:
            break
        x_final, y_final, r_final = _expand_circles(x_final, y_final, r_exp, W, H, n_iter=20)
        
        # Verify feasibility
        if not _in_bounds(x_final, y_final, r_final, W, H):
            continue
        if _pairwise_overlap(x_final, y_final, r_final):
            continue
        
        sum_r = np.sum(r_final)
        if sum_r > best_sum:
            best_sum = sum_r
            best_x = x_final.copy()
            best_y = y_final.copy()
            best_r = r_final.copy()
            best_W = W
            best_H = H
    
    # If no valid solution found, fall back to simple grid
    if best_x is None:
        W, H = 1.0, 1.0
        side = 5
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) >= N:
                    break
                cells.append(((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3))
        best_x = np.array([c[0] for c in cells])
        best_y = np.array([c[1] for c in cells])
        best_r = np.array([c[2] for c in cells])
    
    # Return array of shape (n, 3): [x, y, r]
    result = np.column_stack([best_x, best_y, best_r])
    return result

# EVOLVE-BLOCK-END
