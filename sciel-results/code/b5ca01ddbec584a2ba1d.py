"""Circle packing in perimeter-4 rectangle - robust multi-start local search."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21

def _check_valid(circles, tol=1e-7):
    """Check if circles are valid: disjoint interiors, bounding rect perimeter <= 4."""
    if circles.shape[0] != N:
        return False
    xs = circles[:, 0]
    ys = circles[:, 1]
    rs = circles[:, 2]
    
    # Radii must be positive
    if np.any(rs < -tol):
        return False
        
    # Bounding box perimeter
    min_x = np.min(xs - rs)
    max_x = np.max(xs + rs)
    min_y = np.min(ys - rs)
    max_y = np.max(ys + rs)
    w = max_x - min_x
    h = max_y - min_y
    if 2 * (w + h) > 4 + tol:
        return False
        
    # Check pairwise separation
    # Vectorized distance calculation for all pairs
    diff_x = xs[:, np.newaxis] - xs[np.newaxis, :]
    diff_y = ys[:, np.newaxis] - ys[np.newaxis, :]
    dists = np.sqrt(diff_x**2 + diff_y**2)
    req_dists = rs[:, np.newaxis] + rs[np.newaxis, :]
    
    # Check if any distance is less than required distance (excluding self)
    # We ignore the diagonal
    np.fill_diagonal(dists, np.inf)
    np.fill_diagonal(req_dists, 0)
    
    if np.any(dists < req_dists - tol):
        return False
        
    return True

def _penalty_objective(params, W, H, lam=1e5):
    """
    Objective: minimize -sum(r) + penalties.
    params: flattened array of shape (3N,)
    """
    n = N
    x = params.reshape(n, 3)
    xs, ys, rs = x[:, 0], x[:, 1], x[:, 2]
    
    # 1. Radii positivity penalty
    neg_r = np.maximum(0, -rs)
    penalty_r = np.sum(neg_r ** 2)
    
    # 2. Boundary penalties
    # Circles must be inside [0, W] x [0, H]
    # Left: x - r >= 0 -> x >= r
    # Right: x + r <= W -> x <= W - r
    # Bottom: y - r >= 0 -> y >= r
    # Top: y + r <= H -> y <= H - r
    
    viol_left = np.maximum(0, rs - xs)
    viol_right = np.maximum(0, xs + rs - W)
    viol_bottom = np.maximum(0, rs - ys)
    viol_top = np.maximum(0, ys + rs - H)
    
    penalty_bound = np.sum(viol_left**2) + np.sum(viol_right**2) + \
                    np.sum(viol_bottom**2) + np.sum(viol_top**2)
    
    # 3. Separation penalties
    penalty_sep = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            dx = xs[j] - xs[i]
            dy = ys[j] - ys[i]
            d = np.sqrt(dx**2 + dy**2)
            req = rs[i] + rs[j]
            viol = max(0, req - d)
            penalty_sep += viol ** 2
            
    total_penalty = lam * (penalty_r + penalty_bound + penalty_sep)
    return -np.sum(rs) + total_penalty

def _local_search(circles, W, H, deadline, rng):
    """
    Perform local search to improve the packing.
    """
    best_circles = circles.copy()
    best_sum = np.sum(circles[:, 2])
    
    # Initial scaling to fit perimeter
    min_x = np.min(best_circles[:, 0] - best_circles[:, 2])
    max_x = np.max(best_circles[:, 0] + best_circles[:, 2])
    min_y = np.min(best_circles[:, 1] - best_circles[:, 2])
    max_y = np.max(best_circles[:, 1] + best_circles[:, 2])
    w_cur = max_x - min_x
    h_cur = max_y - min_y
    perim_cur = 2 * (w_cur + h_cur)
    
    # Scale to fit W, H if possible, otherwise just use as is
    # We want to maximize radii, so we should scale UP if we have room, or scale DOWN if we exceed.
    # Actually, the optimizer will handle the size. We just need to start in a reasonable place.
    
    # Optimization loop
    for step in range(50):
        if time.time() > deadline:
            break
            
        def obj(p):
            return _penalty_objective(p, W, H, lam=1e4)
            
        try:
            res = minimize(obj, best_circles.flatten(), method='L-BFGS-B',
                          options={'maxiter': 100, 'gtol': 1e-6, 'ftol': 1e-8})
            new_circles = res.x.reshape(N, 3)
            
            # Check validity and update best
            # We allow small violations during search, but track the best valid one
            if _check_valid(new_circles, tol=1e-5):
                s = np.sum(new_circles[:, 2])
                if s > best_sum:
                    best_sum = s
                    best_circles = new_circles.copy()
            else:
                # Even if not strictly valid, if the penalty is low, it might be a good local minimum
                # We can still use it as a starting point for the next step
                pass
                
            # Force-directed relaxation to clean up overlaps
            # This helps escape shallow local minima caused by numerical noise
            new_circles = _relax(new_circles, W, H, steps=50)
            if _check_valid(new_circles, tol=1e-5):
                s = np.sum(new_circles[:, 2])
                if s > best_sum:
                    best_sum = s
                    best_circles = new_circles.copy()
                    
        except Exception:
            break
            
    return best_circles, best_sum

def _relax(circles, W, H, steps=100):
    """
    Simple force-directed relaxation to resolve overlaps and boundary violations.
    """
    circles = circles.copy()
    xs = circles[:, 0]
    ys = circles[:, 1]
    rs = circles[:, 2]
    
    for _ in range(steps):
        forces_x = np.zeros(N)
        forces_y = np.zeros(N)
        
        # Overlap forces
        for i in range(N):
            for j in range(i + 1, N):
                dx = xs[j] - xs[i]
                dy = ys[j] - ys[i]
                d = np.sqrt(dx**2 + dy**2)
                req = rs[i] + rs[j]
                if d < req and d > 1e-9:
                    overlap = req - d
                    fx = overlap * dx / d
                    fy = overlap * dy / d
                    forces_x[i] -= fx
                    forces_y[i] -= fy
                    forces_x[j] += fx
                    forces_y[j] += fy
                elif d <= 1e-9:
                    angle = np.random.uniform(0, 2 * np.pi)
                    forces_x[i] -= np.cos(angle)
                    forces_y[i] -= np.sin(angle)
                    forces_x[j] += np.cos(angle)
                    forces_y[j] += np.sin(angle)
                    
        # Boundary forces
        for i in range(N):
            if xs[i] < rs[i]:
                forces_x[i] += (rs[i] - xs[i])
            if xs[i] > W - rs[i]:
                forces_x[i] -= (xs[i] - (W - rs[i]))
            if ys[i] < rs[i]:
                forces_y[i] += (rs[i] - ys[i])
            if ys[i] > H - rs[i]:
                forces_y[i] -= (ys[i] - (H - rs[i]))
                
        # Apply forces
        step_size = 0.05
        xs += step_size * forces_x
        ys += step_size * forces_y
        
        # Clip to bounds (hard constraint)
        xs = np.clip(xs, rs, W - rs)
        ys = np.clip(ys, rs, H - rs)
        
        circles[:, 0] = xs
        circles[:, 1] = ys
        
    return circles

def _make_init(W, H, rng, type='grid'):
    """Generate initial configuration."""
    circles = np.zeros((N, 3))
    
    if type == 'grid':
        # Approximate grid
        nx = 5
        ny = 5
        dx = W / nx
        dy = H / ny
        r = min(dx, dy) * 0.4
        count = 0
        for i in range(nx):
            for j in range(ny):
                if count >= N:
                    break
                cx = (i + 0.5) * dx
                cy = (j + 0.5) * dy
                # Add small jitter
                cx += rng.uniform(-0.01, 0.01)
                cy += rng.uniform(-0.01, 0.01)
                circles[count] = [cx, cy, r]
                count += 1
        # Fill rest
        while count < N:
            circles[count] = [rng.uniform(0, W), rng.uniform(0, H), 0.01]
            count += 1
            
    elif type == 'hex':
        # Hexagonal packing
        r = min(W, H) / 10.0
        dx = 2 * r * 0.9
        dy = dx * np.sqrt(3) / 2
        count = 0
        for j in range(int(H / dy) + 1):
            offset = dx / 2 if j % 2 else 0
            for i in range(int(W / dx) + 1):
                if count >= N:
                    break
                cx = i * dx + offset + r
                cy = j * dy + r
                if cx + r <= W and cy + r <= H:
                    circles[count] = [cx, cy, r]
                    count += 1
        while count < N:
            circles[count] = [rng.uniform(0, W), rng.uniform(0, H), 0.01]
            count += 1
            
    else: # random
        for i in range(N):
            circles[i] = [rng.uniform(0.1, W-0.1), rng.uniform(0.1, H-0.1), 0.05]
            
    return circles

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    rng = np.random.RandomState(42)
    
    best_circles = None
    best_sum = 0.0
    
    # Search over aspect ratios
    # Perimeter 4 => 2(W+H)=4 => W+H=2 => H = 2-W
    # W can range from small to large.
    # Symmetry: W and H are interchangeable. So W in [0.5, 1.5] covers all unique shapes.
    aspect_ratios = np.linspace(0.5, 1.5, 15)
    
    for W in aspect_ratios:
        H = 2.0 - W
        
        # Try different initializations
        for init_type in ['grid', 'hex', 'random']:
            if time.time() > deadline:
                break
                
            init_circles = _make_init(W, H, rng, init_type)
            
            # Run local search
            circles, s = _local_search(init_circles, W, H, deadline, rng)
            
            if s > best_sum:
                best_sum = s
                best_circles = circles.copy()
                
            # Try a few more random restarts if we have time
            for _ in range(2):
                if time.time() > deadline - 5:
                    break
                init_circles = _make_init(W, H, rng, 'random')
                # Perturb the best found so far slightly
                if best_circles is not None:
                    # Map best_circles to this W,H? 
                    # This is tricky because W,H changed.
                    # Better to just use a fresh random init.
                    pass
                circles, s = _local_search(init_circles, W, H, deadline, rng)
                if s > best_sum:
                    best_sum = s
                    best_circles = circles.copy()
                    
    if best_circles is not None:
        return best_circles
        
    # Fallback
    w, h = 1.0, 1.0
    nx, ny = 5, 5
    cells = []
    for i in range(nx):
        for j in range(ny):
            if len(cells) >= N:
                break
            cells.append([(i + 0.5) * w / nx, (j + 0.5) * h / ny, min(w/nx, h/ny) * 0.4])
    return np.array(cells[:N])
