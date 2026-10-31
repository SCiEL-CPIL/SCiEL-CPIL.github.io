"""Circle packing in perimeter-4 rectangle: force-directed + uniform scale binary search."""
import numpy as np
import time
import random

N = 21
DEADLINE = 115.0  # safety margin before 150s limit

def _aspect_to_dims(ar):
    w = 2.0 / (1.0 + ar)
    h = 2.0 - w
    return w, h

def _hex_grid_init(w, h, n, seed):
    """Generate hexagonal grid positions for n circles in rectangle w x h."""
    rng = np.random.default_rng(seed)
    # Estimate rows/cols for hex packing
    area_per = w * h / n
    # Hex packing density ~ 0.9069
    r_est = np.sqrt(area_per / (np.pi * 1.15))  # slightly conservative
    # Number of rows
    n_rows = max(1, int(np.round(h / (1.866 * r_est))))
    n_cols = max(1, int(np.ceil(n / n_rows)))
    
    while n_rows * n_cols < n:
        if n_rows <= n_cols:
            n_rows += 1
        else:
            n_cols += 1
    while n_rows * n_cols >= n + n_rows:
        # Try to reduce
        if n_cols > 1:
            n_cols -= 1
        elif n_rows > 1:
            n_rows -= 1
        else:
            break
    
    xs = np.zeros(n)
    ys = np.zeros(n)
    count = 0
    row_spacing = h / n_rows
    col_spacing = w / n_cols
    hex_offset = 0.5 * col_spacing
    
    for row in range(n_rows):
        for col in range(n_cols):
            if count >= n:
                break
            x = (col + 0.5) * col_spacing
            y = (row + 0.5) * row_spacing
            if row % 2 == 1:
                x += hex_offset
                # Wrap around if needed
                if x > w - r_est:
                    x -= col_spacing
                elif x < r_est:
                    x += col_spacing
            xs[count] = x
            ys[count] = y
            count += 1
    
    # Jitter
    jitter = 0.15 * r_est
    xs += rng.normal(0, jitter, n)
    ys += rng.normal(0, jitter, n)
    # Clip to bounds
    xs = np.clip(xs, r_est * 0.5, w - r_est * 0.5)
    ys = np.clip(ys, r_est * 0.5, h - r_est * 0.5)
    return xs, ys

def _random_init(w, h, n, seed):
    """Random initialization with minimum spacing."""
    rng = np.random.default_rng(seed)
    xs = np.zeros(n)
    ys = np.zeros(n)
    # Place circles one by one with rejection sampling
    min_r = 0.02
    for i in range(n):
        for attempt in range(200):
            x = rng.uniform(min_r, w - min_r)
            y = rng.uniform(min_r, h - min_r)
            # Check distance to existing
            if i > 0:
                dx = xs[:i] - x
                dy = ys[:i] - y
                dists = np.sqrt(dx**2 + dy**2)
                if np.min(dists) < 2 * min_r:
                    continue
            xs[i] = x
            ys[i] = y
            break
        else:
            xs[i] = (i % 5 + 0.5) * w / 5
            ys[i] = (i // 5 + 0.5) * h / 5
    return xs, ys

def _force_relax(xs, ys, r, w, h, max_iters=2000, dt=0.005, repulse_k=100.0):
    """
    Force-directed relaxation: circles repel each other and are attracted to stay in bounds.
    Returns relaxed positions and the achieved uniform radius.
    """
    n = len(xs)
    xs = xs.astype(float).copy()
    ys = ys.astype(float).copy()
    r = np.full(n, r)
    
    best_sum_r = 0.0
    best_xs, best_ys, best_r = xs.copy(), ys.copy(), r.copy()
    
    for it in range(max_iters):
        # Compute forces
        fx = np.zeros(n)
        fy = np.zeros(n)
        
        # Pairwise repulsion
        for i in range(n):
            for j in range(i + 1, n):
                dx = xs[j] - xs[i]
                dy = ys[j] - ys[i]
                dist = np.sqrt(dx * dx + dy * dy)
                if dist < 1e-9:
                    dist = 1e-9
                    dx = 0.01 * random.random()
                    dy = 0.01 * random.random()
                min_dist = r[i] + r[j]
                if dist < min_dist * 1.5:  # Only interact when close
                    # Repulsive force: stronger when overlapping
                    overlap = min_dist - dist
                    if overlap > 0:
                        force = repulse_k * overlap
                    else:
                        force = repulse_k * 0.1 * (min_dist - dist) / (min_dist * 0.5)
                    if force > 0:
                        inv_dist = 1.0 / dist
                        fx[i] -= force * dx * inv_dist
                        fy[i] -= force * dy * inv_dist
                        fx[j] += force * dx * inv_dist
                        fy[j] += force * dy * inv_dist
        
        # Boundary forces (soft walls)
        wall_k = 500.0
        for i in range(n):
            if xs[i] < r[i]:
                fx[i] += wall_k * (r[i] - xs[i])
            if xs[i] > w - r[i]:
                fx[i] -= wall_k * (xs[i] - (w - r[i]))
            if ys[i] < r[i]:
                fy[i] += wall_k * (r[i] - ys[i])
            if ys[i] > h - r[i]:
                fy[i] -= wall_k * (ys[i] - (h - r[i]))
        
        # Update positions with damping
        damping = 0.95
        for i in range(n):
            xs[i] += dt * fx[i] * damping
            ys[i] += dt * fy[i] * damping
            # Hard clip to bounds
            xs[i] = np.clip(xs[i], r[i], w - r[i])
            ys[i] = np.clip(ys[i], r[i], h - r[i])
        
        # Check feasibility and track best
        feasible = True
        for i in range(n):
            for j in range(i + 1, n):
                dx = xs[j] - xs[i]
                dy = ys[j] - ys[i]
                if np.sqrt(dx * dx + dy * dy) < r[i] + r[j] - 1e-9:
                    feasible = False
                    break
            if not feasible:
                break
        
        if feasible:
            s = np.sum(r)
            if s > best_sum_r:
                best_sum_r = s
                best_xs = xs.copy()
                best_ys = ys.copy()
                best_r = r.copy()
        
        # Adaptive dt: reduce if forces are large
        max_force = np.max(np.sqrt(fx**2 + fy**2))
        if max_force > 10:
            dt *= 0.5
        elif max_force < 1:
            dt = min(dt * 1.1, 0.01)
        
        # Convergence check
        if max_force < 0.01:
            break
    
    return best_xs, best_ys, best_r, best_sum_r

def _try_radius_scale(xs, ys, w, h, r_base, max_iters=500):
    """
    Given positions, try to find the maximum uniform radius scale factor
    such that all circles fit without overlap.
    Uses iterative expansion with force relaxation.
    """
    n = len(xs)
    r = np.full(n, r_base)
    xs = xs.copy()
    ys = ys.copy()
    
    best_sum = 0.0
    best_xs, best_ys, best_r = xs.copy(), ys.copy(), r.copy()
    
    for it in range(max_iters):
        # Check current feasibility
        feasible = True
        for i in range(n):
            for j in range(i + 1, n):
                dx = xs[j] - xs[i]
                dy = ys[j] - ys[i]
                if np.sqrt(dx * dx + dy * dy) < r[i] + r[j] - 1e-9:
                    feasible = False
                    break
            if not feasible:
                break
        for i in range(n):
            if xs[i] < r[i] - 1e-9 or xs[i] > w - r[i] + 1e-9 or \
               ys[i] < r[i] - 1e-9 or ys[i] > h - r[i] + 1e-9:
                feasible = False
                break
        
        if feasible:
            s = np.sum(r)
            if s > best_sum:
                best_sum = s
                best_xs = xs.copy()
                best_ys = ys.copy()
                best_r = r.copy()
        
        # Try expanding radii
        new_r = r * 1.002
        # Check if expansion is feasible with current positions
        ok = True
        for i in range(n):
            for j in range(i + 1, n):
                dx = xs[j] - xs[i]
                dy = ys[j] - ys[i]
                if np.sqrt(dx * dx + dy * dy) < new_r[i] + new_r[j] - 1e-9:
                    ok = False
                    break
            if not ok:
                break
        for i in range(n):
            if xs[i] < new_r[i] - 1e-9 or xs[i] > w - new_r[i] + 1e-9 or \
               ys[i] < new_r[i] - 1e-9 or ys[i] > h - new_r[i] + 1e-9:
                ok = False
                break
        
        if ok:
            r = new_r
        else:
            # Need to relax positions to accommodate larger radii
            xs, ys, r, _ = _force_relax(xs, ys, new_r, w, h, max_iters=200, dt=0.003, repulse_k=100.0)
            # Check if we made progress
            if np.sum(r) <= np.sum(best_r) * 0.99:
                # No progress, try smaller expansion
                r = best_r.copy()
                xs = best_xs.copy()
                ys = best_ys.copy()
                break
    
    return best_xs, best_ys, best_r, best_sum

def _full_relax_and_scale(xs, ys, w, h, seed):
    """
    Full pipeline: initialize, relax, then scale up radii.
    """
    n = len(xs)
    # Initial radius estimate
    area = w * h
    r_est = np.sqrt(0.85 * area / (n * np.pi))  # 85% of ideal
    r_init = r_est * 0.7  # Start smaller
    
    # Phase 1: Relax with initial radii
    xs1, ys1, r1, _ = _force_relax(xs, ys, np.full(n, r_init), w, h, max_iters=500, dt=0.005, repulse_k=100.0)
    
    # Phase 2: Scale up
    xs2, ys2, r2, best_sum = _try_radius_scale(xs1, ys1, w, h, r_init, max_iters=300)
    
    return xs2, ys2, r2, best_sum

def construct():
    deadline = time.time() + DEADLINE
    best_score = 0.0
    best_result = None
    
    # Aspect ratios to try
    aspect_ratios = [1.0, 1.1, 0.9, 1.2, 0.85, 1.3, 0.75, 1.15, 0.95, 1.25, 0.8, 1.4, 0.7, 1.5, 0.65]
    
    # Sort by proximity to 1.0
    aspect_ratios.sort(key=lambda ar: abs(ar - 1.0))
    
    # Multiple restarts per aspect ratio
    n_restarts = 3
    
    for ar in aspect_ratios:
        if time.time() > deadline - 5:
            break
        
        w, h = _aspect_to_dims(ar)
        
        for restart in range(n_restarts):
            if time.time() > deadline - 3:
                break
            
            seed = hash((ar, restart)) % (2**31)
            
            # Choose initialization method
            if restart == 0:
                xs, ys = _hex_grid_init(w, h, N, seed)
            elif restart == 1:
                xs, ys = _random_init(w, h, N, seed)
            else:
                # Perturbed hex
                xs, ys = _hex_grid_init(w, h, N, seed + 999)
                xs += np.random.default_rng(seed + 1234).normal(0, 0.05, N)
                ys += np.random.default_rng(seed + 5678).normal(0, 0.05, N)
                xs = np.clip(xs, 0.02, w - 0.02)
                ys = np.clip(ys, 0.02, h - 0.02)
            
            # Run full relaxation and scaling
            xs2, ys2, r2, score = _full_relax_and_scale(xs, ys, w, h, seed)
            
            if score > best_score:
                best_score = score
                best_result = np.column_stack([xs2, ys2, r2])
    
    # Final validation and fallback
    if best_result is not None:
        x, y, r = best_result[:, 0], best_result[:, 1], best_result[:, 2]
        # Determine which aspect ratio was used (re-derive from best)
        # We need to validate against the correct w, h
        # Since we don't track which ar was best, validate all possible
        for ar in aspect_ratios:
            w, h = _aspect_to_dims(ar)
            valid = True
            if np.any(r < -1e-9):
                valid = False
            elif np.any(x < r - 1e-9) or np.any(x > w - r + 1e-9):
                valid = False
            elif np.any(y < r - 1e-9) or np.any(y > h - r + 1e-9):
                valid = False
            else:
                for i in range(N):
                    for j in range(i + 1, N):
                        dx = x[j] - x[i]
                        dy = y[j] - y[i]
                        if np.sqrt(dx * dx + dy * dy) < r[i] + r[j] - 1e-9:
                            valid = False
                            break
                    if not valid:
                        break
            if valid:
                return best_result
    
    # Fallback: simple grid
    side = int(np.ceil(np.sqrt(N)))
    cells = []
    for i in range(side):
        for j in range(side):
            if len(cells) < N:
                cells.append(((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 0.005))
    return np.array(cells)
