"""Circle packing in perimeter-4 rectangle: Kinetic Jiggle-and-Grow."""
import numpy as np
import time
import random

N = 21

def _is_valid(cx, cy, r, W, H, tol=1e-7):
    """Check if packing is valid."""
    if W <= 0.01 or H <= 0.01:
        return False
    if np.any(cx - r < -tol) or np.any(cy - r < -tol):
        return False
    if np.any(cx + r > W + tol) or np.any(cy + r > H + tol):
        return False
    for i in range(N):
        for j in range(i + 1, N):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            if dx * dx + dy * dy < (r[i] + r[j]) ** 2 - tol:
                return False
    return True

def _hexagonal_init(w, h, n, seed=0):
    """Generate a hexagonal close-packed layout."""
    random.seed(seed)
    np.random.seed(seed)
    
    # Estimate optimal radius
    r_est = np.sqrt((w * h) / (n * 2 * np.sqrt(3)))
    
    # Determine grid dimensions
    cols = int(np.ceil(np.sqrt(n * w / h)))
    rows = int(np.ceil(n / cols))
    
    x = np.zeros(n)
    y = np.zeros(n)
    
    for i in range(n):
        row = i // cols
        col = i % cols
        # Hexagonal offset
        offset = 0.5 * (row % 2)
        x[i] = (col + 0.5 + offset) * (w / (cols + 0.5))
        y[i] = (row + 0.5) * (h / rows)
    
    # Calculate initial radius
    min_dx = w / (cols + 0.5)
    min_dy = h / rows
    r_init = min(min_dx, min_dy) * 0.4
    r_init = min(r_init, w/2, h/2)
    
    r = np.zeros(n)
    for i in range(n):
        r[i] = min(r_init, x[i], w - x[i], y[i], h - y[i])
    
    # Add small random perturbation to break symmetry
    if seed > 0:
        noise = 0.01 * np.random.random(n)
        x = np.clip(x + (noise - 0.5) * 0.02, r, w - r)
        y = np.clip(y + (noise - 0.5) * 0.02, r, h - r)
    
    return x, y, r

def _jiggle(x, y, r, w, h, temp, iterations, rng):
    """
    Force-directed relaxation to resolve overlaps and boundary violations.
    Uses a temperature parameter to control the jitter and step size.
    """
    x = x.copy().astype(float)
    y = y.copy().astype(float)
    
    for _ in range(iterations):
        fx = np.zeros(N)
        fy = np.zeros(N)
        
        # Wall forces (push away from walls if violating)
        for i in range(N):
            if x[i] < r[i]:
                fx[i] += (r[i] - x[i]) * 100
            if x[i] > w - r[i]:
                fx[i] -= (x[i] - (w - r[i])) * 100
            if y[i] < r[i]:
                fy[i] += (r[i] - y[i]) * 100
            if y[i] > h - r[i]:
                fy[i] -= (y[i] - (h - r[i])) * 100
        
        # Repulsion forces between overlapping circles
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d_sq = dx*dx + dy*dy
                d = np.sqrt(d_sq)
                if d < 1e-10:
                    dx, dy = 1e-5, 1e-5
                    d = np.sqrt(dx*dx + dy*dy)
                
                overlap = r[i] + r[j] - d
                if overlap > 0:
                    # Force is proportional to overlap
                    force = overlap * 50
                    fx[i] += force * dx / d
                    fy[i] += force * dy / d
                    fx[j] -= force * dx / d
                    fy[j] -= force * dy / d
        
        # Add random jitter (simulated annealing style)
        if temp > 0:
            fx += rng.uniform(-temp, temp, N)
            fy += rng.uniform(-temp, temp, N)
        
        # Apply forces
        # Step size is fixed relative to forces, but forces are scaled by temp in jitter
        # We use a small dt to ensure stability
        dt = 0.01
        x += fx * dt
        y += fy * dt
        
        # Clamp to boundaries (hard constraint)
        x = np.clip(x, r, w - r)
        y = np.clip(y, r, h - r)
        
    return x, y

def _greedy_grow(x, y, r, w, h):
    """
    Iteratively expand radii until no more improvement possible.
    This is a local search that maximizes sum(r) for fixed x, y.
    """
    r = r.copy().astype(float)
    
    # We do a few passes to ensure convergence
    for _ in range(5):
        improved = False
        for i in range(N):
            # Max radius limited by walls
            r_max = min(x[i], w - x[i], y[i], h - y[i])
            
            # Max radius limited by neighbors
            for j in range(N):
                if i == j:
                    continue
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx*dx + dy*dy)
                r_allowed = d - r[j]
                if r_allowed < r_max:
                    r_max = r_allowed
            
            # Update radius
            if r_max > r[i] + 1e-10:
                r[i] = max(r_max, 1e-10)
                improved = True
        
        if not improved:
            break
            
    return r

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    best_value = 0.0
    best_result = None
    
    def update_best(x, y, r, w, h):
        nonlocal best_value, best_result
        if _is_valid(x, y, r, w, h):
            val = float(np.sum(r))
            if val > best_value:
                best_value = val
                best_result = (x.copy(), y.copy(), r.copy(), w, h)
    
    # Search over aspect ratios: w + h = 2, so w in [0.3, 1.7]
    # We prioritize near-square and common ratios
    aspect_ratios = [1.0, 0.8, 1.25, 0.7, 1.4, 0.6, 1.6, 0.55, 1.45, 0.9, 1.1, 1.3, 0.75, 1.2, 0.85, 1.15]
    
    # For each aspect ratio, run a "jiggle-and-grow" loop
    # We allocate time per aspect ratio based on remaining time
    total_aspects = len(aspect_ratios)
    
    for idx, w in enumerate(aspect_ratios):
        if time.time() > deadline - 10:
            break
        h = 2.0 - w
        if h < 0.25:
            continue
        
        # Budget for this aspect ratio
        # We want to spend more time on early ones if they are promising, 
        # but for now we distribute roughly evenly with a decay
        remaining_aspects = total_aspects - idx
        time_per_aspect = (deadline - time.time() - 5) / remaining_aspects
        
        # Run multiple seeds for this aspect ratio
        max_seeds = 5
        seed_time_budget = time_per_aspect / max_seeds
        
        for seed in range(max_seeds):
            if time.time() > deadline - 5:
                break
            
            # Initialize
            x, y, r = _hexagonal_init(w, h, N, seed=seed * 100 + idx)
            
            # Initial greedy grow to get a decent starting radius
            r = _greedy_grow(x, y, r, w, h)
            
            # Jiggle and Grow loop
            # We use a simulated annealing style temperature decay
            temp = 0.05
            decay = 0.95
            min_temp = 0.001
            
            # We run this loop until time budget for this seed is exhausted
            seed_start = time.time()
            
            while time.time() - seed_start < seed_time_budget:
                # Jiggle (relax positions)
                # Number of iterations depends on temp? Or fixed?
                # Fixed 50 iterations seems reasonable for speed
                x, y = _jiggle(x, y, r, w, h, temp, 50, np.random.default_rng(seed + int(time.time()*1000)))
                
                # Grow (maximize radii)
                r = _greedy_grow(x, y, r, w, h)
                
                # Update best
                update_best(x, y, r, w, h)
                
                # Decay temperature
                temp *= decay
                if temp < min_temp:
                    break
                
                # If no overlap, we might want to try to grow more aggressively?
                # The greedy grow handles that.
                
                # Check if we are stuck? If sum(r) doesn't change, we might break early?
                # But jiggle might help escape local minima even if grow is stuck.
                # So we continue until time/temp limit.
            
    # Fallback
    if best_result is None:
        w, h = 1.0, 1.0
        x, y, r = _hexagonal_init(w, h, N, seed=0)
        r = _greedy_grow(x, y, r, w, h)
        best_result = (x, y, r, w, h)
    
    x, y, r, w, h = best_result
    
    # Final safety shrink
    for _ in range(3):
        r_trial = r * (1 - 1e-9)
        if _is_valid(x, y, r_trial, w, h):
            r = r_trial
        else:
            break
    
    return np.column_stack((x, y, r))
