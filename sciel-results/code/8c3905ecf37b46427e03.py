import numpy as np
import time
import random

N = 21
DEADLINE_PAD = 20.0

def construct():
    t0 = time.time()
    deadline = t0 + 120.0 - DEADLINE_PAD

    best_val = 0.0
    best_solution = None

    # Try several aspect ratios
    aspect_ratios = [1.0, 0.85, 1.15, 0.75, 1.25, 0.67, 1.35, 0.6, 1.5, 0.55]

    for w0 in aspect_ratios:
        if time.time() > deadline - 10:
            break
        h0 = 2.0 - w0
        if h0 < 0.4:
            continue

        # Run SA for this aspect ratio
        x, y, r = _simulated_annealing(w0, h0, deadline)
        
        # Final polish: grow radii to max
        x, y, r = _grow_radii(x, y, r, w0, h0)
        
        val = _sum_if_valid(x, y, r, w0, h0)
        if val > best_val:
            best_val = val
            best_solution = (x.copy(), y.copy(), r.copy(), w0, h0)

    if best_solution is None:
        # Fallback
        w0, h0 = 1.0, 1.0
        x, y = _grid_init(w0, h0)
        r = np.full(N, 0.05)
        x = np.clip(x, r, w0 - r)
        y = np.clip(y, r, h0 - r)
        return np.column_stack((x, y, r))

    x, y, r, w0, h0 = best_solution
    # Safety shrink
    for _ in range(10):
        if _is_valid(x, y, r * (1 - 1e-6), w0, h0):
            r = r * (1 - 1e-6)
        else:
            break
            
    return np.column_stack((x, y, r))

def _grid_init(w, h):
    side = int(np.ceil(np.sqrt(N)))
    x = np.empty(N)
    y = np.empty(N)
    for i in range(N):
        row, col = divmod(i, side)
        x[i] = (col + 0.5) / side * w
        y[i] = (row + 0.5) / side * h
    return x, y

def _grow_radii(x, y, r, w, h):
    """Iteratively grow radii to max feasible."""
    x = x.copy()
    y = y.copy()
    r = r.copy()
    for _ in range(50):
        changed = False
        for i in range(N):
            # Max radius determined by walls and other circles
            mr = min(x[i], w - x[i], y[i], h - y[i])
            for j in range(N):
                if i == j: continue
                d = np.hypot(x[i] - x[j], y[i] - y[j])
                mr = min(mr, d - r[j])
            if mr > r[i] + 1e-9:
                r[i] = max(0.0, mr)
                changed = True
        if not changed:
            break
    return x, y, r

def _simulated_annealing(w, h, deadline):
    """SA to find good positions and radii."""
    # Initialize
    x, y = _grid_init(w, h)
    # Add some noise
    x += np.random.uniform(-0.05, 0.05, N)
    y += np.random.uniform(-0.05, 0.05, N)
    x = np.clip(x, 0.05, w - 0.05)
    y = np.clip(y, 0.05, h - 0.05)
    r = np.full(N, 0.05)
    
    # Grow initial radii
    x, y, r = _grow_radii(x, y, r, w, h)
    
    current_cost = _cost(x, y, r, w, h)
    best_cost = current_cost
    best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
    
    T = 0.1
    T_min = 1e-4
    cooling = 0.995
    
    while time.time() < deadline - 5 and T > T_min:
        # Perturb
        new_x = x.copy()
        new_y = y.copy()
        new_r = r.copy()
        
        # Move 1-2 circles
        num_moves = random.randint(1, 2)
        for _ in range(num_moves):
            idx = random.randint(0, N - 1)
            # Move position
            delta_x = random.gauss(0, 0.05)
            delta_y = random.gauss(0, 0.05)
            new_x[idx] += delta_x
            new_y[idx] += delta_y
            
            # Try to increase radius
            if random.random() < 0.5:
                new_r[idx] *= (1.0 + random.uniform(0.01, 0.05))
        
        # Clip positions
        new_x = np.clip(new_x, 0.01, w - 0.01)
        new_y = np.clip(new_y, 0.01, h - 0.01)
        new_r = np.clip(new_r, 0.01, min(w, h) / 2)
        
        # Evaluate
        new_cost = _cost(new_x, new_y, new_r, w, h)
        
        # Acceptance
        if new_cost < current_cost or random.random() < np.exp(-(new_cost - current_cost) / T):
            current_cost = new_cost
            x, y, r = new_x, new_y, new_r
            if current_cost < best_cost:
                best_cost = current_cost
                best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
        
        T *= cooling
        
    return best_x, best_y, best_r

def _cost(x, y, r, w, h):
    """Penalty function: minimize overlaps and wall violations, maximize sum of radii."""
    # Overlaps
    penalty = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            d = np.hypot(x[i] - x[j], y[i] - y[j])
            overlap = r[i] + r[j] - d
            if overlap > 0:
                penalty += overlap * overlap
        
        # Walls
        wall_dist = min(x[i], w - x[i], y[i], h - y[i])
        wall_overlap = r[i] - wall_dist
        if wall_overlap > 0:
            penalty += wall_overlap * wall_overlap * 10.0 # Higher weight for walls
    
    # Reward for large radii
    reward = np.sum(r)
    
    return penalty - 1.0 * reward

def _is_valid(x, y, r, w, h, tol=1e-8):
    if w <= 0.05 or h <= 0.05:
        return False
    if np.any(x - r < -tol) or np.any(y - r < -tol):
        return False
    if np.any(x + r > w + tol) or np.any(y + r > h + tol):
        return False
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            if dx * dx + dy * dy < (r[i] + r[j]) ** 2 - tol:
                return False
    return True

def _sum_if_valid(x, y, r, w, h):
    if _is_valid(x, y, r, w, h):
        return float(np.sum(r))
    return 0.0
