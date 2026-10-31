"""Circle packing in perimeter-4 rectangle.
Approach: Simulated Annealing.
- Uses a stochastic local search to maximize the sum of radii.
- Penalties for constraint violations (overlap, out-of-bounds).
- Multiple restarts with different aspect ratios and random initializations.
"""
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

    # Several aspect ratios for diversity (w + h = 2).
    # We will try a few distinct aspect ratios with multiple random restarts each.
    aspect_ratios = [1.0, 0.85, 1.18, 0.75, 1.33, 0.67, 1.5, 0.6, 1.75, 0.55,
                     1.25, 0.8, 1.4, 0.7, 1.6, 0.63, 1.1, 0.9]
    
    # Shuffle to randomize order if needed, or just iterate
    random.shuffle(aspect_ratios)

    for w0 in aspect_ratios:
        if time.time() > deadline - 20:
            break
        h0 = 2.0 - w0
        if h0 < 0.35:
            continue

        # Run multiple restarts for this aspect ratio
        num_restarts = 5
        for restart in range(num_restarts):
            if time.time() > deadline - 10:
                break
            
            x, y, r = _random_init(w0, h0)
            x, y, r = _simulated_annealing(x, y, r, w0, h0, deadline)
            
            val = _sum_if_valid(x, y, r, w0, h0)
            if val > best_val:
                best_val = val
                best_solution = (x.copy(), y.copy(), r.copy(), w0, h0)

    if best_solution is None:
        # Fallback: trivial valid grid in a unit square.
        w0, h0 = 1.0, 1.0
        x, y = _grid_init(w0, h0)
        r = np.full(N, 0.05)
        x = np.clip(x, r, w0 - r)
        y = np.clip(y, r, h0 - r)
        return np.column_stack((x, y, r))

    x, y, r, w0, h0 = best_solution
    
    # Final safety shrink to guarantee strict validity.
    # We ensure the solution is strictly valid by shrinking slightly if needed.
    # The annealing process should have produced a valid solution, but we double check.
    if not _is_valid(x, y, r, w0, h0):
        # If invalid, try to repair by shrinking
        for _ in range(100):
            if _is_valid(x, y, r * (1 - 1e-5), w0, h0):
                r = r * (1 - 1e-5)
            else:
                break
        # If still invalid, fallback to grid
        if not _is_valid(x, y, r, w0, h0):
             w0, h0 = 1.0, 1.0
             x, y = _grid_init(w0, h0)
             r = np.full(N, 0.05)
             x = np.clip(x, r, w0 - r)
             y = np.clip(y, r, h0 - r)

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


def _random_init(w, h):
    """Initialize with small random circles."""
    x = np.random.uniform(0.1, w - 0.1, N)
    y = np.random.uniform(0.1, h - 0.1, N)
    # Start with small radii to avoid initial overlaps
    r = np.full(N, min(w, h) / 20.0)
    return x, y, r


def _evaluate(x, y, r, w, h, penalty_weight=100.0):
    """Calculate objective function: sum of radii - penalty for violations."""
    score = np.sum(r)
    
    # Boundary violations
    for i in range(N):
        if x[i] - r[i] < 0:
            score -= penalty_weight * (r[i] - x[i])
        if x[i] + r[i] > w:
            score -= penalty_weight * (x[i] + r[i] - w)
        if y[i] - r[i] < 0:
            score -= penalty_weight * (r[i] - y[i])
        if y[i] + r[i] > h:
            score -= penalty_weight * (y[i] + r[i] - h)
            
    # Overlap violations
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = np.hypot(dx, dy)
            min_d = r[i] + r[j]
            if d < min_d:
                overlap = min_d - d
                score -= penalty_weight * overlap * overlap # Quadratic penalty for overlaps
    
    return score


def _simulated_annealing(x, y, r, w, h, deadline):
    """Run simulated annealing to optimize the packing."""
    x = x.copy()
    y = y.copy()
    r = r.copy()
    
    current_score = _evaluate(x, y, r, w, h)
    best_score = current_score
    best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
    
    # Annealing parameters
    T = 0.5 # Initial temperature
    alpha = 0.995 # Cooling rate
    min_T = 1e-5
    
    # Time limit per run is roughly (deadline - t0) / total_runs
    # We will just run until deadline is close or T is too low
    while time.time() < deadline - 5 and T > min_T:
        # Select a random circle to perturb
        i = np.random.randint(N)
        
        # Perturb position
        dx = np.random.normal(0, 0.05)
        dy = np.random.normal(0, 0.05)
        
        new_x = x.copy()
        new_y = y.copy()
        new_r = r.copy()
        
        new_x[i] += dx
        new_y[i] += dy
        
        # Clamp to bounds (softly, or just let penalty handle it? 
        # Clamping helps stability but might get stuck. Let's clamp to [0, w] and [0, h] 
        # but allow radius to cause penalty if it goes out. 
        # Actually, for SA, it's often better to allow moves outside and penalize.
        # But here we clamp position to keep it in the box, and radius determines if it's valid.
        new_x[i] = np.clip(new_x[i], 0, w)
        new_y[i] = np.clip(new_y[i], 0, h)
        
        # Try to adjust radius: 
        # With some probability, try to grow the radius slightly.
        # With some probability, shrink it.
        # Or, simply calculate the max possible radius for this new position 
        # given other circles, and set it to that? 
        # That would be "greedy growth" which is close to the old approach.
        # To stay distinct, let's do random radius perturbation.
        
        if np.random.random() < 0.5:
             # Try to grow
             delta_r = np.random.uniform(0, 0.05)
             new_r[i] += delta_r
        else:
             # Try to shrink (or keep same)
             delta_r = np.random.uniform(-0.05, 0.05)
             new_r[i] = max(0.01, new_r[i] + delta_r)
        
        # Ensure new radius doesn't make it obviously invalid (e.g., larger than box)
        max_possible_r = min(new_x[i], w - new_x[i], new_y[i], h - new_y[i])
        new_r[i] = min(new_r[i], max_possible_r)
        
        new_score = _evaluate(new_x, new_y, new_r, w, h)
        
        delta = new_score - current_score
        
        if delta > 0 or np.random.random() < np.exp(delta / T):
            x, y, r = new_x, new_y, new_r
            current_score = new_score
            
            if current_score > best_score:
                best_score = current_score
                best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
        
        T *= alpha
        
        # Occasionally, if we are stuck (low score), do a larger random jump?
        # Or just rely on the cooling.
        
    return best_x, best_y, best_r


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
