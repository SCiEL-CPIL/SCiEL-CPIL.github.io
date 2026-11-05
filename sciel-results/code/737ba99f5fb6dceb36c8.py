import random
import time
import numpy as np
from collections import Counter

def _precompute_triples(n=30):
    """Precompute list of all triples (i, j, k) with i < j < k."""
    triples = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                triples.append((i, j, k))
    return triples

# Precompute once at module level for speed
_TRIPLES = _precompute_triples(30)
_N = 30

def _compute_initial_counts(points):
    """
    Compute the initial area counts for the given points.
    Returns a Counter of absolute doubled areas (excluding 0).
    """
    pts = np.array(points, dtype=np.int64)
    xs = pts[:, 0]
    ys = pts[:, 1]
    
    # Extract coordinates for all triples
    i_idx = np.array([t[0] for t in _TRIPLES])
    j_idx = np.array([t[1] for t in _TRIPLES])
    k_idx = np.array([t[2] for t in _TRIPLES])
    
    x1 = xs[i_idx]
    y1 = ys[i_idx]
    x2 = xs[j_idx]
    y2 = ys[j_idx]
    x3 = xs[k_idx]
    y3 = ys[k_idx]
    
    # Calculate doubled area: |x1(y2 - y3) - x2(y1 - y3) + x3(y1 - y2)|
    # Or using determinant form: | (x2-x1)(y3-y1) - (y2-y1)(x3-x1) |
    # The latter is slightly more stable for small integers? Let's use the standard cross product magnitude.
    # det = (x2 - x1)*(y3 - y1) - (y2 - y1)*(x3 - x1)
    dets = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
    abs_dets = np.abs(dets)
    
    # Filter out zero areas
    valid_dets = abs_dets[abs_dets > 0]
    
    if len(valid_dets) == 0:
        return Counter()
        
    # Convert to list for Counter (numpy array is also iterable, but list is explicit)
    # Counter can take an iterable of numpy ints directly
    return Counter(valid_dets.tolist())

def _get_affected_triples(idx):
    """
    Return list of indices in _TRIPLES that involve point idx.
    Also return the position of idx in that triple (0, 1, or 2) to help with calculation?
    Actually, it's easier to just iterate and check.
    To make it faster, we can precompute for each point, the list of triple indices.
    """
    # This is called frequently, so we should cache or precompute.
    pass

# Precompute affected triple indices for each point
_AFFECTED_TRIPLES = [[] for _ in range(_N)]
for t_idx, (i, j, k) in enumerate(_TRIPLES):
    _AFFECTED_TRIPLES[i].append(t_idx)
    _AFFECTED_TRIPLES[j].append(t_idx)
    _AFFECTED_TRIPLES[k].append(t_idx)

def _update_counts(points, idx, old_pos, new_pos, counts):
    """
    Update the counts dictionary when point idx moves from old_pos to new_pos.
    Returns the new max count.
    """
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    
    old_x, old_y = old_pos
    new_x, new_y = new_pos
    
    affected_t_idxs = _AFFECTED_TRIPLES[idx]
    
    # We need to subtract old areas and add new areas for these triples.
    # To avoid modifying the counts dict while iterating if we were doing it in one pass,
    # we can collect the changes or just do two passes.
    # Two passes is safe and simple.
    
    # 1. Subtract old areas
    for t_idx in affected_t_idxs:
        i, j, k = _TRIPLES[t_idx]
        
        # Get coordinates, replacing idx with old_pos
        if i == idx:
            xi, yi = old_x, old_y
        else:
            xi, yi = xs[i], ys[i]
            
        if j == idx:
            xj, yj = old_x, old_y
        else:
            xj, yj = xs[j], ys[j]
            
        if k == idx:
            xk, yk = old_x, old_y
        else:
            xk, yk = xs[k], ys[k]
            
        # Calculate old area
        # det = (xj - xi)*(yk - yi) - (yj - yi)*(xk - xi)
        det = (xj - xi) * (yk - yi) - (yj - yi) * (xk - xi)
        area = abs(det)
        
        if area > 0:
            counts[area] -= 1
            if counts[area] == 0:
                del counts[area]
                
    # 2. Add new areas
    for t_idx in affected_t_idxs:
        i, j, k = _TRIPLES[t_idx]
        
        # Get coordinates, replacing idx with new_pos
        if i == idx:
            xi, yi = new_x, new_y
        else:
            xi, yi = xs[i], ys[i]
            
        if j == idx:
            xj, yj = new_x, new_y
        else:
            xj, yj = xs[j], ys[j]
            
        if k == idx:
            xk, yk = new_x, new_y
        else:
            xk, yk = xs[k], ys[k]
            
        # Calculate new area
        det = (xj - xi) * (yk - yi) - (yj - yi) * (xk - xi)
        area = abs(det)
        
        if area > 0:
            if area in counts:
                counts[area] += 1
            else:
                counts[area] = 1
            # We don't need to track max here, we can compute it at the end or track it.
            # Computing max(counts.values()) is O(U) where U is number of unique areas.
            # U is likely small (< 1000). So it's fast.
            
    if not counts:
        return 0
        
    return max(counts.values())

def _score(points):
    """
    Compute the largest multiplicity of a positive doubled area.
    Uses full recomputation for initial setup or verification.
    """
    counts = _compute_initial_counts(points)
    if not counts:
        return 0
    return max(counts.values())

def construct():
    # Deadline: 55 seconds to be safe against 60s kill
    start_time = time.time()
    deadline = start_time + 55.0
    
    # Initial Configuration: A grid is a good start for high multiplicities
    # 6x5 grid covers 30 points.
    # Let's try a few different initial grids to see which is better.
    
    def _generate_grid(cols, rows, offset_x=0, offset_y=0):
        pts = []
        for r in range(rows):
            for c in range(cols):
                pts.append((c + offset_x, r + offset_y))
        return pts
    
    # Candidate initial configurations
    candidates = []
    
    # Standard 6x5 grid
    candidates.append(_generate_grid(6, 5))
    
    # 5x6 grid
    candidates.append(_generate_grid(5, 6))
    
    # 10x3 grid
    candidates.append(_generate_grid(10, 3))
    
    # 15x2 grid
    candidates.append(_generate_grid(15, 2))
    
    # Randomized grid (jittered)
    rng = random.Random(42)
    base_grid = _generate_grid(6, 5)
    jittered_grid = []
    for p in base_grid:
        # Small jitter
        dx = rng.randint(-1, 1)
        dy = rng.randint(-1, 1)
        np_ = (p[0] + dx, p[1] + dy)
        # Ensure distinctness and range
        if np_ in jittered_grid:
             np_ = p
        jittered_grid.append(np_)
    # Fix collisions if any
    seen = set()
    final_jittered = []
    for p in jittered_grid:
        if p in seen:
             # find a free spot nearby
             for dx in range(-2, 3):
                 for dy in range(-2, 3):
                     np_ = (p[0]+dx, p[1]+dy)
                     if -30 <= np_[0] <= 30 and -30 <= np_[1] <= 30 and np_ not in seen:
                         p = np_
                         break
                 else:
                     continue
                 break
        seen.add(p)
        final_jittered.append(p)
    candidates.append(final_jittered)
    
    # Evaluate candidates
    best_score = -1
    best_points = None
    best_counts = None
    
    for cand in candidates:
        # Check validity
        if len(set(cand)) != 30:
            continue
        if not all(-30 <= p[0] <= 30 and -30 <= p[1] <= 30 for p in cand):
            continue
            
        s = _score(cand)
        if s > best_score:
            best_score = s
            best_points = cand[:]
            best_counts = _compute_initial_counts(cand)
            
    if best_points is None:
        # Fallback to random
        best_points = _generate_grid(6, 5)
        best_score = _score(best_points)
        best_counts = _compute_initial_counts(best_points)

    current_points = best_points[:]
    current_counts = best_counts.copy()
    current_score = best_score
    
    # Local Search
    # Strategy: Simulated Annealing
    # Move one point to a random valid location.
    # Accept if better.
    # Accept worse with probability exp(-delta/T).
    
    temp = 10.0
    cooling = 0.9995
    min_temp = 0.1
    
    # Precompute valid moves? No, just generate random points.
    # To speed up, we can restrict moves to a neighborhood or random grid points.
    # Random grid points in [-30, 30] are 61x61 = 3721 points.
    # We can precompute a list of all valid points.
    all_valid_points = [(x, y) for x in range(-30, 31) for y in range(-30, 31)]
    
    steps = 0
    while time.time() < deadline:
        steps += 1
        
        # Pick a random point to move
        idx = rng.randint(0, _N - 1)
        old_pos = current_points[idx]
        
        # Pick a new position
        # Bias towards nearby points? Or purely random?
        # Purely random explores more, but might be too disruptive.
        # Let's try a mix: 50% nearby, 50% random.
        if rng.random() < 0.5:
            # Nearby
            ox, oy = old_pos
            dx = rng.randint(-3, 3)
            dy = rng.randint(-3, 3)
            new_pos = (ox + dx, oy + dy)
            if not (-30 <= new_pos[0] <= 30 and -30 <= new_pos[1] <= 30):
                new_pos = old_pos
        else:
            new_pos = rng.choice(all_valid_points)
            
        if new_pos == old_pos:
            continue
            
        # Check if new_pos is occupied by another point
        # We need to ensure distinctness.
        # If new_pos is occupied by point j, we can either skip or swap?
        # Swapping is more complex. Let's just skip if occupied.
        # To check if occupied, we can keep a set of current points.
        # But maintaining a set is O(1) update.
        
        # Optimization: Keep a set of points for fast lookup
        # We'll just check if new_pos is in current_points (O(n)) which is 30. Fast enough.
        if new_pos in current_points:
            continue
            
        # Update counts
        # We need to temporarily update current_points to get the new coords for other points?
        # No, _update_counts takes points, idx, old_pos, new_pos.
        # It uses points for the *other* coordinates.
        # So we must ensure `points` reflects the state BEFORE the move for the other points.
        # Which it does.
        
        new_score = _update_counts(current_points, idx, old_pos, new_pos, current_counts)
        
        # Update current state
        current_points[idx] = new_pos
        
        delta = new_score - current_score
        
        if delta >= 0:
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_points = current_points[:]
                best_counts = current_counts.copy()
        else:
            # Worse move
            # Accept with probability
            if temp > min_temp:
                prob = np.exp(delta / temp)
                if rng.random() < prob:
                    current_score = new_score
                else:
                    # Revert
                    # We need to revert counts and points
                    # Revert points
                    current_points[idx] = old_pos
                    # Revert counts: swap old and new
                    _update_counts(current_points, idx, new_pos, old_pos, current_counts)
            else:
                # Revert
                current_points[idx] = old_pos
                _update_counts(current_points, idx, new_pos, old_pos, current_counts)
                
        # Cool down
        temp *= cooling
        
        # Restart if stuck?
        # If no improvement for a long time, maybe restart from best?
        # For now, just let annealing handle it.
        
        # Check time frequently
        if steps % 100 == 0:
            if time.time() > deadline - 5:
                break
                
    return best_points

# Ensure the function is named exactly as required
# The task asks for construct() -> ...

if __name__ == "__main__":
    # Quick test
    import time
    start = time.time()
    pts = construct()
    end = time.time()
    score = _score(pts)
    print(f"Score: {score}, Time: {end - start:.2f}s")
    print(f"Points: {pts[:5]}...")
