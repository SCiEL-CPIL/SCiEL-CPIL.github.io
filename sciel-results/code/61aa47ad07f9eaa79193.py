# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from math import gcd

def _normalize_line(A, B, C):
    """Normalize line coefficients (A, B, C) to a canonical form."""
    if A == 0 and B == 0:
        return (0, 0, 0)
    
    g = gcd(abs(A), abs(B))
    g = gcd(g, abs(C))
    
    if g > 1:
        A //= g
        B //= g
        C //= g
        
    if A < 0:
        A = -A
        B = -B
        C = -C
    elif A == 0 and B < 0:
        B = -B
        C = -C
        
    return (A, B, C)

def _count_incidences(points, lines):
    """
    Count incidences between points and lines.
    points: (N, 2) array
    lines: (M, 3) array
    Returns total incidences.
    """
    px = points[:, 0:1] # (N, 1)
    py = points[:, 1:2] # (N, 1)
    lA = lines[:, 0:1].T # (1, M)
    lB = lines[:, 1:2].T # (1, M)
    lC = lines[:, 2:3].T # (1, M)
    
    vals = px * lA + py * lB + lC
    incs = np.sum(vals == 0)
    return incs

def _evaluate_configuration(center_coords, slopes, intercepts, max_coord=200):
    """
    Evaluate a configuration defined by 9 centers, 27 slopes, and 27 intercepts.
    Returns (score, points_array, lines_array) or (0, None, None) if invalid.
    
    center_coords: (9, 2) array of (x, y) for concurrency centers.
    slopes: (27, 2) array of (dx, dy) for direction vectors of lines.
    intercepts: (27,) array of integer offsets.
    """
    centers = center_coords
    n_centers = 9
    lines = []
    
    # Construct 27 lines
    for i in range(27):
        c_idx = i // 3
        cx, cy = centers[c_idx]
        dx, dy = slopes[i]
        c_val = intercepts[i]
        
        # Line: dy*x - dx*y + c_val = 0
        A = dy
        B = -dx
        C = c_val
        
        # Check if line passes through center (cx, cy)
        if A * cx + B * cy + C != 0:
            return 0, None, None
            
        lines.append(_normalize_line(A, B, C))
        
    # Check for duplicate lines
    if len(set(lines)) != 27:
        return 0, None, None
        
    # Compute all 54 intersection points (pairs of centers)
    points = []
    seen_points = set()
    
    for i in range(n_centers):
        for j in range(i + 1, n_centers):
            c1 = centers[i]
            c2 = centers[j]
            x1, y1 = c1
            x2, y2 = c2
            
            # Lines through c1: i*3, i*3+1, i*3+2
            # Lines through c2: j*3, j*3+1, j*3+2
            
            # We need to pick one line from group i and one from group j
            # To maximize integer points, we should try all 9 combinations?
            # No, the problem implies a fixed structure.
            # Actually, the "strategy" says place points at pairwise intersections.
            # But which lines?
            # In a pencil, all lines pass through the center.
            # The intersection of a line from pencil i and pencil j is a point.
            # There are 3*3=9 intersection points for each pair of pencils.
            # But we only need 6 points per pair to total 54 (9*6=54).
            # Wait, 9 pairs * 6 points = 54.
            # So for each pair (i,j), we pick 6 of the 9 intersection points.
            # Which 6? The ones that are integers and distinct.
            
            # Let's compute all 9 intersections for pair (i,j)
            pair_points = []
            for k in range(3):
                for l in range(3):
                    l1_idx = i * 3 + k
                    l2_idx = j * 3 + l
                    
                    A1, B1, C1 = lines[l1_idx]
                    A2, B2, C2 = lines[l2_idx]
                    
                    det = A1 * B2 - A2 * B1
                    if det == 0:
                        continue # Parallel (should not happen if slopes distinct)
                        
                    num_x = B1 * C2 - B2 * C1
                    num_y = A2 * C1 - A1 * C2
                    
                    if num_x % det == 0 and num_y % det == 0:
                        x = num_x // det
                        y = num_y // det
                        if -max_coord <= x <= max_coord and -max_coord <= y <= max_coord:
                            pair_points.append((x, y))
                            
            # Select up to 6 distinct points from this pair
            # To maximize incidences, we should pick points that lie on more lines?
            # But these points are defined by the intersection of specific line pairs.
            # Each such point lies on exactly 2 lines (one from i, one from j)
            # UNLESS it coincides with a center or another intersection.
            
            # For now, just pick 6 distinct valid points
            unique_pair_points = list(set(pair_points))
            if len(unique_pair_points) < 6:
                # Not enough integer points for this pair
                # This configuration might be poor
                pass
                
            # Randomly select 6 if more, else all
            # Actually, to be deterministic and good, we should pick the ones
            # that might lie on other lines?
            # Let's just pick the first 6 unique ones.
            selected = unique_pair_points[:6]
            
            for p in selected:
                if p not in seen_points:
                    points.append(p)
                    seen_points.add(p)
                else:
                    # If duplicate, we need a new point?
                    # If we can't get 54 unique points, score is lower.
                    # For now, just add what we have.
                    pass
                    
    if len(points) < 54:
        # Pad with random points? Or return low score?
        # If we don't have 54 points, we can't form the required array.
        # Let's return 0 if we can't form 54 unique integer points.
        # This enforces the constraint that the configuration must yield 54 distinct integer intersections.
        return 0, None, None
        
    points_arr = np.array(points[:54], dtype=np.int64)
    lines_arr = np.array(lines, dtype=np.int64)
    
    # Count incidences
    score = _count_incidences(points_arr, lines_arr)
    
    return score, points_arr, lines_arr

def szemeredi_trotter_incidences():
    """
    Szemeredi-Trotter incidence maximization.
    Uses a structural pencil-based construction with Simulated Annealing.
    """
    start_time = time.time()
    timeout = 590  # Safety margin
    
    # Parameters
    n_centers = 9
    n_lines = 27
    n_points = 54
    
    # Initial state
    # Centers: try to place them on a grid to ensure integer intersections
    # Slopes: use primitive vectors
    # Intercepts: 0 (to force concurrency at center)
    
    best_score = 0
    best_points = None
    best_lines = None
    
    # Define a set of good slopes (dx, dy)
    # We need 27 distinct slopes.
    # Small integers are better for integer intersections.
    slope_pool = []
    for dx in range(-5, 6):
        for dy in range(-5, 6):
            if dx == 0 and dy == 0:
                continue
            if gcd(abs(dx), abs(dy)) == 1:
                # Normalize sign: dx > 0 or (dx==0 and dy>0)
                if dx < 0 or (dx == 0 and dy < 0):
                    dx = -dx
                    dy = -dy
                slope_pool.append((dx, dy))
                
    # Shuffle and pick 27 unique slopes
    random.shuffle(slope_pool)
    # Ensure we have at least 27
    while len(slope_pool) < 27:
        # Add larger slopes if needed
        for dx in range(6, 10):
            for dy in range(-10, 11):
                if dx == 0 and dy == 0: continue
                if gcd(abs(dx), abs(dy)) == 1:
                    if dx < 0 or (dx == 0 and dy < 0):
                        dx = -dx
                        dy = -dy
                    slope_pool.append((dx, dy))
        random.shuffle(slope_pool)
        
    initial_slopes = slope_pool[:27]
    
    # Initial centers: Grid 3x3
    initial_centers = np.array([
        [0, 0], [1, 0], [2, 0],
        [0, 1], [1, 1], [2, 1],
        [0, 2], [1, 2], [2, 2]
    ], dtype=np.int64)
    
    # Initial intercepts: 0
    initial_intercepts = np.zeros(27, dtype=np.int64)
    
    # Evaluate initial
    score, pts, lns = _evaluate_configuration(initial_centers, np.array(initial_slopes), initial_intercepts)
    if score > best_score:
        best_score = score
        best_points = pts
        best_lines = lns
        current_centers = initial_centers.copy()
        current_slopes = np.array(initial_slopes)
        current_intercepts = initial_intercepts.copy()
    else:
        # Fallback: try random centers
        current_centers = np.random.randint(-10, 10, (9, 2)).astype(np.int64)
        current_slopes = np.array(initial_slopes)
        current_intercepts = np.zeros(27, dtype=np.int64)
        score, pts, lns = _evaluate_configuration(current_centers, current_slopes, current_intercepts)
        if score > best_score:
            best_score = score
            best_points = pts
            best_lines = lns
        else:
            # If still 0, we need a valid fallback.
            # Let's construct a simple grid solution manually.
            # 27 lines: 9 vertical, 9 horizontal, 9 diagonal?
            # This might not yield 54 integer points easily.
            # Let's use the "pencil" logic but with fixed good parameters.
            # Actually, if SA fails to find a valid config, we must return a valid one.
            # Let's create a fallback generator.
            best_points, best_lines = _fallback_solution()
            best_score = _count_incidences(best_points, best_lines)
            # Initialize SA state with something close to fallback?
            # Or just restart SA.
            current_centers = initial_centers.copy()
            current_slopes = np.array(initial_slopes)
            current_intercepts = initial_intercepts.copy()
            # Force a valid state for SA to start from?
            # If initial is invalid, SA is stuck.
            # Let's try to find ANY valid configuration quickly.
            for _ in range(100):
                current_centers = np.random.randint(-5, 5, (9, 2)).astype(np.int64)
                current_intercepts = np.zeros(27, dtype=np.int64)
                score, pts, lns = _evaluate_configuration(current_centers, current_slopes, current_intercepts)
                if score > 0:
                    best_score = max(best_score, score)
                    if pts is not None:
                        best_points = pts
                        best_lines = lns
                    break
                    
            if best_score == 0:
                # Absolute fallback
                best_points, best_lines = _fallback_solution()
                best_score = _count_incidences(best_points, best_lines)
                # Set current state to something valid?
                # We can't easily reverse-engineer the fallback to pencil params.
                # So we will just run SA on the pencil model, and if it finds nothing, return fallback.
                # But we need current state for SA.
                # Let's just use the initial grid again, hoping SA finds a way.
                # Or better: if no valid pencil config found, skip SA and return fallback.
                # But the prompt asks for the strategy.
                # Let's assume we can find a valid pencil config.
                # To be safe, if best_score is 0, we return the fallback.
                # But we should try to improve.
                # Let's initialize current state to the first valid random found or initial.
                pass

    # Simulated Annealing
    T = 1.0
    T_min = 0.01
    cooling = 0.9995
    steps = 0
    max_steps = 200000
    
    while time.time() - start_time < timeout and T > T_min:
        # Check time
        if time.time() - start_time > timeout - 5:
            break
            
        # Perturbation
        new_centers = current_centers.copy()
        new_slopes = current_slopes.copy()
        new_intercepts = current_intercepts.copy()
        
        move_type = random.random()
        
        if move_type < 0.5:
            # Move a center
            idx = random.randint(0, 8)
            new_centers[idx, 0] += random.randint(-2, 2)
            new_centers[idx, 1] += random.randint(-2, 2)
        elif move_type < 0.8:
            # Change a slope
            idx = random.randint(0, 26)
            new_slope = slope_pool[random.randint(0, len(slope_pool)-1)]
            # Ensure slope is different from others in same pencil?
            # Ideally, lines in a pencil should have different slopes.
            # But for now, just change it.
            new_slopes[idx] = new_slope
            # Reset intercept to 0 to maintain concurrency?
            # No, changing slope breaks concurrency if intercept is 0 and center is fixed.
            # Wait, line is dy*x - dx*y + C = 0.
            # If we change dx, dy, we must update C to pass through center.
            cx, cy = new_centers[idx // 3]
            new_dx, new_dy = new_slopes[idx]
            new_C = -(new_dy * cx - new_dx * cy)
            new_intercepts[idx] = new_C
        else:
            # Jitter intercept?
            # No, intercept is fixed by center and slope.
            # So we can't jitter intercept independently.
            # Instead, move center slightly?
            idx = random.randint(0, 8)
            new_centers[idx, 0] += random.randint(-1, 1)
            new_centers[idx, 1] += random.randint(-1, 1)
            # Update intercepts for lines in this pencil
            for k in range(3):
                li = idx * 3 + k
                cx, cy = new_centers[idx]
                dx, dy = new_slopes[li]
                new_intercepts[li] = -(dy * cx - dx * cy)
                
        # Evaluate
        new_score, new_pts, new_lns = _evaluate_configuration(new_centers, new_slopes, new_intercepts)
        
        # Acceptance
        if new_score >= best_score or (new_score > 0 and random.random() < np.exp((new_score - best_score) / T)):
            # Note: if new_score is 0, we should avoid it unless current is also 0.
            # Actually, if new_score is 0, it's invalid.
            # We should only accept if new_score > 0.
            # Or accept if new_score >= current_score (if current is valid).
            # Let's track current_score separately.
            pass # Placeholder for logic below
            
        # Correct Logic:
        # We need to track current_score.
        # If new_score is 0, reject unless current_score is 0 and we want to explore?
        # Better to reject 0 scores to stay in valid space.
        
        # Re-implementing loop logic properly in code below.
        
        T *= cooling
        steps += 1
        
    # Return best
    if best_points is None or best_lines is None:
        best_points, best_lines = _fallback_solution()
        
    return best_points, best_lines

def _fallback_solution():
    """
    Generate a simple valid solution with decent incidence.
    Grid of points and lines.
    """
    # 54 points: 9x6 grid?
    # 27 lines: 9 vertical, 9 horizontal, 9 diagonal?
    
    # Let's use a 9x6 grid of points.
    points = []
    for x in range(9):
        for y in range(6):
            points.append((x, y))
            
    points_arr = np.array(points, dtype=np.int64)
    
    lines = []
    # 9 vertical lines x=0..8
    for x in range(9):
        lines.append(_normalize_line(1, 0, -x))
    # 9 horizontal lines y=0..5 (only 6 needed? No, 27 total)
    # We have 9+6=15. Need 12 more.
    # 9 diagonal lines x-y = c?
    # x-y ranges from -5 to 8.
    # Let's pick 9 diagonals.
    for c in range(-5, 4):
        lines.append(_normalize_line(1, -1, -c))
        
    # We have 9+6+9 = 24 lines. Need 3 more.
    # 3 anti-diagonals x+y = c?
    for c in range(0, 3):
        lines.append(_normalize_line(1, 1, -c))
        
    lines_arr = np.array(lines, dtype=np.int64)
    
    # Check for duplicates
    unique_lines = []
    seen = set()
    for l in lines_arr:
        t = tuple(l)
        if t not in seen:
            unique_lines.append(t)
            seen.add(t)
            
    if len(unique_lines) < 27:
        # Pad with random lines
        while len(unique_lines) < 27:
            A = random.randint(1, 5)
            B = random.randint(1, 5)
            C = random.randint(-10, 10)
            nl = _normalize_line(A, B, C)
            if nl not in seen:
                unique_lines.append(nl)
                seen.add(nl)
                
    lines_arr = np.array(unique_lines[:27], dtype=np.int64)
    
    return points_arr, lines_arr

# RE-IMPLEMENTATION OF SA LOOP WITH PROPER STATE TRACKING

def _run_sa(timeout):
    start_time = time.time()
    n_centers = 9
    n_lines = 27
    
    slope_pool = []
    for dx in range(-5, 6):
        for dy in range(-5, 6):
            if dx == 0 and dy == 0:
                continue
            if gcd(abs(dx), abs(dy)) == 1:
                if dx < 0 or (dx == 0 and dy < 0):
                    dx = -dx
                    dy = -dy
                slope_pool.append((dx, dy))
    random.shuffle(slope_pool)
    
    initial_slopes = slope_pool[:27]
    
    # Initial centers
    current_centers = np.array([
        [0, 0], [1, 0], [2, 0],
        [0, 1], [1, 1], [2, 1],
        [0, 2], [1, 2], [2, 2]
    ], dtype=np.int64)
    
    current_slopes = np.array(initial_slopes)
    
    # Initialize intercepts
    current_intercepts = np.zeros(27, dtype=np.int64)
    for i in range(9):
        cx, cy = current_centers[i]
        for k in range(3):
            li = i * 3 + k
            dx, dy = current_slopes[li]
            current_intercepts[li] = -(dy * cx - dx * cy)
            
    current_score, current_points, current_lines = _evaluate_configuration(current_centers, current_slopes, current_intercepts)
    
    best_score = current_score
    best_points = current_points
    best_lines = current_lines
    
    if best_score == 0:
        # Try to find a valid start
        for _ in range(200):
            trial_centers = np.random.randint(-5, 5, (9, 2)).astype(np.int64)
            trial_intercepts = np.zeros(27, dtype=np.int64)
            for i in range(9):
                cx, cy = trial_centers[i]
                for k in range(3):
                    li = i * 3 + k
                    dx, dy = current_slopes[li]
                    trial_intercepts[li] = -(dy * cx - dx * cy)
            s, p, l = _evaluate_configuration(trial_centers, current_slopes, trial_intercepts)
            if s > 0:
                current_centers = trial_centers
                current_intercepts = trial_intercepts
                current_score = s
                current_points = p
                current_lines = l
                best_score = s
                best_points = p
                best_lines = l
                break
                
    if best_score == 0:
        return None, None, 0
        
    T = 1.0
    T_min = 0.01
    cooling = 0.9995
    
    while time.time() - start_time < timeout - 5 and T > T_min:
        new_centers = current_centers.copy()
        new_slopes = current_slopes.copy()
        new_intercepts = current_intercepts.copy()
        
        move_type = random.random()
        
        if move_type < 0.6:
            # Move a center
            idx = random.randint(0, 8)
            new_centers[idx, 0] += random.randint(-2, 2)
            new_centers[idx, 1] += random.randint(-2, 2)
            # Update intercepts for this pencil
            cx, cy = new_centers[idx]
            for k in range(3):
                li = idx * 3 + k
                dx, dy = new_slopes[li]
                new_intercepts[li] = -(dy * cx - dx * cy)
        else:
            # Change a slope
            idx = random.randint(0, 26)
            # Pick a new slope different from others in same pencil
            pencil_idx = idx // 3
            existing_slopes = set()
            for k in range(3):
                li = pencil_idx * 3 + k
                if li != idx:
                    existing_slopes.add(tuple(new_slopes[li]))
                    
            new_slope = None
            for _ in range(10):
                cand = slope_pool[random.randint(0, len(slope_pool)-1)]
                if cand not in existing_slopes:
                    new_slope = cand
                    break
            if new_slope is None:
                new_slope = slope_pool[random.randint(0, len(slope_pool)-1)]
                
            new_slopes[idx] = new_slope
            # Update intercept
            cx, cy = new_centers[pencil_idx]
            dx, dy = new_slope
            new_intercepts[idx] = -(dy * cx - dx * cy)
            
        new_score, new_points, new_lines = _evaluate_configuration(new_centers, new_slopes, new_intercepts)
        
        # Acceptance
        if new_score > 0:
            if new_score >= current_score:
                current_centers = new_centers
                current_slopes = new_slopes
                current_intercepts = new_intercepts
                current_score = new_score
                current_points = new_points
                current_lines = new_lines
            else:
                # Metropolis
                delta = current_score - new_score
                if random.random() < np.exp(-delta / T):
                    current_centers = new_centers
                    current_slopes = new_slopes
                    current_intercepts = new_intercepts
                    current_score = new_score
                    current_points = new_points
                    current_lines = new_lines
                    
            if current_score > best_score:
                best_score = current_score
                best_points = current_points
                best_lines = current_lines
                
        T *= cooling
        
    return best_points, best_lines, best_score

def szemeredi_trotter_incidences():
    start_time = time.time()
    timeout = 590
    
    best_points, best_lines, best_score = _run_sa(timeout)
    
    if best_points is None or best_lines is None:
        best_points, best_lines = _fallback_solution()
        
    return best_points, best_lines

# EVOLVE-BLOCK-END
