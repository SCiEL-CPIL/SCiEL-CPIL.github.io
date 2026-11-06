import time
import random
from itertools import combinations
from collections import defaultdict

def _count_exact_four_lines(points):
    """Count lines containing exactly 4 of the given points."""
    n = len(points)
    if n < 4:
        return 0
    # Map from line (as a normalized tuple) to set of point indices
    line_map = {}
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            # Direction vector
            dx = x2 - x1
            dy = y2 - y1
            # Normalize direction
            g = abs(_gcd(dx, dy))
            if g == 0:
                continue
            dx_n = dx // g
            dy_n = dy // g
            # Ensure consistent orientation
            if dx_n < 0 or (dx_n == 0 and dy_n < 0):
                dx_n = -dx_n
                dy_n = -dy_n
            # Compute a canonical "offset" for the line: use the cross product
            # to get a unique representation. The line equation: dy_n * x - dx_n * y = c
            c = dy_n * x1 - dx_n * y1
            key = (dx_n, dy_n, c)
            if key not in line_map:
                line_map[key] = set()
            line_map[key].add(i)
            line_map[key].add(j)
    
    count = 0
    for line in line_map.values():
        if len(line) == 4:
            count += 1
        elif len(line) >= 5:
            return -1  # Invalid: five or more collinear
    return count

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return abs(a)

def _has_five_collinear(points):
    """Check if any line contains 5 or more points."""
    n = len(points)
    line_map = {}
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            g = abs(_gcd(dx, dy))
            if g == 0:
                continue
            dx_n = dx // g
            dy_n = dy // g
            if dx_n < 0 or (dx_n == 0 and dy_n < 0):
                dx_n = -dx_n
                dy_n = -dy_n
            c = dy_n * x1 - dx_n * y1
            key = (dx_n, dy_n, c)
            if key not in line_map:
                line_map[key] = set()
            line_map[key].add(i)
            line_map[key].add(j)
    for line in line_map.values():
        if len(line) >= 5:
            return True
    return False

def construct():
    start_time = time.time()
    deadline = start_time + 120  # Safety margin before 150s kill
    
    # Start with the 4x4 grid as baseline
    base_points = [[x, y] for x in range(4) for y in range(4)]
    best_score = _count_exact_four_lines(base_points)
    best_points = [p[:] for p in base_points]
    
    if best_score < 0:
        # Shouldn't happen with 4x4 grid
        best_score = 0
    
    # Try to improve by perturbing points
    # The 4x4 grid has 4 horizontal + 4 vertical + 2 diagonal = 10 lines with exactly 4 points
    # (the other two diagonals of a 4x4 grid don't have 4 points each... let me verify)
    # Actually in a 4x4 grid:
    # Rows: 4 lines of 4
    # Cols: 4 lines of 4
    # Main diagonal: 4 points
    # Anti-diagonal: 4 points
    # So 10 lines total with exactly 4 points
    # No 5 collinear
    
    # Strategy: try moving individual points to new positions to create more 4-point lines
    # while maintaining no 5-collinear constraint
    
    random.seed(42)
    
    # Precompute all candidate positions
    candidates = []
    for x in range(-20, 21):
        for y in range(-20, 21):
            candidates.append([x, y])
    
    def score_config(points):
        s = _count_exact_four_lines(points)
        return s if s >= 0 else -1000
    
    current_score = best_score
    current_points = [p[:] for p in best_points]
    
    # Local search: try moving one point at a time
    improved = True
    while improved and (time.time() - start_time) < deadline - 10:
        improved = False
        for idx in range(16):
            if (time.time() - start_time) > deadline - 10:
                break
            orig = current_points[idx][:]
            # Try random candidate positions
            random.shuffle(candidates)
            for cand in candidates[:50]:
                if (time.time() - start_time) > deadline - 10:
                    break
                # Check if cand is already used
                if any(cand[0] == current_points[i][0] and cand[1] == current_points[i][1] 
                       for i in range(16) if i != idx):
                    continue
                new_points = [p[:] for p in current_points]
                new_points[idx] = cand[:]
                s = score_config(new_points)
                if s > current_score:
                    current_score = s
                    current_points = new_points
                    improved = True
                    break
            if current_points[idx] != orig:
                pass  # already updated
            else:
                # Try a few more targeted moves
                for dx in range(-5, 6):
                    for dy in range(-5, 6):
                        if dx == 0 and dy == 0:
                            continue
                        if (time.time() - start_time) > deadline - 10:
                            break
                        nx = orig[0] + dx
                        ny = orig[1] + dy
                        if abs(nx) > 1000 or abs(ny) > 1000:
                            continue
                        cand = [nx, ny]
                        if any(cand[0] == current_points[i][0] and cand[1] == current_points[i][1] 
                               for i in range(16) if i != idx):
                            continue
                        new_points = [p[:] for p in current_points]
                        new_points[idx] = cand[:]
                        s = score_config(new_points)
                        if s > current_score:
                            current_score = s
                            current_points = new_points
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
    
    # Update best
    if current_score > best_score:
        best_score = current_score
        best_points = [p[:] for p in current_points]
    
    # Try a second phase: more aggressive random restarts with local search
    for restart in range(20):
        if (time.time() - start_time) > deadline - 5:
            break
        
        # Start from best and perturb 2-3 random points
        trial = [p[:] for p in best_points]
        num_perturb = random.randint(2, 4)
        for _ in range(num_perturb):
            idx = random.randint(0, 15)
            while True:
                cand = [random.randint(-15, 15), random.randint(-15, 15)]
                if not any(cand[0] == trial[i][0] and cand[1] == trial[i][1] 
                          for i in range(16) if i != idx):
                    break
            trial[idx] = cand
        
        trial_score = score_config(trial)
        
        # Local search on trial
        t_improved = True
        while t_improved and (time.time() - start_time) < deadline - 5:
            t_improved = False
            for idx in range(16):
                if (time.time() - start_time) > deadline - 5:
                    break
                orig = trial[idx][:]
                for dx in range(-3, 4):
                    for dy in range(-3, 4):
                        if dx == 0 and dy == 0:
                            continue
                        nx = orig[0] + dx
                        ny = orig[1] + dy
                        if abs(nx) > 1000 or abs(ny) > 1000:
                            continue
                        cand = [nx, ny]
                        if any(cand[0] == trial[i][0] and cand[1] == trial[i][1] 
                               for i in range(16) if i != idx):
                            continue
                        new_points = [p[:] for p in trial]
                        new_points[idx] = cand[:]
                        s = score_config(new_points)
                        if s > trial_score:
                            trial_score = s
                            trial = new_points
                            t_improved = True
                            break
                    if t_improved:
                        break
                if t_improved:
                    break
        
        if trial_score > best_score:
            best_score = trial_score
            best_points = [p[:] for p in trial]
    
    return [p[:] for p in best_points]
