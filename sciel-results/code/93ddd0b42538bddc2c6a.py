import numpy as np
import time
import random
import math

def _normalize_line(A, B, C):
    g = math.gcd(math.gcd(abs(A), abs(B)), abs(C))
    if g == 0:
        return (0, 0, 0)
    A //= g
    B //= g
    C //= g
    if A < 0 or (A == 0 and B < 0):
        A, B, C = -A, -B, -C
    return (A, B, C)

def _line_key(A, B, C):
    return _normalize_line(A, B, C)

def _is_valid(points, lines):
    # Check shapes
    if points.shape != (54, 2) or lines.shape != (27, 3):
        return False
    if points.dtype != np.int64 or lines.dtype != np.int64:
        return False
    
    # Check points
    if not np.all(np.isfinite(points)):
        return False
    # Check for repeated points
    # Sort and check for duplicates
    sorted_points = points[np.lexsort((points[:, 1], points[:, 0]))]
    for i in range(1, 54):
        if sorted_points[i, 0] == sorted_points[i-1, 0] and sorted_points[i, 1] == sorted_points[i-1, 1]:
            return False
            
    # Check lines
    seen_lines = set()
    for i in range(27):
        A, B, C = lines[i]
        if A == 0 and B == 0:
            return False
        key = _line_key(A, B, C)
        if key in seen_lines:
            return False
        seen_lines.add(key)
        
    return True

def _count_incidences(points, lines):
    # Vectorized calculation
    # points: (N, 2), lines: (M, 3)
    # We want sum over i, j of (A_i * x_j + B_i * y_j + C_i == 0)
    # This is equivalent to summing for each line the number of points on it.
    # Using broadcasting: (M, 1, 2) * (1, N, 2) -> (M, N, 2) sum axis 2 + C
    # M=27, N=54. 27*54*2 ops is very fast.
    pts = points.astype(np.int64)
    lns = lines.astype(np.int64)
    
    # (27, 1, 2) * (1, 54, 2)
    # A*x + B*y
    terms = lns[:, np.newaxis, :] * pts[np.newaxis, :, :]
    # terms is (27, 54, 3). We want sum of first two components? No.
    # A is lns[:, 0], B is lns[:, 1], C is lns[:, 2]
    # pts[:, 0] is x, pts[:, 1] is y
    # val = A*x + B*y + C
    # val[i, j] = lns[i, 0]*pts[j, 0] + lns[i, 1]*pts[j, 1] + lns[i, 2]
    
    # Efficient way:
    # Ax = lns[:, 0:1] * pts[np.newaxis, :, 0]  # (27, 54)
    # By = lns[:, 1:2] * pts[np.newaxis, :, 1]  # (27, 54)
    # C  = lns[:, 2:3] * np.ones((1, 54))       # (27, 54)
    # Total = Ax + By + C
    # Count = np.sum(Total == 0)
    
    Ax = lns[:, 0:1] * pts[np.newaxis, :, 0]
    By = lns[:, 1:2] * pts[np.newaxis, :, 1]
    C = lns[:, 2:3]
    total = Ax + By + C
    return np.sum(total == 0)

def _generate_candidate_lines(points, max_coeff=5):
    # Generate all lines passing through at least 2 points? Or just all small coefficient lines?
    # For SA, we need to be able to swap lines.
    # A good pool of lines is all lines with small A, B that pass through the current points.
    # But points change.
    # Strategy: Generate a pool of lines based on the *initial* grid or current points?
    # If we generate based on current points, the pool changes every step.
    # Let's generate a static pool of "simple" lines (small A, B) and check incidence.
    # Or better: Generate lines defined by pairs of points in the current set?
    # N=54, pairs = 1431. This is a manageable pool.
    # Plus axis-aligned and simple diagonal lines.
    
    pool = set()
    N = len(points)
    # Lines from pairs of points
    for i in range(N):
        for j in range(i+1, N):
            x1, y1 = points[i]
            x2, y2 = points[j]
            A = y1 - y2
            B = x2 - x1
            C = x1*y2 - x2*y1
            key = _line_key(A, B, C)
            if key != (0, 0, 0):
                pool.add(key)
                
    # Also add some standard grid lines just in case points drift
    # But the pair-based pool should cover most "dense" lines in the current configuration.
    # To ensure we don't lose track of "good" lines that might not be through 2 points anymore
    # (if points moved), we might need a broader pool.
    # However, the objective is to maximize incidences. Lines with high incidence
    # are likely to be defined by multiple points, so pair-based is a strong candidate.
    # Let's also add all lines with |A|,|B| <= 2 that pass through at least one point?
    # No, that's too many.
    
    return list(pool)

def szemeredi_trotter_incidences():
    start_time = time.time()
    deadline = start_time + 590  # Safety margin
    
    # Initial configuration: 6x9 grid
    # 54 points
    k, l = 6, 9
    xs = np.arange(1, k + 1)
    ys = np.arange(1, l + 1)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    
    # Initial lines: Top 27 lines from this grid
    candidates = []
    seen = set()
    # Small coefficients for grid lines
    for A in range(-3, 4):
        for B in range(-3, 4):
            if A == 0 and B == 0: continue
            for C in range(-20, 21):
                key = _line_key(A, B, C)
                if key in seen: continue
                seen.add(key)
                # Count
                dots = points[:, 0] * A + points[:, 1] * B + C
                count = np.sum(dots == 0)
                if count > 0:
                    candidates.append((count, A, B, C))
    
    candidates.sort(key=lambda x: x[0], reverse=True)
    lines_list = []
    seen_sel = set()
    for count, A, B, C in candidates:
        if len(lines_list) == 27: break
        key = _line_key(A, B, C)
        if key in seen_sel: continue
        seen_sel.add(key)
        lines_list.append((A, B, C))
        
    lines = np.array(lines_list, dtype=np.int64)
    
    best_points = points.copy()
    best_lines = lines.copy()
    best_score = _count_incidences(points, lines)
    
    # Simulated Annealing
    T = 100.0
    cooling = 0.999
    
    current_points = points.copy()
    current_lines = lines.copy()
    current_score = best_score
    
    # Precompute a pool of lines? 
    # The pool of lines depends on points. 
    # Instead of regenerating the full pool every step (expensive), 
    # we can propose moves on lines by:
    # 1. Swapping a current line with a random line from a pre-computed "global" pool of simple lines.
    # 2. Or perturbing the line coefficients.
    
    # Let's use a hybrid:
    # - Point moves: perturb one point by a small integer amount.
    # - Line moves: replace one line with a line from a dynamic pool generated from current points (top 100 lines by incidence).
    
    # To speed up, we'll generate the dynamic pool every K steps or on demand.
    # Given N=54, generating all pair lines (1431) and sorting is fast (< 1ms).
    
    step = 0
    while time.time() < deadline:
        if step % 100 == 0 and time.time() > deadline - 5:
            break
            
        # Propose move
        if random.random() < 0.5:
            # Move Point
            idx = random.randint(0, 53)
            dx = random.randint(-2, 2)
            dy = random.randint(-2, 2)
            if dx == 0 and dy == 0:
                continue
                
            old_x, old_y = current_points[idx]
            new_x, new_y = old_x + dx, old_y + dy
            
            # Check if new point is duplicate
            # Simple check: if (new_x, new_y) is in current_points (excluding idx)
            # This is O(N). N=54.
            dup = False
            for i in range(54):
                if i == idx: continue
                if current_points[i, 0] == new_x and current_points[i, 1] == new_y:
                    dup = True
                    break
            
            if dup:
                continue
                
            current_points[idx, 0] = new_x
            current_points[idx, 1] = new_y
            
            new_score = _count_incidences(current_points, current_lines)
            
            delta = new_score - current_score
            if delta > 0 or random.random() < math.exp(delta / T):
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_points = current_points.copy()
                    best_lines = current_lines.copy()
            else:
                current_points[idx, 0] = old_x
                current_points[idx, 1] = old_y
        else:
            # Move Line
            # Generate candidate lines from current points
            # We want to replace one line with a better one.
            # Find the line with lowest incidence? Or random?
            # Let's pick a random line to replace.
            idx = random.randint(0, 26)
            
            # Generate pool of lines from current points
            # To save time, we can just generate lines from pairs and pick a random one,
            # or pick the best one not already in the set.
            # Generating all 1431 lines and finding the best new one is O(N^2 + N^2 log N) ~ 2000 ops.
            # This is fast.
            
            pool = set()
            # Add current lines to pool to avoid duplicates? No, we want to ADD new lines.
            # We want to replace lines[idx] with a line NOT in current_lines.
            
            # Generate all lines from pairs
            N = 54
            line_counts = {}
            for i in range(N):
                for j in range(i+1, N):
                    x1, y1 = current_points[i]
                    x2, y2 = current_points[j]
                    A = y1 - y2
                    B = x2 - x1
                    C = x1*y2 - x2*y1
                    key = _line_key(A, B, C)
                    if key != (0, 0, 0):
                        if key not in line_counts:
                            line_counts[key] = 0
                        # We need to count incidences for this line.
                        # Doing this inside the loop is expensive if we count for every pair.
                        # Better: collect unique keys, then count.
                        pass
            
            # Unique keys
            unique_keys = list(line_counts.keys())
            
            # Count incidences for each unique key
            # This is O(U * N) where U is number of unique lines. U <= 1431.
            # 1431 * 54 ~ 77k ops. Very fast.
            
            best_new_line = None
            best_new_count = -1
            
            # Optimization: We only need to consider lines that are NOT in current_lines.
            current_keys = set()
            for i in range(27):
                current_keys.add(_line_key(current_lines[i, 0], current_lines[i, 1], current_lines[i, 2]))
                
            # We can filter unique_keys to exclude current_keys
            candidates_to_check = [k for k in unique_keys if k not in current_keys]
            
            # If no candidates, skip
            if not candidates_to_check:
                continue
                
            # Pick a random subset to check? Or check all?
            # Checking all is fast enough.
            # But we want the BEST new line to maximize chance of improvement.
            
            # To speed up, we can just pick a random candidate and see if it improves?
            # SA usually works with random proposals.
            # Let's pick a random candidate from candidates_to_check.
            
            # Actually, picking the best candidate is a "greedy" step within SA, which might be okay
            # but reduces exploration. Let's pick a random one.
            
            # To make it more effective, let's pick from the top 10% of candidates by incidence?
            # That requires counting all.
            
            # Let's just count all and pick the best one that is not currently present.
            # This is a "Best Improvement" move for the line part.
            
            best_key = None
            best_count_val = 0
            
            # Vectorized counting for all candidates?
            # candidates_to_check is a list of tuples (A, B, C).
            # Convert to array
            if candidates_to_check:
                cand_arr = np.array(candidates_to_check, dtype=np.int64)
                # (M, 1, 2) * (1, N, 2)
                Ax = cand_arr[:, 0:1] * current_points[np.newaxis, :, 0]
                By = cand_arr[:, 1:2] * current_points[np.newaxis, :, 1]
                C = cand_arr[:, 2:3]
                total = Ax + By + C
                counts = np.sum(total == 0, axis=1)
                
                # Find max
                max_idx = np.argmax(counts)
                if counts[max_idx] > 0:
                    best_key = candidates_to_check[max_idx]
                    best_count_val = counts[max_idx]
                    
            if best_key is None:
                continue
                
            # Proposed new line
            new_line = np.array([best_key[0], best_key[1], best_key[2]], dtype=np.int64)
            
            # Calculate new score
            # Old line incidence:
            old_A, old_B, old_C = current_lines[idx]
            old_dots = current_points[:, 0] * old_A + current_points[:, 1] * old_B + old_C
            old_count = np.sum(old_dots == 0)
            
            # New line incidence:
            new_count = best_count_val
            
            delta = new_count - old_count
            
            if delta > 0 or random.random() < math.exp(delta / T):
                current_lines[idx] = new_line
                current_score += delta
                if current_score > best_score:
                    best_score = current_score
                    best_points = current_points.copy()
                    best_lines = current_lines.copy()
            # If rejected, no change needed as we didn't modify current_lines
            
        step += 1
        T *= cooling
        
    # Final check
    if _is_valid(best_points, best_lines):
        return best_points, best_lines
    else:
        # Fallback to initial grid
        return points, lines

szemeredi_trotter_incidences()
