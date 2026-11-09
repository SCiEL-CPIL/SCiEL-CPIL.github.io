import numpy as np
import time
import random
import math
from math import gcd

def _normalize_line(A, B, C):
    g = gcd(abs(A), gcd(abs(B), abs(C)))
    if g == 0:
        return (0, 0, 0)
    A, B, C = A // g, B // g, C // g
    if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
        A, B, C = -A, -B, -C
    return (A, B, C)

def _line_key(A, B, C):
    return _normalize_line(A, B, C)

def _is_valid(points, lines):
    if points.shape != (54, 2) or lines.shape != (27, 3):
        return False
    pt_set = set()
    for i in range(54):
        x, y = points[i]
        if not np.isfinite(x) or not np.isfinite(y):
            return False
        if abs(x - round(x)) > 1e-6 or abs(y - round(y)) > 1e-6:
            return False
        xi, yi = int(round(x)), int(round(y))
        if (xi, yi) in pt_set:
            return False
        pt_set.add((xi, yi))
    line_set = set()
    for i in range(27):
        A, B, C = lines[i]
        if not np.isfinite(A) or not np.isfinite(B) or not np.isfinite(C):
            return False
        if abs(A - round(A)) > 1e-6 or abs(B - round(B)) > 1e-6 or abs(C - round(C)) > 1e-6:
            return False
        Ai, Bi, Ci = int(round(A)), int(round(B)), int(round(C))
        if Ai == 0 and Bi == 0:
            return False
        key = _line_key(Ai, Bi, Ci)
        if key in line_set:
            return False
        line_set.add(key)
    return True

def _count_incidentences(points, lines):
    n = len(points)
    m = len(lines)
    count = 0
    for i in range(n):
        x, y = points[i]
        for j in range(m):
            A, B, C = lines[j]
            if A * x + B * y + C == 0:
                count += 1
    return count

def _generate_line_from_two_points(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    A = y2 - y1
    B = x1 - x2
    C = x2 * y1 - x1 * y2
    if A == 0 and B == 0:
        return None
    return _line_key(A, B, C)

def _best_lines_for_points(points):
    # Generate all unique lines defined by pairs of points
    n = len(points)
    line_set = set()
    for i in range(n):
        for j in range(i+1, n):
            key = _generate_line_from_two_points(points[i], points[j])
            if key is not None:
                line_set.add(key)
    
    # If we have fewer than 27 lines, we need to add arbitrary lines? 
    # The problem says "27 lines". If pairs don't generate enough, we might have an issue.
    # With 54 points, there are ~1400 pairs. So we have plenty.
    
    # Calculate incidences for each candidate line
    candidates = []
    for key in line_set:
        A, B, C = key
        cnt = 0
        for p in points:
            x, y = p
            if A * x + B * y + C == 0:
                cnt += 1
        candidates.append((cnt, key))
    
    # Sort by count descending
    candidates.sort(key=lambda x: x[0], reverse=True)
    
    # Pick top 27
    top_keys = [k for _, k in candidates[:27]]
    
    # Ensure validity (uniqueness is guaranteed by set, but let's check A!=0 or B!=0 which is in _line_key)
    lines_arr = np.array(top_keys, dtype=np.int64)
    
    # If for some reason we don't have 27 valid unique lines (shouldn't happen with 54 distinct points),
    # we might need a fallback, but let's assume the set is large enough.
    
    return lines_arr

def szemeredi_trotter_incidences():
    start_time = time.time()
    deadline = start_time + 590  # Safety margin
    
    # Initialize with a good grid
    xs = list(range(1, 7))
    ys = list(range(1, 10))
    current_points = np.array([(x, y) for x in xs for y in ys], dtype=np.int64)
    current_lines = _best_lines_for_points(current_points)
    
    if not _is_valid(current_points, current_lines):
        # Fallback
        return current_points, np.zeros((27, 3), dtype=np.int64)
        
    current_score = _count_incidentences(current_points, current_lines)
    best_points = current_points.copy()
    best_lines = current_lines.copy()
    best_score = current_score
    
    # Simulated Annealing Parameters
    T = 10.0
    cooling_rate = 0.999
    
    # Precompute point indices for quick access
    n = 54
    
    while time.time() < deadline:
        # Generate a neighbor
        # 50% chance move a point, 50% chance move a line (by regenerating from pairs)
        # Actually, moving a point invalidates the "best lines" property unless we recompute.
        # Recomputing best lines takes O(N^2 * N) = O(N^3) ~ 150k ops. 
        # 500s / 150k ops ~ 3000 iterations. That's small.
        # Let's optimize the line finding.
        
        # Alternative: Perturb a point slightly, then update lines greedily?
        # Or: Perturb a point, and keep lines fixed? No, lines must be best for points to maximize.
        # The objective is max I(P, L). For any fixed P, best L is top 27 pairs.
        # So the search is effectively over P.
        # Let's focus search on P, and always recompute optimal L for that P.
        
        # Move: Pick a random point, displace it by a small random integer vector
        idx = random.randint(0, n - 1)
        dx = random.randint(-3, 3)
        dy = random.randint(-3, 3)
        if dx == 0 and dy == 0:
            continue
            
        new_points = current_points.copy()
        new_points[idx][0] += dx
        new_points[idx][1] += dy
        
        # Check for duplicate points
        pt_set = set()
        valid = True
        for p in new_points:
            key = (int(p[0]), int(p[1]))
            if key in pt_set:
                valid = False
                break
            pt_set.add(key)
        
        if not valid:
            continue
            
        # Find best lines for new_points
        new_lines = _best_lines_for_points(new_points)
        
        # Check validity of lines (should be fine if points are distinct)
        if not _is_valid(new_points, new_lines):
            continue
            
        new_score = _count_incidentences(new_points, new_lines)
        
        # Acceptance criterion
        delta = new_score - current_score
        if delta >= 0 or random.random() < math.exp(delta / T):
            current_points = new_points
            current_lines = new_lines
            current_score = new_score
            
            if current_score > best_score:
                best_score = current_score
                best_points = current_points.copy()
                best_lines = current_lines.copy()
        
        # Cool down
        T *= cooling_rate
        if T < 0.01:
            T = 0.01
            
        # Optional: Random restart if stuck for too long? 
        # Or just let it run.
        
        # Every few iterations, try a larger jump?
        # For now, simple SA.
        
    return best_points, best_lines

# To speed up _best_lines_for_points, we can cache or optimize.
# But 54 points -> 1431 pairs. 
# For each pair, calculate line. 
# Then for each line, count incidences with 54 points.
# 1431 * 54 ~ 77k operations.
# 77k ops is very fast in Python (maybe 5-10ms).
# 500s / 10ms = 50,000 iterations. That's plenty.

# Let's double check the logic.
# The task is to maximize incidences.
# For a fixed set of points, the optimal set of lines is indeed the top K lines by incidence count.
# Why? Because the lines are independent in the sum. Sum of incidences = sum over lines of (incidences of that line).
# To maximize the sum, we pick the lines with the highest individual counts.
# So the problem reduces to: Find P (54 points) that maximizes S(P) = sum of top 27 line counts in P.
# My SA searches P, and for each P, computes the optimal L.
# This is correct.

# One potential issue: _best_lines_for_points might return lines that are not "valid" in the sense of the problem?
# The problem requires lines to be distinct and A!=0 or B!=0.
# _generate_line_from_two_points ensures A!=0 or B!=0 (unless points are same, which we prevent).
# _line_key ensures normalization.
# The set ensures uniqueness.
# So it should be valid.

# Let's add a small optimization: if the set of points changes, the set of candidate lines changes.
# We recompute from scratch.

# Let's run a quick mental check on performance.
# 50,000 iterations * (1431 pairs + 54*1431 checks)
# 50,000 * 77,000 = 3.85 * 10^9 operations.
# Python does ~10^7 simple ops/sec? Maybe slower.
# 3.85 * 10^9 / 10^7 = 385 seconds.
# This might be too slow!
# We need to speed up _count_incidentences or _best_lines_for_points.

# Optimization:
# Instead of counting incidences for ALL 1431 lines, we can be smarter?
# Or use Numpy for the counting step.
# Vectorize the incidence check.

def _best_lines_for_points_fast(points):
    n = len(points)
    # Generate lines
    lines = []
    seen = set()
    for i in range(n):
        for j in range(i+1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            A = y2 - y1
            B = x1 - x2
            C = x2 * y1 - x1 * y2
            if A == 0 and B == 0: continue
            key = _line_key(A, B, C)
            if key not in seen:
                seen.add(key)
                lines.append(key)
                
    if len(lines) < 27:
        # Should not happen, but handle gracefully
        # Pad with arbitrary lines? No, just return what we have and pad with zeros?
        # But zeros are invalid. 
        # With 54 distinct points, we always have >= 27 distinct lines.
        pass
        
    # Convert lines to numpy array (27, 3) or (L, 3)
    lines_arr = np.array(lines, dtype=np.int64)
    
    # Vectorized incidence count
    # points: (54, 2)
    # lines_arr: (L, 3)
    # We want for each line j, sum over i of (A_j*x_i + B_j*y_i + C_j == 0)
    
    # Expand:
    # x: (54, 1)
    # y: (54, 1)
    # A: (1, L)
    # B: (1, L)
    # C: (1, L)
    
    x = points[:, 0:1].astype(np.int64)
    y = points[:, 1:2].astype(np.int64)
    
    A = lines_arr[:, 0:1].T.astype(np.int64)
    B = lines_arr[:, 1:2].T.astype(np.int64)
    C = lines_arr[:, 2:3].T.astype(np.int64)
    
    # Check for overflow? int64 should be fine for small coordinates.
    # If coordinates are large, A*x might overflow. 
    # Current grid is 1..6, 1..9. SA moves by +/-3. So coords stay small.
    
    val = A * x + B * y + C
    incidences = np.sum(val == 0, axis=0)
    
    # Get top 27 indices
    # If L > 27
    if len(incidences) > 27:
        # Get indices of top 27
        # np.argpartition is faster than sort
        top_idx = np.argpartition(incidences, -27)[-27:]
        # Sort them by count descending? Not strictly necessary for validity, but good for consistency.
        # Actually, we just need the lines.
        top_lines = lines_arr[top_idx]
    else:
        top_lines = lines_arr
        
    # Ensure shape is (27, 3)
    if len(top_lines) < 27:
        # Pad with dummy lines? This would make it invalid if not handled.
        # But as established, L should be >= 27.
        # If not, we return what we have and let _is_valid fail? 
        # Better to ensure we have 27.
        # If we have < 27, we have a problem.
        # Let's assume we have enough.
        pass
        
    return np.array(top_lines, dtype=np.int64)

# Update the main function to use the fast version

def szemeredi_trotter_incidences():
    start_time = time.time()
    deadline = start_time + 590  # Safety margin
    
    # Initialize with a good grid
    xs = list(range(1, 7))
    ys = list(range(1, 10))
    current_points = np.array([(x, y) for x in xs for y in ys], dtype=np.int64)
    current_lines = _best_lines_for_points_fast(current_points)
    
    if not _is_valid(current_points, current_lines):
        # Fallback
        return current_points, np.zeros((27, 3), dtype=np.int64)
        
    current_score = _count_incidentences(current_points, current_lines)
    best_points = current_points.copy()
    best_lines = current_lines.copy()
    best_score = current_score
    
    # Simulated Annealing Parameters
    T = 10.0
    cooling_rate = 0.999
    
    n = 54
    
    while time.time() < deadline:
        idx = random.randint(0, n - 1)
        dx = random.randint(-3, 3)
        dy = random.randint(-3, 3)
        if dx == 0 and dy == 0:
            continue
            
        new_points = current_points.copy()
        new_points[idx][0] += dx
        new_points[idx][1] += dy
        
        pt_set = set()
        valid = True
        for p in new_points:
            key = (int(p[0]), int(p[1]))
            if key in pt_set:
                valid = False
                break
            pt_set.add(key)
        
        if not valid:
            continue
            
        new_lines = _best_lines_for_points_fast(new_points)
        
        if not _is_valid(new_points, new_lines):
            continue
            
        new_score = _count_incidentences(new_points, new_lines)
        
        delta = new_score - current_score
        if delta >= 0 or random.random() < math.exp(delta / T):
            current_points = new_points
            current_lines = new_lines
            current_score = new_score
            
            if current_score > best_score:
                best_score = current_score
                best_points = current_points.copy()
                best_lines = current_lines.copy()
        
        T *= cooling_rate
        if T < 0.01:
            T = 0.01

    return best_points, best_lines
