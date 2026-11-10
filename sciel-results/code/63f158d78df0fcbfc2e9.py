import time
import random
import math
from collections import defaultdict

def _get_line_key(p1, p2):
    """
    Returns a normalized tuple (A, B, C) representing the line Ax + By + C = 0
    passing through p1 and p2.
    """
    x1, y1 = p1
    x2, y2 = p2
    A = y1 - y2
    B = x2 - x1
    C = x1 * y2 - x2 * y1
    
    # Normalize
    if A == 0 and B == 0:
        return None
        
    g = math.gcd(math.gcd(abs(A), abs(B)), abs(C))
    if g > 0:
        A //= g
        B //= g
        C //= g
        
    # Canonical form: A > 0, or A == 0 and B > 0, or A == 0 and B == 0 and C > 0
    if A < 0:
        A = -A
        B = -B
        C = -C
    elif A == 0:
        if B < 0:
            B = -B
            C = -C
        elif B == 0:
            if C < 0:
                C = -C
    return (A, B, C)

def _calculate_line_counts(points):
    """
    Calculates the number of points on each line defined by pairs of points.
    Returns a dict: line_key -> count
    """
    n = len(points)
    line_counts = defaultdict(int)
    for i in range(n):
        for j in range(i + 1, n):
            key = _get_line_key(points[i], points[j])
            if key:
                line_counts[key] += 1
    return line_counts

def _score_from_counts(line_counts):
    """
    Given line_counts, returns (num_4_lines, is_invalid)
    """
    count_4 = 0
    is_invalid = False
    for count in line_counts.values():
        if count == 4:
            count_4 += 1
        elif count >= 5:
            is_invalid = True
            break
    return count_4, is_invalid

def _delta_delta_score(points, idxs, new_points):
    """
    Efficiently calculates the change in score when moving points at indices `idxs` to `new_points`.
    `points` is the current list of 16 points.
    `new_points` is a list of new coordinates corresponding to `idxs`.
    Returns (delta_score, is_invalid_new, is_invalid_old)
    
    Logic:
    1. Identify all lines that contain at least one point from `idxs`.
    2. For each such line, record its old count.
    3. Remove contributions of old points to these lines.
    4. Add contributions of new points to these lines.
    5. Compare old and new counts to determine delta in 4-lines and validity.
    """
    n = len(points)
    old_points = [points[i] for i in idxs]
    new_pts = new_points
    
    # 1. Find all lines affected
    # A line is affected if it contains any of the moved points.
    # We need to know the count of these lines in the old configuration.
    
    # First, compute counts for lines involving moved points in the OLD configuration.
    # We don't need the full line_counts, just the ones that touch the moved points.
    # However, a line might not touch a moved point but its count changes? No.
    # Only lines passing through at least one moved point change count.
    # Lines passing through NO moved points keep the same count.
    
    # So, we need to:
    # A. Calculate old counts for all lines passing through any moved point.
    # B. Calculate new counts for all lines passing through any new point.
    # C. The set of "affected lines" is the union of lines from A and B.
    # D. For lines in A but not B, their count decreases.
    # E. For lines in B but not A, their count increases.
    # F. For lines in both, their count changes by (new_contrib - old_contrib).
    
    # Let's build the old counts for lines touching moved points.
    old_line_counts = defaultdict(int)
    for i in idxs:
        p_i = points[i]
        for j in range(n):
            if j in idxs:
                continue
            # Line between p_i and p_j
            # Wait, if we iterate all pairs involving moved points, we might count the same line multiple times?
            # No, we are counting points on the line.
            # A line is defined by any pair.
            # If we iterate over all pairs (u, v) where u is in idxs and v is any other point,
            # we find all lines that pass through at least one moved point.
            # We need the TOTAL count of points on that line in the old config.
            # The current method of summing pairs only gives the number of PAIRS on the line.
            # If a line has k points, it contributes C(k, 2) to the pair count.
            # We need k.
            
            # This approach is tricky because we don't have the global line_counts.
            # We must reconstruct the global line_counts or a subset.
            
            # Actually, the simplest robust way is to recompute the full line_counts for the new configuration?
            # That's O(N^2) = 120 ops.
            # Is O(N^2) fast enough for millions of iterations?
            # 120 ops * 1M = 120M ops. In Python, this is ~5-10 seconds.
            # So we can do ~200k-500k iterations in 150s.
            
            # Let's stick to full recount but optimize the function.
            pass
    
    # Revised Plan:
    # Just do full recount but make it as fast as possible.
    # The previous program did ~20k iterations.
    # If we can do 200k iterations, we have 10x more search power.
    
    # Let's implement a very tight loop for counting.
    # We will return the new score and validity.
    
    new_points_list = [p[:] for p in points]
    for i, idx in enumerate(idxs):
        new_points_list[idx] = new_pts[i]
        
    # Check validity and count
    line_counts = defaultdict(int)
    for i in range(n):
        x1, y1 = new_points_list[i]
        for j in range(i + 1, n):
            x2, y2 = new_points_list[j]
            A = y1 - y2
            B = x2 - x1
            C = x1 * y2 - x2 * y1
            
            if A == 0 and B == 0:
                continue
                
            g = math.gcd(math.gcd(abs(A), abs(B)), abs(C))
            if g > 0:
                A //= g
                B //= g
                C //= g
                
            if A < 0:
                A = -A
                B = -B
                C = -C
            elif A == 0:
                if B < 0:
                    B = -B
                    C = -C
                elif B == 0:
                    if C < 0:
                        C = -C
            
            key = (A, B, C)
            line_counts[key] += 1
            
    count_4 = 0
    is_invalid = False
    for count in line_counts.values():
        if count == 4:
            count_4 += 1
        elif count >= 5:
            is_invalid = True
            break
            
    return count_4, is_invalid

def _construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    random.seed(42)
    
    best_points = None
    best_score = 0
    
    # Initial configurations
    # 1. 4x4 Grid
    grid_4x4 = [[x, y] for x in range(4) for y in range(4)]
    
    # 2. Randomized Grids
    inits = [grid_4x4]
    for _ in range(20):
        pts = [p[:] for p in grid_4x4]
        for i in range(16):
            if random.random() < 0.3:
                pts[i][0] += random.randint(-2, 2)
                pts[i][1] += random.randint(-2, 2)
        # Ensure distinct
        unique = []
        for p in pts:
            if p not in unique:
                unique.append(p)
        if len(unique) == 16:
            inits.append(unique)
            
    # 3. Random points
    for _ in range(20):
        pts = []
        while len(pts) < 16:
            p = [random.randint(-10, 10), random.randint(-10, 10)]
            if p not in pts:
                pts.append(p)
        inits.append(pts)
        
    for init in inits:
        if time.time() > deadline:
            break
            
        current_points = [p[:] for p in init]
        current_score, is_invalid = _score_from_counts(_calculate_line_counts(current_points))
        if is_invalid:
            continue
            
        if current_score > best_score:
            best_score = current_score
            best_points = [p[:] for p in current_points]
            
        # Simulated Annealing
        T = 5.0
        T_min = 0.1
        cooling = 0.9999 # Very slow cooling
        
        steps = 50000
        for i in range(steps):
            if time.time() > deadline:
                break
                
            # Move: Pick 1 or 2 points to move
            num_moved = 1 if random.random() < 0.8 else 2
            idxs = random.sample(range(16), num_moved)
            
            new_coords = []
            for idx in idxs:
                ox, oy = current_points[idx]
                # Local move
                nx = ox + random.randint(-5, 5)
                ny = oy + random.randint(-5, 5)
                if abs(nx) > 1000 or abs(ny) > 1000:
                    continue
                new_coords.append([nx, ny])
                
            # Check duplicates
            valid_move = True
            new_set = set()
            for idx, np in zip(idxs, new_coords):
                if np in new_set:
                    valid_move = False
                    break
                new_set.add(tuple(np))
            
            # Check against non-moved points
            if valid_move:
                for k in range(16):
                    if k not in idxs:
                        if current_points[k] in new_set:
                            valid_move = False
                            break
            
            if not valid_move:
                continue
                
            new_score, is_invalid = _delta_delta_score(current_points, idxs, new_coords)
            
            if is_invalid:
                continue
                
            delta = new_score - current_score
            if delta > 0 or random.random() < math.exp(delta / T):
                for idx, np in zip(idxs, new_coords):
                    current_points[idx] = np
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_points = [p[:] for p in current_points]
                    
            T *= cooling
            if T < T_min:
                T = T_min
                
    if best_points is None:
        best_points = grid_4x4
        
    return best_points

def construct():
    return _construct()
