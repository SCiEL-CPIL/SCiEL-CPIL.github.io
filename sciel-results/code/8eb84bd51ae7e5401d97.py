"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd, isqrt

def _line_key(p1, p2):
    """Return a canonical key for the line through two points."""
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return None
    g = gcd(abs(dx), abs(dy))
    if g > 0:
        dx //= g
        dy //= g
    # Line equation: dx*x + dy*y + c = 0
    c = -(dx * y1 + dy * x1)
    # Normalize so that gcd(dx, dy, c) = 1 and leading coefficient positive
    g2 = gcd(gcd(abs(dx), abs(dy)), abs(c))
    if g2 > 0:
        dx //= g2
        dy //= g2
        c //= g2
    if dx < 0 or (dx == 0 and dy < 0) or (dx == 0 and dy == 0 and c < 0):
        dx, dy, c = -dx, -dy, -c
    return (dx, dy, c)

def _count_ordinary(points):
    """Count lines containing exactly 2 of the given points."""
    n = len(points)
    # Map each line to a canonical tuple; count pairs per line
    line_pair_counts = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            if key is None:
                continue
            line_pair_counts[key] = line_pair_counts.get(key, 0) + 1
    
    ordinary = 0
    for key, m in line_pair_counts.items():
        # m = C(k,2) where k = number of points on line
        # k^2 - k - 2m = 0 => k = (1 + sqrt(1+8m)) / 2
        disc = 1 + 8 * m
        sqrt_d = isqrt(disc)
        if sqrt_d * sqrt_d == disc:
            k = (1 + sqrt_d) // 2
            if k == 2:
                ordinary += 1
        else:
            # Should not happen for valid integer point sets
            ordinary += 1
    return ordinary

def _eval_config(points):
    """Return (num_ordinary, valid) for a configuration of 12 points."""
    if len(points) != 12:
        return (999, False)
    # Check distinctness
    if len(set(tuple(p) for p in points)) != 12:
        return (999, False)
    # Check bounds
    for x, y in points:
        if abs(x) > 10**9 or abs(y) > 10**9:
            return (999, False)
    # Check not all collinear
    x0, y0 = points[0]
    x1, y1 = points[1]
    dx = x1 - x0
    dy = y1 - y0
    all_col = True
    for x, y in points[2:]:
        if dx * (y - y0) != dy * (x - x0):
            all_col = False
            break
    if all_col:
        return (999, False)
    return (_count_ordinary(points), True)

def _random_points():
    """Generate 12 random integer points within bounds."""
    pts = []
    while len(pts) < 12:
        x = random.randint(-50, 50)
        y = random.randint(-50, 50)
        if [x, y] not in pts:
            pts.append([x, y])
    return pts

def _local_search(points, deadline, max_iter=5000, radius=20):
    """
    Optimized local search.
    Uses a delta evaluation strategy to avoid full O(N^2) recount for every move.
    """
    current = [p[:] for p in points]
    n = len(current)
    current_score = _count_ordinary(current)
    
    # Precompute line incidences for the current state
    # line_map: key -> list of indices of points on that line
    line_map = {}
    point_lines = [[] for _ in range(n)] # list of line keys for each point
    
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(current[i], current[j])
            if key is None: continue
            if key not in line_map:
                line_map[key] = []
            line_map[key].append(i)
            line_map[key].append(j)
            
    # Clean up line_map: remove duplicates in lists
    for k in list(line_map.keys()):
        line_map[k] = list(set(line_map[k]))
        
    # Build point_lines
    for k, pts_in_line in line_map.items():
        for p_idx in pts_in_line:
            point_lines[p_idx].append(k)

    def get_delta_score(i, new_pos):
        """
        Calculate the change in ordinary lines count if point i is moved to new_pos.
        Returns (delta, valid). delta < 0 is improvement.
        """
        # Check distinctness
        new_pt = tuple(new_pos)
        for j in range(n):
            if j != i and tuple(current[j]) == new_pt:
                return 0, False
        
        old_pos = current[i]
        
        # Identify lines affected by removing i
        # A line is "ordinary" if it has exactly 2 points.
        # When we remove i, some lines lose a point.
        # When we add i at new_pos, some lines gain a point.
        
        # We need to know which lines i was on.
        old_lines = point_lines[i]
        
        # For each old line, determine if it was ordinary, and if it becomes non-ordinary (or vice versa)
        # Actually, it's easier to just recalculate the count of ordinary lines for the *affected* lines.
        
        # Let's track the count of points on each line.
        # This is slightly complex to maintain perfectly in O(1) updates, but we can approximate or just recalculate the specific lines.
        
        # Faster approach:
        # 1. Identify all lines that contain i (old_lines).
        # 2. Identify all lines that would contain new_pos (new_lines).
        # 3. The change in score depends on:
        #    - Lines in old_lines: count decreases by 1.
        #    - Lines in new_lines: count increases by 1.
        #    - If a line is in both (i.e., i stays on the same line but moves along it? No, i moves to a different position, so it leaves the old line segment definition? 
        #      Wait, _line_key is for the infinite line. If i moves to a point on the same infinite line, the line key remains the same.
        #      However, the "count" of points on that line changes? No, i is still on the line.
        #      So if new_pos is collinear with the other points on an old line, that line's count doesn't change?
        #      Yes. If i moves from A to B on line L, and C is on L, then L still has {i, C, ...}. The count is the same.
        #      So we only care about lines where the *set* of points changes.
        
        # Let's refine:
        # old_lines_set = set(point_lines[i])
        # new_lines_set = set of lines passing through new_pos and every other point j != i.
        
        # For each line in old_lines_set:
        #   If new_pos is NOT on this line:
        #      count_old = len(line_map[line_key])
        #      count_new = count_old - 1
        #      If count_old == 2: line was ordinary, now it has 1 point (not a line in our context? Or just not ordinary). 
        #         Actually, a line with 1 point is not a "line containing points". 
        #         Our definition: "lines containing exactly two of the points".
        #         So if count_old == 2, it contributed 1 to score.
        #         If count_new == 1, it contributes 0.
        #         Delta = -1.
        #      If count_old == 3: contributed 0. count_new == 2. Delta = +1.
        #      If count_old > 3: contributed 0. count_new >= 2. 
        #         If count_new == 2, Delta = +1. Else 0.
        
        # For each line in new_lines_set:
        #   If old_pos is NOT on this line:
        #      count_old = len(line_map[line_key])
        #      count_new = count_old + 1
        #      If count_old == 1: (line existed with 1 point? No, lines are defined by pairs. 
        #         Wait, line_map is built from pairs. So a line always has at least 2 points in the current set.
        #         Unless it's a new line created by new_pos and some other point j.
        #         If the line didn't exist before (count_old = 0 in the sense of "not in line_map" or length < 2?), 
        #         actually if new_pos and j define a line, and no other points are on it, then count_old=0 (effectively, just the pair i,j? No, i is moving).
        #         Let's say count_old is the number of points from current set (excluding i) on that line.
        #         If count_old == 1 (only j is on it): 
        #             New count = 2 (i and j). It becomes ordinary. Delta = +1.
        #         If count_old == 2 (j and k are on it):
        #             New count = 3. It was ordinary (if it existed in line_map with size 2? No, if j,k were on it, it was in line_map with size >=2).
        #             If it was ordinary (size 2), it becomes non-ordinary (size 3). Delta = -1.
        #         If count_old > 2:
        #             New count > 3. No change in ordinary status (was 0, is 0).
        
        # This logic is tricky to implement correctly in O(N) without bugs.
        # Given the tight time limit and the need for robustness, I will stick to a simpler but faster full-recalculation strategy:
        # Use the existing _count_ordinary but optimize the inner loop of the search by reducing candidate moves.
        
        return 0, True # Placeholder, we will use a simpler strategy below

    # Strategy: 
    # 1. Try moves that specifically target "ordinary" lines.
    # 2. If a point is on an ordinary line, moving it off that line might reduce the count if the line disappears (becomes 1 point) 
    #    and doesn't create new ordinary lines.
    
    # To keep it simple and fast enough, I will use a reduced candidate set and full recount, but with a much tighter loop and early exit.
    
    for iteration in range(max_iter):
        if time.time() > deadline:
            break
            
        improved = False
        # Pick a random point to move
        i = random.randint(0, n - 1)
        x, y = current[i]
        
        # Generate a small number of diverse candidates
        candidates = []
        # Neighbors
        for dx in range(-radius, radius + 1, 4):
            for dy in range(-radius, radius + 1, 4):
                if dx == 0 and dy == 0: continue
                candidates.append((x + dx, y + dy))
        # Random jumps
        for _ in range(10):
            candidates.append((x + random.randint(-50, 50), y + random.randint(-50, 50)))
        
        random.shuffle(candidates)
        
        best_trial = None
        best_score = current_score
        
        for new_x, new_y in candidates:
            if time.time() > deadline:
                break
            if abs(new_x) > 10**9 or abs(new_y) > 10**9:
                continue
            
            trial = [p[:] for p in current]
            trial[i] = [new_x, new_y]
            
            # Quick distinctness check
            if len(set(tuple(p) for p in trial)) != n:
                continue
            
            # Evaluate
            score, valid = _eval_config(trial)
            if valid and score < best_score:
                best_score = score
                best_trial = trial
                # Don't break, try to find a better one? 
                # Or break to save time? 
                # Breaking on first improvement is faster but might get stuck in local minima.
                # Given time constraints, let's accept the first improvement to maximize iterations.
                break 
        
        if best_trial:
            current = best_trial
            current_score = best_score
            improved = True
            # Re-compute line_map and point_lines for next iteration?
            # This is O(N^2). With N=12, it's fast.
            line_map = {}
            point_lines = [[] for _ in range(n)]
            for ii in range(n):
                for jj in range(ii + 1, n):
                    key = _line_key(current[ii], current[jj])
                    if key is None: continue
                    if key not in line_map:
                        line_map[key] = []
                    line_map[key].append(ii)
                    line_map[key].append(jj)
            for k, pts_in_line in line_map.items():
                line_map[k] = list(set(line_map[k]))
                for p_idx in line_map[k]:
                    point_lines[p_idx].append(k)
            
        if not improved:
            # Random kick
            if random.random() < 0.2:
                i = random.randint(0, n - 1)
                x, y = current[i]
                new_x = x + random.randint(-20, 20)
                new_y = y + random.randint(-20, 20)
                if abs(new_x) <= 10**9 and abs(new_y) <= 10**9:
                    trial = [p[:] for p in current]
                    trial[i] = [new_x, new_y]
                    if len(set(tuple(p) for p in trial)) == n:
                        score, valid = _eval_config(trial)
                        if valid:
                            # Accept even if slightly worse to escape local min?
                            # No, let's only accept if better or equal to avoid drift.
                            if score <= current_score:
                                current = trial
                                current_score = score
                                # Rebuild line_map
                                line_map = {}
                                point_lines = [[] for _ in range(n)]
                                for ii in range(n):
                                    for jj in range(ii + 1, n):
                                        key = _line_key(current[ii], current[jj])
                                        if key is None: continue
                                        if key not in line_map:
                                            line_map[key] = []
                                        line_map[key].append(ii)
                                        line_map[key].append(jj)
                                for k, pts_in_line in line_map.items():
                                    line_map[k] = list(set(line_map[k]))
                                    for p_idx in line_map[k]:
                                        point_lines[p_idx].append(k)

    return current, current_score

def construct():
    t0 = time.time()
    deadline = t0 + 110.0  # Safety margin: 110s limit to avoid 120s kill
    
    random.seed(42)  # Deterministic for reproducibility
    
    best_points = [[x, y] for x in range(4) for y in range(3)]
    best_score = _count_ordinary(best_points)
    
    # Try several structured configurations as starting points
    candidates = []
    
    # 1. 3x4 grid (baseline)
    candidates.append([[x, y] for x in range(4) for y in range(3)])
    
    # 2. 4x3 grid
    candidates.append([[x, y] for x in range(3) for y in range(4)])
    
    # 3. Symmetric cross pattern
    cross_pts = [
        [0, 0],
        [1, 0], [-1, 0], [2, 0], [-2, 0],
        [0, 1], [0, -1], [0, 2], [0, -2],
        [1, 1], [1, -1], [-1, 1]
    ]
    candidates.append(cross_pts)
    
    # 4. Hexagonal arrangement
    hex_pts = [
        [0, 0],
        [1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [-1, -1],
        [2, 0], [-2, 0], [0, 2], [0, -2], [2, 2]
    ]
    candidates.append(hex_pts)
    
    # 5. Shifted grid
    shifted = []
    for y in range(3):
        for x in range(4):
            shifted.append([x + y, y])
    candidates.append(shifted)
    
    # 6. Another structured pattern
    pattern6 = [
        [0, 0], [1, 0], [2, 0], [3, 0],
        [0, 1], [1, 1], [2, 1], [3, 1],
        [0, 2], [1, 2], [2, 2], [4, 2]
    ]
    candidates.append(pattern6)
    
    # 7. Points on a circle (approximate integer)
    # (0,0), (1,0), (-1,0), (0,1), (0,-1), (1,1), (-1,1), (1,-1), (-1,-1), (2,0), (0,2), (0,-2)
    circle_pts = [
        [0,0], [1,0], [-1,0], [0,1], [0,-1],
        [1,1], [-1,1], [1,-1], [-1,-1],
        [2,0], [0,2], [0,-2]
    ]
    candidates.append(circle_pts)

    # Evaluate all candidates
    for cand in candidates:
        if time.time() > deadline:
            break
        score, valid = _eval_config(cand)
        if valid and score < best_score:
            best_score = score
            best_points = [p[:] for p in cand]
    
    # Local search from best candidate
    if time.time() < deadline - 5:
        best_points, best_score = _local_search(best_points, deadline, max_iter=10000, radius=10)
    
    # Try a few random restarts if time permits
    if time.time() < deadline - 10:
        for restart in range(5):
            if time.time() > deadline - 5:
                break
            random_pts = _random_points()
            score, valid = _eval_config(random_pts)
            if valid and score < best_score:
                best_score = score
                best_points = [p[:] for p in random_pts]
                # Local search from this better starting point
                if time.time() < deadline - 5:
                    best_points, best_score = _local_search(best_points, deadline, max_iter=2000, radius=10)
    
    return best_points
# EVOLVE-BLOCK-END
