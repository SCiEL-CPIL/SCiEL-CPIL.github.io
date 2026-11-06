import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin for safety

    # Precompute all lines with >= 3 integer points in [-100,100]^2
    # A line is determined by a normalized direction (dx, dy) and a "base point"
    # For efficiency, we only need lines that can contribute 3+ points to our 20-point set
    # 
    # Strategy: For each pair of distinct integer points (p1, p2) in the box,
    # compute the line through them, and collect all integer points on that line within the box.
    # But 201*201/2 pairs is ~20000, and for each we'd scan... that's too slow.
    #
    # Better: Enumerate lines by (slope, intercept) in a smart way.
    # A line y = mx + b with rational slope m = p/q (in lowest terms) contains integer points
    # when b is chosen appropriately.
    # 
    # Alternative efficient approach: For each pair of points, compute the line, but only
    # keep lines that have >= 3 integer points in the box. Use a dict keyed by normalized line.
    #
    # Actually, the most efficient approach for our purpose:
    # We want to find a set of lines L1, L2, ..., Lk such that:
    #   - Each Li has exactly 3 or 4 integer points in our final set
    #   - The union of all points on all Li is exactly 20 distinct points
    #   - No point is on more than 2 lines (to keep structure manageable)
    #   - No 4 points are collinear (so each line has at most 3 of our points... wait, 
    #     the constraint is NO FOUR collinear, so a line can have 3 of our points)
    #
    # Key insight: If we can partition 20 points into lines of 3 (and some of 2 or 4),
    # with each line having exactly 3 points from our set, then every triple on that line
    # is a collinear triple. The "no 3 collinear" maximum subset would need to pick at most
    # 2 from each 3-point line.
    #
    # If we have k lines of 3 points, disjoint, covering 3k points (plus 20-3k singleton points),
    # the max no-3-collinear subset is at most 2k + (20-3k) = 20 - k.
    # To minimize this, maximize k. Max k = 6 (18 points on 6 lines of 3, 2 singletons) -> score <= 14.
    # But lines can share points! If two lines share a point, it's more complex.
    #
    # Even better: if we can arrange that the 20 points lie on as few lines as possible,
    # with each line containing 3-4 of our points, the no-3-collinear number is bounded.
    #
    # Let me try a different concrete approach:
    # 1. Generate many random "line configurations" - sets of lines where each line has 3+ integer pts
    # 2. For each configuration, try to find 20 points that lie on these lines
    # 3. Evaluate and keep best

    # Precompute: for each normalized direction (dx, dy) with gcd(dx,dy)=1, 
    # and for each possible "offset", find all integer points on that line in the box.
    # This is still large. Let's be smarter.

    # Approach: Randomly select pairs of points, form lines, collect points on those lines.
    # Then do a set-cover-like selection.

    def get_points_on_line(p1, p2):
        """Get all integer points on the line through p1 and p2 within [-100,100]^2."""
        x1, y1 = p1
        x2, y2 = p2
        dx = x2 - x1
        dy = y2 - y1
        
        # Normalize direction
        if dx == 0 and dy == 0:
            return set()
        g = abs(dx)
        h = abs(dy)
        while g != 0 and h != 0:
            g, h = h, g % h
        if g == 0:
            g = 1
        dx_n = dx // g
        dy_n = dy // g
        
        # Parametrize: p = p1 + t*(dx_n, dy_n) for integer t
        # Find range of t such that -100 <= x1 + t*dx_n <= 100 and -100 <= y1 + t*dy_n <= 100
        t_min = -10**9
        t_max = 10**9
        
        if dx_n > 0:
            t_min = max(t_min, (-100 - x1) // dx_n if (-100 - x1) % dx_n == 0 else (-100 - x1) // dx_n + 1)
            # Actually let's just be safe:
            t_min = max(t_min, ceil_div(-100 - x1, dx_n))
            t_max = min(t_max, floor_div(100 - x1, dx_n))
        elif dx_n < 0:
            t_min = max(t_min, ceil_div(-100 - x1, dx_n))
            t_max = min(t_max, floor_div(100 - x1, dx_n))
        else:
            pass  # vertical, dx_n = 0
        
        if dy_n > 0:
            t_min = max(t_min, ceil_div(-100 - y1, dy_n))
            t_max = min(t_max, floor_div(100 - y1, dy_n))
        elif dy_n < 0:
            t_min = max(t_min, ceil_div(-100 - y1, dy_n))
            t_max = min(t_max, floor_div(100 - y1, dy_n))
        
        points = set()
        for t in range(t_min, t_max + 1):
            x = x1 + t * dx_n
            y = y1 + t * dy_n
            if -100 <= x <= 100 and -100 <= y <= 100:
                points.add((x, y))
        return points

    def ceil_div(a, b):
        if b > 0:
            return (a + b - 1) // b if a >= 0 else -((-a) // b)
        else:
            return -((-a + (-b) - 1) // (-b)) if a >= 0 else (a // (-b))

    def floor_div(a, b):
        if b > 0:
            return a // b if a >= 0 else -((-a + b - 1) // b)
        else:
            return -((-a) // (-b)) if a >= 0 else (a // (-b))

    # Simpler: just iterate t in a reasonable range
    def get_points_on_line_simple(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        dx = x2 - x1
        dy = y2 - y1
        
        if dx == 0 and dy == 0:
            return set()
        
        g = abs(dx)
        h = abs(dy)
        while g != 0 and h != 0:
            g, h = h, g % h
        if g == 0:
            g = 1
        dx_n = dx // g
        dy_n = dy // g
        
        # t range: we need -100 <= x1 + t*dx_n <= 100 and -100 <= y1 + t*dy_n <= 100
        # For safety, t in [-200, 200] is more than enough since |dx_n|,|dy_n| >= 1
        points = set()
        for t in range(-250, 251):
            x = x1 + t * dx_n
            y = y1 + t * dy_n
            if -100 <= x <= 100 and -100 <= y <= 100:
                points.add((x, y))
        return points

    def has_4_collinear_fast(pts_set):
        """Check if any 4 points in pts_set are collinear. pts_set is a set of tuples."""
        pts = list(pts_set)
        n = len(pts)
        for i in range(n):
            line_counts = defaultdict(int)
            for j in range(n):
                if i == j:
                    continue
                dx = pts[j][0] - pts[i][0]
                dy = pts[j][1] - pts[i][1]
                if dx == 0 and dy == 0:
                    continue
                if dx < 0 or (dx == 0 and dy < 0):
                    dx, dy = -dx, -dy
                g = abs(dx)
                h = abs(dy)
                while g != 0 and h != 0:
                    g, h = h, g % h
                if g == 0:
                    g = 1
                dx, dy = dx // g, dy // g
                line_counts[(dx, dy)] += 1
            if any(c >= 3 for c in line_counts.values()):
                return True
        return False

    def max_no_3_collinear(points):
        """Exact branch and bound for largest subset with no 3 collinear."""
        n = len(points)
        
        # Precompute collinear triples
        collinear_triples = []
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    p1, p2, p3 = points[i], points[j], points[k]
                    if (p2[0] - p1[0]) * (p3[1] - p1[1]) == (p3[0] - p1[0]) * (p2[1] - p1[1]):
                        collinear_triples.append((i, j, k))
        
        # For each point, precompute which pairs with it are collinear
        # bad[i] = set of (j, k) such that i, j, k are collinear, j < k, and j,k != i
        bad = [set() for _ in range(n)]
        for (i, j, k) in collinear_triples:
            bad[i].add((j, k))
            bad[j].add((i, k))
            bad[k].add((i, j))
        
        best = 0
        
        def backtrack(mask, start):
            nonlocal best
            current = bin(mask).count('1')
            remaining = n - start
            if current + remaining <= best:
                return
            if start == n:
                if current > best:
                    best = current
                return
            
            # Try including start
            valid = True
            for (a, b) in bad[start]:
                if a < start and b < start:
                    if (mask & (1 << a)) and (mask & (1 << b)):
                        valid = False
                        break
            
            if valid:
                backtrack(mask | (1 << start), start + 1)
            
            # Try excluding start
            backtrack(mask, start + 1)
        
        backtrack(0, 0)
        return best

    def count_collinear_triples(pts_list):
        """Count number of collinear triples in a list of points."""
        n = len(pts_list)
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    p1, p2, p3 = pts_list[i], pts_list[j], pts_list[k]
                    if (p2[0] - p1[0]) * (p3[1] - p1[1]) == (p3[0] - p1[0]) * (p2[1] - p1[1]):
                        count += 1
        return count

    # Main strategy: Construct configurations by selecting lines and their points
    best_config = None
    best_score = 21  # Initialize to worst possible

    # Precompute a pool of useful lines (with 3-6 integer points in box)
    # We'll sample random pairs of points and compute their lines
    line_pool = {}  # key: frozenset of points on line, value: the line's point set
    
    # Sample many random pairs to build a diverse line pool
    num_line_samples = 2000
    sampled_points = set()
    for _ in range(num_line_samples):
        p1 = (random.randint(-100, 100), random.randint(-100, 100))
        p2 = (random.randint(-100, 100), random.randint(-100, 100))
        if p1 == p2:
            continue
        pts_on_line = get_points_on_line_simple(p1, p2)
        if 3 <= len(pts_on_line) <= 10:  # Only keep lines with 3-10 points (manageable)
            key = frozenset(pts_on_line)
            if key not in line_pool:
                line_pool[key] = pts_on_line
    
    # Also add some structured lines: horizontal, vertical, diagonal
    for x in range(-100, 101, 7):
        pts = set((x, y) for y in range(-100, 101))
        if 3 <= len(pts) <= 10:
            line_pool[frozenset(pts)] = pts
    for y in range(-100, 101, 7):
        pts = set((x, y) for x in range(-100, 101))
        if 3 <= len(pts) <= 10:
            line_pool[frozenset(pts)] = pts
    # Diagonal lines with small slopes
    for slope_num in [1, -1, 2, -2, 3, -3]:
        for slope_den in [1, 2, 3]:
            if slope_num == 0:
                continue
            for b in range(-100, 101, 5):
                # y = (slope_num/slope_den) * x + b
                pts = set()
                for x in range(-100, 101):
                    num = slope_num * x + slope_den * b
                    if num % slope_den == 0:
                        y = num // slope_den
                        if -100 <= y <= 100:
                            pts.add((x, y))
                if 3 <= len(pts) <= 10:
                    line_pool[frozenset(pts)] = pts

    line_list = list(line_pool.values())
    random.shuffle(line_list)

    # Strategy: Greedy set-cover with backtracking
    # Select lines one by one. Each line contributes its points to our set.
    # We want exactly 20 points total.
    # Score: maximize collinear triples (which should minimize no-3-collinear number)
    
    def greedy_line_cover():
        """Select lines to cover exactly 20 points, maximizing collinear triples."""
        selected_lines = []
        covered = set()
        
        for line in line_list:
            if time.time() > deadline - 5:
                break
            if len(covered) >= 20:
                break
            
            # How many new points does this line add?
            new_points = line - covered
            if len(new_points) == 0:
                continue
            
            # Adding this line would give us len(covered) + len(new_points) points
            # We need exactly 20, so we can only add lines that don't overshoot
            # (or we can trim later)
            if len(covered) + len(new_points) <= 20:
                # Check: does adding this line create 4 collinear points?
                test_set = covered | line
                if len(test_set) <= 20:
                    # Quick check: for the new points, see if any 4 are collinear
                    # This is expensive, so only do it for small sets
                    if len(test_set) <= 15:
                        if has_4_collinear_fast(test_set):
                            continue
                    selected_lines.append(line)
                    covered = covered | line
        
        # Now we have 'covered' set, need to pad to 20 with random points
        if len(covered) < 20:
            all_box_points = [(x, y) for x in range(-100, 101) for y in range(-100, 101)]
            random.shuffle(all_box_points)
            for p in all_box_points:
                if len(covered) >= 20:
                    break
                if p not in covered:
                    test_set = covered | {p}
                    if len(test_set) <= 20:
                        if has_4_collinear_fast(test_set):
                            continue
                    covered.add(p)
        
        return list(covered)

    # Run multiple iterations with different random seeds/orderings
    num_iterations = 50
    for it in range(num_iterations):
        if time.time() > deadline - 15:
            break
        
        # Shuffle line list for diversity
        random.shuffle(line_list)
        
        pts = greedy_line_cover()
        
        if len(pts) != 20:
            continue
        
        # Validate
        pts_set = set(pts)
        if len(pts_set) != 20:
            continue
        if any(abs(p[0]) > 100 or abs(p[1]) > 100 for p in pts):
            continue
        if has_4_collinear_fast(pts_set):
            continue
        
        # Evaluate
        score = max_no_3_collinear(pts)
        
        if score < best_score:
            best_score = score
            best_config = [list(p) for p in pts]
        
        # Local improvement: try swapping individual points
        if best_config is not None and time.time() < deadline - 10:
            improved = [list(p) for p in best_config]
            improved_set = set(map(tuple, improved))
            
            for step in range(200):
                if time.time() > deadline - 3:
                    break
                
                idx = random.randint(0, 19)
                old_p = tuple(improved[idx])
                new_x = random.randint(-100, 100)
                new_y = random.randint(-100, 100)
                new_p = (new_x, new_y)
                
                if new_p == old_p or new_p in improved_set:
                    continue
                
                # Try the swap
                improved[idx] = [new_x, new_y]
                new_set = set(map(tuple, improved))
                
                if len(new_set) != 20 or has_4_collinear_fast(new_set):
                    improved[idx] = [old_p[0], old_p[1]]
                    continue
                
                new_score = max_no_3_collinear(improved)
                if new_score < best_score:
                    best_score = new_score
                    best_config = [list(p) for p in improved]
                    improved_set = new_set
                else:
                    improved[idx] = [old_p[0], old_p[1]]

    # Fallback if nothing found
    if best_config is None:
        # Construct a simple valid configuration
        # Use points on y = x^2 mod something, but ensure no 4 collinear
        # Simple: use a perturbed grid
        fallback = []
        used = set()
        for i in range(20):
            while True:
                x = random.randint(-100, 100)
                y = random.randint(-100, 100)
                if (x, y) not in used:
                    used.add((x, y))
                    fallback.append([x, y])
                    break
        # Verify no 4 collinear (should be fine for random)
        if has_4_collinear_fast(set(map(tuple, fallback))):
            # Try again with different points
            for _ in range(10):
                fallback = []
                used = set()
                for i in range(20):
                    while True:
                        x = random.randint(-100, 100)
                        y = random.randint(-100, 100)
                        if (x, y) not in used:
                            used.add((x, y))
                            fallback.append([x, y])
                            break
                if not has_4_collinear_fast(set(map(tuple, fallback))):
                    break
        best_config = fallback

    return best_config
