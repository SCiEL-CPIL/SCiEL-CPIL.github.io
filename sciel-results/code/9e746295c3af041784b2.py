import time
import random
from collections import defaultdict
from math import gcd

def construct():
    start_time = time.time()
    deadline = 140.0  # Leave 10s margin for 150s limit

    def canonical_line(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        dx = x2 - x1
        dy = y2 - y1
        g = gcd(abs(dx), abs(dy))
        if g == 0:
            return (0, 0, 0)
        dx //= g
        dy //= g
        if dx < 0:
            dx = -dx
            dy = -dy
        elif dx == 0 and dy < 0:
            dy = -dy
        a = dy
        b = -dx
        c = a * x1 + b * y1
        return (a, b, c)

    def full_score(points):
        """Count lines with exactly 4 points and lines with 5+ points."""
        line_counts = defaultdict(int)
        n = len(points)
        for i in range(n):
            for j in range(i + 1, n):
                key = canonical_line(points[i], points[j])
                line_counts[key] += 1
        num_4 = 0
        num_5_plus = 0
        for count in line_counts.values():
            # A line with k points has k*(k-1)/2 pairs
            # We need to map pair count to point count
            # 2 pts -> 1 pair
            # 3 pts -> 3 pairs
            # 4 pts -> 6 pairs
            # 5 pts -> 10 pairs
            # 6 pts -> 15 pairs
            if count == 6:
                num_4 += 1
            elif count >= 10:
                num_5_plus += 1
        return num_4, num_5_plus

    def evaluate_score(n4, n5):
        # Maximize n4, minimize n5
        return n4 - 10000 * n5

    def initial_grid():
        pts = []
        for i in range(4):
            for j in range(4):
                pts.append([i, j])
        return pts

    def random_config():
        """Generate a random configuration with no 5 collinear."""
        pts = []
        used = set()
        for _ in range(16):
            for _attempt in range(100):
                x = random.randint(-50, 50)
                y = random.randint(-50, 50)
                if (x, y) not in used:
                    # Check no 5 collinear with existing points
                    if not causes_5_collinear(pts, (x, y)):
                        pts.append([x, y])
                        used.add((x, y))
                        break
            else:
                # Fallback: just add a unique point
                while (x, y) in used:
                    x = random.randint(-50, 50)
                    y = random.randint(-50, 50)
                pts.append([x, y])
                used.add((x, y))
        return pts

    def causes_5_collinear(pts, new_pt):
        """Check if adding new_pt to pts would create 5 collinear."""
        n = len(pts)
        if n < 4:
            return False
        # For each pair of existing points, check if new_pt is on the line
        # and if there are already 3 other points on that line
        # Actually, simpler: for each line formed by 3 existing points, 
        # check if new_pt is on it.
        # But 3 points might not be collinear.
        # Efficient way: 
        # Count for each line through new_pt and an existing point, 
        # how many other points are on that line.
        line_counts = defaultdict(int)
        for p in pts:
            key = canonical_line(p, new_pt)
            line_counts[key] += 1
        for count in line_counts.values():
            # If count >= 4, then new_pt + 4 existing points are collinear -> 5 total
            if count >= 4:
                return True
        return False

    def local_search(initial_pts, max_iters, time_limit):
        points = [p[:] for p in initial_pts]
        n4, n5 = full_score(points)
        current_score = evaluate_score(n4, n5)
        best_score = current_score
        best_points = [p[:] for p in points]
        
        # Pre-compute lines for faster updates? 
        # For 16 points, full_score is O(16^2) = 256 ops. 
        # 100,000 iters = 25M ops. This is fine in Python if optimized.
        
        start_ls = time.time()
        for i in range(max_iters):
            if time.time() > time_limit:
                break
            
            # Choose move
            if random.random() < 0.5:
                # Random jitter
                idx = random.randint(0, 15)
                ox, oy = points[idx]
                step = random.randint(1, 5)
                nx = ox + random.randint(-step, step)
                ny = oy + random.randint(-step, step)
                if abs(nx) > 1000 or abs(ny) > 1000:
                    continue
                if any(points[k][0] == nx and points[k][1] == ny for k in range(16) if k != idx):
                    continue
                old_pos = points[idx][:]
                points[idx] = [nx, ny]
                new_n4, new_n5 = full_score(points)
                new_score = evaluate_score(new_n4, new_n5)
                if new_score > current_score:
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_points = [p[:] for p in points]
                else:
                    points[idx] = old_pos
            else:
                # Smart Move: Try to complete a 3-point line to 4
                # Find a line with exactly 3 points
                line_counts = defaultdict(list)
                n = len(points)
                for a in range(n):
                    for b in range(a + 1, n):
                        key = canonical_line(points[a], points[b])
                        line_counts[key].append((a, b))
                
                # Reconstruct point sets for lines with 3 points (6 pairs)
                # This is expensive. Let's just try random pairs and see if they form a 3-point line
                # Or better: iterate over all pairs, group by line key, check size.
                # We already have line_counts with pairs.
                # A line with 3 points has 3 pairs.
                # A line with 4 points has 6 pairs.
                
                # Let's pick a random pair (i, j)
                i, j = random.sample(range(16), 2)
                key = canonical_line(points[i], points[j])
                
                # Find all points on this line
                pts_on_line = [k for k in range(16) if canonical_line(points[i], points[k]) == key]
                # Note: canonical_line(p_i, p_k) == key implies collinear.
                # But we need to be careful with direction. 
                # canonical_line normalizes direction, so it's fine.
                
                if len(pts_on_line) == 3:
                    # We have 3 points on a line. Try to move a 4th point to this line.
                    # Pick a point NOT on the line
                    candidates = [k for k in range(16) if k not in pts_on_line]
                    if candidates:
                        idx = random.choice(candidates)
                        # Find a new position on the line
                        # Line equation: a*x + b*y = c
                        a, b, c = key
                        found_new_pos = False
                        for _ in range(10): # Try a few random x values
                            if b == 0:
                                # Vertical line x = const
                                nx = c # a=1, b=0 => x=c
                                ny = random.randint(-1000, 1000)
                                if abs(ny) <= 1000:
                                    # Check distinct
                                    if not any(points[k][0] == nx and points[k][1] == ny for k in range(16) if k != idx):
                                        old_pos = points[idx][:]
                                        points[idx] = [nx, ny]
                                        new_n4, new_n5 = full_score(points)
                                        new_score = evaluate_score(new_n4, new_n5)
                                        if new_score > current_score:
                                            current_score = new_score
                                            if current_score > best_score:
                                                best_score = current_score
                                                best_points = [p[:] for p in points]
                                            found_new_pos = True
                                            break
                                        else:
                                            points[idx] = old_pos
                            else:
                                # b*y = c - a*x => y = (c - a*x) / b
                                nx = random.randint(-1000, 1000)
                                val = c - a * nx
                                if val % b == 0:
                                    ny = val // b
                                    if abs(ny) <= 1000:
                                        # Check distinct
                                        if not any(points[k][0] == nx and points[k][1] == ny for k in range(16) if k != idx):
                                            old_pos = points[idx][:]
                                            points[idx] = [nx, ny]
                                            new_n4, new_n5 = full_score(points)
                                            new_score = evaluate_score(new_n4, new_n5)
                                            if new_score > current_score:
                                                current_score = new_score
                                                if current_score > best_score:
                                                    best_score = current_score
                                                    best_points = [p[:] for p in points]
                                                found_new_pos = True
                                                break
                                            else:
                                                points[idx] = old_pos

        return best_points, best_score

    # Start with the 4x4 grid
    init_pts = initial_grid()
    n4_init, n5_init = full_score(init_pts)
    best_score = evaluate_score(n4_init, n5_init)
    best_points = init_pts

    # Run local search from grid
    # We can run multiple times with different random seeds implicitly
    while time.time() < deadline - 10.0:
        # Try from grid
        bp, bs = local_search(init_pts, 50000, deadline - 5.0)
        if bs > best_score:
            best_score = bs
            best_points = bp
        
        # Try from random
        rand_pts = random_config()
        rn4, rn5 = full_score(rand_pts)
        rsc = evaluate_score(rn4, rn5)
        if rsc >= best_score - 10: # Only if promising
            bp, bs = local_search(rand_pts, 50000, deadline - 5.0)
            if bs > best_score:
                best_score = bs
                best_points = bp

    # Final validation
    fn4, fn5 = full_score(best_points)
    if fn5 > 0:
        # Fallback to grid if somehow invalid
        best_points = initial_grid()

    return best_points

if __name__ == "__main__":
    import time
    t0 = time.time()
    pts = construct()
    t1 = time.time()
    print(f"Time: {t1-t0:.1f}s")
    print(pts)
    print(len(pts))
    # Verify no 5 collinear
    from collections import defaultdict
    from math import gcd
    def _cl(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        dx = x2 - x1
        dy = y2 - y1
        g = gcd(abs(dx), abs(dy))
        if g == 0: return (0,0,0)
        dx //= g
        dy //= g
        if dx < 0:
            dx = -dx
            dy = -dy
        elif dx == 0 and dy < 0:
            dy = -dy
        a = dy
        b = -dx
        c = a * x1 + b * y1
        return (a, b, c)
    
    line_counts = defaultdict(int)
    for i in range(len(pts)):
        for j in range(i+1, len(pts)):
            line_counts[_cl(pts[i], pts[j])] += 1
            
    max_collinear = 0
    for k, v in line_counts.items():
        if v == 10: max_collinear = max(max_collinear, 5)
        elif v >= 15: max_collinear = max(max_collinear, 6)
        elif v == 6: max_collinear = max(max_collinear, 4)
        elif v == 3: max_collinear = max(max_collinear, 3)
        elif v == 1: max_collinear = max(max_collinear, 2)
        
    print("Max collinear:", max_collinear)
    count_4 = 0
    for k, v in line_counts.items():
        if v == 6:
            count_4 += 1
    print("4-lines:", count_4)
