import time
import random
from collections import defaultdict
from math import gcd

def construct():
    start_time = time.time()
    deadline = 140.0  # 10s margin

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

    def score(points):
        """
        Returns (n4, n5_plus).
        n4: number of lines with exactly 4 points.
        n5_plus: number of lines with >= 5 points.
        """
        line_counts = defaultdict(int)
        n = len(points)
        for i in range(n):
            for j in range(i + 1, n):
                key = canonical_line(points[i], points[j])
                line_counts[key] += 1
        
        num_4 = 0
        num_5_plus = 0
        for count in line_counts.values():
            # A line with k points has C(k,2) pairs.
            # If C(k,2) == 6, k=4.
            # If C(k,2) >= 10, k>=5.
            if count == 6:
                num_4 += 1
            elif count >= 10:
                num_5_plus += 1
        return num_4, num_5_plus

    def evaluate_score(n4, n5):
        return n4 - 10000 * n5

    def initial_grid():
        pts = []
        for i in range(4):
            for j in range(4):
                pts.append([i, j])
        return pts

    def random_valid_config():
        pts = []
        used = set()
        for _ in range(16):
            placed = False
            for _ in range(1000):
                x = random.randint(-20, 20)
                y = random.randint(-20, 20)
                if (x, y) not in used:
                    # Check no 5 collinear
                    valid = True
                    # Quick check: count collinear with existing
                    # This is O(N^2) per point, so O(N^3) total. N=16, so 4096 ops. Fast.
                    # But we can just check if adding it creates a 5-line.
                    # Simpler: just add it, and let the SA fix it? 
                    # No, we want a valid start.
                    # Let's just check max collinear count.
                    temp_pts = pts + [[x, y]]
                    n4, n5 = score(temp_pts)
                    if n5 == 0:
                        pts.append([x, y])
                        used.add((x, y))
                        placed = True
                        break
            if not placed:
                # Fallback: random unique
                while True:
                    x = random.randint(-100, 100)
                    y = random.randint(-100, 100)
                    if (x, y) not in used:
                        pts.append([x, y])
                        used.add((x, y))
                        break
        return pts

    def sa_search(initial_pts, max_iters, t_start, t_end):
        points = [p[:] for p in initial_pts]
        n4, n5 = score(points)
        current_score = evaluate_score(n4, n5)
        best_score = current_score
        best_points = [p[:] for p in points]

        iters = 0
        while iters < max_iters and time.time() < deadline:
            t = t_start * (t_end / t_start) ** (iters / max_iters)
            iters += 1

            # Move Selection
            # 1. Random move
            # 2. Heuristic move (complete a 3-line)
            
            move_type = random.random()
            
            if move_type < 0.7:
                # Random move
                idx = random.randint(0, 15)
                ox, oy = points[idx]
                step = random.randint(1, 5)
                nx = ox + random.randint(-step, step)
                ny = oy + random.randint(-step, step)
                
                # Bounds check
                if abs(nx) > 1000 or abs(ny) > 1000:
                    continue
                # Distinctness check
                if any(points[k][0] == nx and points[k][1] == ny for k in range(16) if k != idx):
                    continue

                points[idx] = [nx, ny]
                new_n4, new_n5 = score(points)
                new_score = evaluate_score(new_n4, new_n5)
                
                delta = new_score - current_score
                if delta > 0 or random.random() < (2.71828 ** (delta / t)):
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_points = [p[:] for p in points]
                else:
                    points[idx] = [ox, oy]

            else:
                # Heuristic: Find a line with 3 points and try to complete it
                # This requires finding 3-collinear sets.
                # To do this efficiently, we can look at the current line_counts.
                # But we don't store them. Let's compute them.
                # This is O(N^2).
                line_counts = defaultdict(list)
                for i in range(16):
                    for j in range(i + 1, 16):
                        key = canonical_line(points[i], points[j])
                        line_counts[key].append((i, j))
                
                # Find lines with exactly 3 points (6 pairs? No, 3 points -> 3 pairs)
                # Wait, C(3,2) = 3.
                # So we look for keys with count == 3.
                candidates = []
                for key, pairs in line_counts.items():
                    if len(pairs) == 3:
                        # Extract the 3 unique points
                        pts_on_line = set()
                        for i, j in pairs:
                            pts_on_line.add(i)
                            pts_on_line.add(j)
                        if len(pts_on_line) == 3:
                            candidates.append((key, list(pts_on_line)))
                
                if candidates:
                    key, pt_indices = random.choice(candidates)
                    # Generate a 4th point on this line
                    a, b, c = key
                    # Get one point to start from
                    p_idx = pt_indices[0]
                    x0, y0 = points[p_idx]
                    # Direction vector for integer points on ax+by=c
                    # x = x0 + b*t, y = y0 - a*t
                    # Try small t values
                    for t in range(-5, 6):
                        if t == 0: continue
                        nx = x0 + b * t
                        ny = y0 - a * t
                        if abs(nx) > 1000 or abs(ny) > 1000:
                            continue
                        # Check distinctness
                        if any(points[k][0] == nx and points[k][1] == ny for k in range(16)):
                            continue
                        
                        # Move a random point NOT on this line to (nx, ny)
                        # Points on line are pt_indices
                        available = [k for k in range(16) if k not in pt_indices]
                        if not available:
                            continue
                        move_idx = random.choice(available)
                        
                        old_pos = points[move_idx][:]
                        points[move_idx] = [nx, ny]
                        
                        new_n4, new_n5 = score(points)
                        new_score = evaluate_score(new_n4, new_n5)
                        
                        delta = new_score - current_score
                        if delta > 0 or random.random() < (2.71828 ** (delta / t)):
                            current_score = new_score
                            if current_score > best_score:
                                best_score = current_score
                                best_points = [p[:] for p in points]
                        else:
                            points[move_idx] = old_pos
                        break # Try only one t per heuristic move

        return best_points, best_score

    # Run search
    best_points = initial_grid()
    best_score = -1
    
    # Try 4x4 grid
    init_pts = initial_grid()
    n4, n5 = score(init_pts)
    sc = evaluate_score(n4, n5)
    if sc > best_score:
        best_score = sc
        best_points = init_pts
    
    # SA from grid
    bp, bs = sa_search(init_pts, 1000000, 10.0, 0.001)
    if bs > best_score:
        best_score = bs
        best_points = bp

    # Try random restarts if time permits
    while time.time() < deadline - 5.0:
        rand_pts = random_valid_config()
        rn4, rn5 = score(rand_pts)
        rsc = evaluate_score(rn4, rn5)
        if rsc >= best_score - 5:
            iters = min(500000, max(50000, int((deadline - time.time()) * 1000)))
            bp, bs = sa_search(rand_pts, iters, 5.0, 0.001)
            if bs > best_score:
                best_score = bs
                best_points = bp

    # Final validation
    fn4, fn5 = score(best_points)
    if fn5 > 0:
        best_points = initial_grid()

    return best_points

if __name__ == "__main__":
    t0 = time.time()
    pts = construct()
    t1 = time.time()
    print(f"Time: {t1-t0:.1f}s")
    print(pts)
    n4, n5 = score(pts)
    print(f"Score: {n4}, 5+ lines: {n5}")
