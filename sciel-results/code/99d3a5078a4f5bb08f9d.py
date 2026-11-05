import time
import random
from math import gcd

_START = time.time()
_DEADLINE = _START + 140.0  # safety margin before 150s kill

def _count_lines(points):
    """Count lines by number of points. Returns (ordinary_count, triple_count, all_counts)."""
    n = len(points)
    if n < 2:
        return 0, 0, {}
    line_counts = {}
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            A = y1 - y2
            B = x2 - x1
            C = x1 * y2 - x2 * y1
            g = gcd(gcd(abs(A), abs(B)), abs(C)) if (A or B or C) else 1
            if g:
                A //= g
                B //= g
                C //= g
            if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
                A, B, C = -A, -B, -C
            key = (A, B, C)
            line_counts[key] = line_counts.get(key, 0) + 1
    ordinary = sum(1 for c in line_counts.values() if c == 1)
    triples = sum(1 for c in line_counts.values() if c == 3)
    return ordinary, triples, line_counts

def _objective(points):
    """Objective: Minimize ordinary lines, secondary: Maximize 3-point lines."""
    ord_lines, tri_lines, _ = _count_lines(points)
    return ord_lines * 100 - tri_lines

def _valid(points):
    """12 distinct integer points, |coord|<=1e9, not all collinear."""
    if len(points) != 12:
        return False
    seen = set()
    for x, y in points:
        if abs(x) > 10**9 or abs(y) > 10**9:
            return False
        t = (x, y)
        if t in seen:
            return False
        seen.add(t)
    # Check not all collinear
    if len(points) >= 3:
        x0, y0 = points[0]
        x1, y1 = points[1]
        x2, y2 = points[2]
        if (x1 - x0) * (y2 - y0) == (y1 - y0) * (x2 - x0):
            # Check if all others are on this line
            all_collinear = True
            for p in points[3:]:
                if (p[0] - x0) * (y1 - y0) != (p[1] - y0) * (x1 - x0):
                    all_collinear = False
                    break
            if all_collinear:
                return False
    return True

def _random_point():
    return [random.randint(-50, 50), random.randint(-50, 50)]

def _make_seed_grid():
    """3x4 grid seed."""
    points = []
    for y in range(4):
        for x in range(3):
            points.append([x, y])
    return points

def _make_seed_hexagonal():
    """Hexagonal lattice seed - 12 points in hexagonal arrangement."""
    points = []
    # Center point
    points.append([0, 0])
    # 6 points on inner hexagon (radius 1)
    for angle_deg in range(0, 360, 60):
        import math
        x = round(math.cos(math.radians(angle_deg)))
        y = round(math.sin(math.radians(angle_deg)))
        points.append([x, y])
    # 5 more points on outer hexagon to make 12
    # Use specific positions to avoid duplicates
    extra = [[2, 0], [1, 1], [-1, 1], [-2, 0], [0, 2]]
    for p in extra:
        if p not in points:
            points.append(p)
    # Ensure we have exactly 12 distinct points
    unique = []
    for p in points:
        if p not in unique:
            unique.append(p)
    while len(unique) < 12:
        unique.append([random.randint(-3, 3), random.randint(-3, 3)])
    return unique[:12]

def _make_seed_boroczky():
    """Böröczky-type configuration: 9 points on conic + 3 at infinity approximated."""
    # Use 9 points on a parabola y = x^2
    points = []
    for i in range(-4, 5):
        points.append([i, i * i])
    # Add 3 more points strategically
    # Points that create additional collinearities
    points.append([0, 0])  # Already there, need different
    # Let's use a different approach: 12 points with known structure
    # Use 4x3 grid but perturb
    pts = []
    for y in range(3):
        for x in range(4):
            pts.append([x, y])
    return pts

def _get_lines_with_counts(points):
    """Get all lines with their point counts."""
    n = len(points)
    line_points = {}
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            A = y1 - y2
            B = x2 - x1
            C = x1 * y2 - x2 * y1
            g = gcd(gcd(abs(A), abs(B)), abs(C)) if (A or B or C) else 1
            if g:
                A //= g
                B //= g
                C //= g
            if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
                A, B, C = -A, -B, -C
            key = (A, B, C)
            if key not in line_points:
                line_points[key] = set()
            line_points[key].add(i)
    return line_points

def _find_line_completion_moves(points, max_moves=20):
    """Find moves that place a point on a line with 2+ other points to create 3-point lines."""
    line_points = _get_lines_with_counts(points)
    n = len(points)
    moves = []
    
    # For each line with exactly 2 points, find where to place a third point
    for line_key, pt_set in line_points.items():
        if len(pt_set) == 2:
            i, j = list(pt_set)
            x1, y1 = points[i]
            x2, y2 = points[j]
            # The line is determined by these two points
            # We can try to move another point onto this line
            for k in range(n):
                if k == i or k == j:
                    continue
                # Check if moving point k onto line (i,j) would be valid
                # The line through (x1,y1) and (x2,y2)
                # Parametric: (x1 + t*(x2-x1), y1 + t*(y2-y1)) for t != 0,1
                # We need integer coordinates
                dx = x2 - x1
                dy = y2 - y1
                g = gcd(abs(dx), abs(dy)) if (dx or dy) else 1
                if g == 0:
                    continue
                dx_reduced = dx // g
                dy_reduced = dy // g
                # Try small t values
                for t in range(-5, 6):
                    if t == 0 or t == 1:
                        continue
                    nx = x1 + t * dx_reduced
                    ny = y1 + t * dy_reduced
                    if abs(nx) <= 10**9 and abs(ny) <= 10**9:
                        # Check if this position is already taken
                        new_pos = (nx, ny)
                        taken = False
                        for m in range(n):
                            if (points[m][0], points[m][1]) == new_pos:
                                taken = True
                                break
                        if not taken:
                            moves.append((k, [nx, ny], line_key))
    
    # Also consider moving points to create 3-point lines from scratch
    # For pairs of points, find a third point that makes a nice line
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            g = gcd(abs(dx), abs(dy)) if (dx or dy) else 1
            if g == 0:
                continue
            dx_reduced = dx // g
            dy_reduced = dy // g
            for t in range(-3, 4):
                if t == 0 or t == 1:
                    continue
                nx = x1 + t * dx_reduced
                ny = y1 + t * dy_reduced
                if abs(nx) <= 10**9 and abs(ny) <= 10**9:
                    new_pos = (nx, ny)
                    taken = False
                    for m in range(n):
                        if (points[m][0], points[m][1]) == new_pos:
                            taken = True
                            break
                    if not taken:
                        moves.append((random.randrange(n), [nx, ny], None))
    
    return moves[:max_moves]

def _local_search_line_completion(points, max_iters=1000, time_limit=10.0):
    """Local search that specifically targets line completion."""
    cur = [p[:] for p in points]
    cur_obj = _objective(cur)
    best = [p[:] for p in cur]
    best_obj = cur_obj
    
    start_time = time.time()
    rng = random.Random(int(time.time() * 1000) % 2**32)
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        # Try line completion moves
        moves = _find_line_completion_moves(cur, max_moves=10)
        
        if moves:
            # Pick a random move
            k, new_pos, _ = rng.choice(moves)
            new_pts = [p[:] for p in cur]
            new_pts[k] = new_pos[:]
            
            if _valid(new_pts):
                new_obj = _objective(new_pts)
                if new_obj < cur_obj:
                    cur = new_pts
                    cur_obj = new_obj
                    if cur_obj < best_obj:
                        best_obj = cur_obj
                        best = [p[:] for p in cur]
                continue
        
        # Fallback: random move
        idx = rng.randrange(12)
        new_pts = [p[:] for p in cur]
        new_pts[idx][0] += rng.randint(-3, 3)
        new_pts[idx][1] += rng.randint(-3, 3)
        
        if _valid(new_pts):
            new_obj = _objective(new_pts)
            if new_obj < cur_obj:
                cur = new_pts
                cur_obj = new_obj
                if cur_obj < best_obj:
                    best_obj = cur_obj
                    best = [p[:] for p in cur]
    
    return best, best_obj

def _simulate_annealing(seed_points, max_time=10.0, rng=None):
    if rng is None:
        rng = random.Random(int(time.time() * 1000) % 2**32)
    cur = [p[:] for p in seed_points]
    if not _valid(cur):
        return None, float('inf')
    
    cur_obj = _objective(cur)
    best = [p[:] for p in cur]
    best_obj = cur_obj
    
    T = 1000.0
    T_min = 0.1
    cooling_rate = 0.999
    
    start_time = time.time()
    
    while T > T_min and time.time() - start_time < max_time:
        new_pts = [p[:] for p in cur]
        move_type = rng.random()
        
        if move_type < 0.4:
            # Move one point
            idx = rng.randrange(12)
            new_pts[idx][0] += rng.randint(-5, 5)
            new_pts[idx][1] += rng.randint(-5, 5)
        elif move_type < 0.7:
            # Move two points (to maintain/create collinearity)
            i = rng.randrange(12)
            j = rng.randrange(12)
            if i != j:
                new_pts[i][0] += rng.randint(-5, 5)
                new_pts[i][1] += rng.randint(-5, 5)
                new_pts[j][0] += rng.randint(-5, 5)
                new_pts[j][1] += rng.randint(-5, 5)
        else:
            # Large jump
            idx = rng.randrange(12)
            new_pts[idx] = _random_point()
            
        if not _valid(new_pts):
            T *= cooling_rate
            continue
            
        new_obj = _objective(new_pts)
        delta = new_obj - cur_obj
        
        if delta < 0 or rng.random() < pow(2.71828, -delta / T):
            cur = new_pts
            cur_obj = new_obj
            if cur_obj < best_obj:
                best_obj = cur_obj
                best = [p[:] for p in cur]
                
        T *= cooling_rate
        
    return best, best_obj

def construct():
    best_pts = _make_seed_grid()
    best_score = _objective(best_pts)
    
    # Create diverse seeds
    seeds = [_make_seed_grid(), _make_seed_hexagonal(), _make_seed_boroczky()]
    for _ in range(3):
        pts = [_random_point() for _ in range(12)]
        if _valid(pts):
            seeds.append(pts)
            
    # Phase 1: Line completion local search on each seed
    for seed in seeds:
        if time.time() > _DEADLINE - 30:
            break
        pts, score = _local_search_line_completion(seed, max_iters=500, time_limit=8.0)
        if pts and score < best_score:
            best_score = score
            best_pts = pts
    
    # Phase 2: Simulated annealing refinement on best and top seeds
    time_remaining = _DEADLINE - time.time()
    if time_remaining > 10:
        # Refine the best solution
        pts, score = _simulate_annealing(best_pts, max_time=min(time_remaining - 5, 20.0))
        if pts and score < best_score:
            best_score = score
            best_pts = pts
        
        # Also try refining other good seeds
        for seed in seeds[:2]:
            if time.time() > _DEADLINE - 10:
                break
            pts, score = _simulate_annealing(seed, max_time=min(_DEADLINE - time.time() - 5, 15.0))
            if pts and score < best_score:
                best_score = score
                best_pts = pts
    
    if best_pts is None:
        best_pts = _make_seed_grid()
        
    return best_pts
