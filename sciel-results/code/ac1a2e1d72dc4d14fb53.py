"""Direct construction: Simulated Annealing for 12-point configuration with few ordinary lines.

Uses Simulated Annealing to search for integer coordinates that minimize the number of
ordinary lines. The objective function penalizes ordinary lines and rewards 3-point lines.
"""
import time
import random
from math import gcd

_START = time.time()
_DEADLINE = _START + 145.0  # safety margin before 150s kill


def _count_ordinary(points):
    """Count lines containing exactly 2 of the given points (n=12)."""
    n = len(points)
    if n < 2:
        return 0
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
    return sum(1 for c in line_counts.values() if c == 1)

def _count_3point_lines(points):
    """Count lines containing exactly 3 of the given points (n=12)."""
    n = len(points)
    if n < 3:
        return 0
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
    return sum(1 for c in line_counts.values() if c == 3)

def _objective(points):
    """Objective: Minimize ordinary lines, secondary: Maximize 3-point lines."""
    ord_lines = _count_ordinary(points)
    tri_lines = _count_3point_lines(points)
    return ord_lines * 100 - tri_lines


def _collinear_check(points):
    """Check if all points are collinear."""
    if len(points) < 3:
        return True
    x0, y0 = points[0]
    x1, y1 = points[1]
    for p in points[2:]:
        if (p[0] - x0) * (y1 - y0) != (p[1] - y0) * (x1 - x0):
            return False
    return True


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
    return not _collinear_check(points)


def _random_point():
    """Generate a random valid integer point."""
    return [random.randint(-1000, 1000), random.randint(-1000, 1000)]


def _make_seed_grid():
    """Create a 4x3 grid seed."""
    points = []
    for i in range(4):
        for j in range(3):
            points.append([i, j])
    return points


def _simulate_annealing(seed_points, max_time=10.0):
    """Run Simulated Annealing to optimize a configuration."""
    rng = random.Random(int(time.time() * 1000) % 2**32)
    cur = [p[:] for p in seed_points]
    if not _valid(cur):
        # Try to fix validity
        for i in range(12):
            if not _valid(cur):
                cur[i] = _random_point()
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
        # Generate neighbor
        new_pts = [p[:] for p in cur]
        move_type = rng.random()
        
        if move_type < 0.5:
            # Small perturbation
            idx = rng.randrange(12)
            new_pts[idx][0] += rng.randint(-5, 5)
            new_pts[idx][1] += rng.randint(-5, 5)
        elif move_type < 0.8:
            # Snap to line
            i = rng.randrange(12)
            j = rng.randrange(12)
            k = rng.randrange(12)
            if i == j or i == k or j == k:
                continue
            x1, y1 = new_pts[j]
            x2, y2 = new_pts[k]
            dx = x2 - x1
            dy = y2 - y1
            if dx == 0 and dy == 0:
                continue
            # Find integer point on line y - y1 = m (x - x1)
            # y = m x + c
            # We want to move point i to a point on this line.
            # Simple approach: pick a random parameter t and see if it's integer.
            # Line: (y2-y1)x - (x2-x1)y + (x2*y1 - x1*y2) = 0
            A = y1 - y2
            B = x2 - x1
            C = x1 * y2 - x2 * y1
            g = gcd(gcd(abs(A), abs(B)), abs(C)) if (A or B or C) else 1
            if g:
                A //= g
                B //= g
                C //= g
            # A x + B y = -C
            # We need integer solutions.
            # If gcd(A,B) divides -C, solutions exist.
            g_ab = gcd(abs(A), abs(B))
            if -C % g_ab != 0:
                continue
            # Find one solution
            # A/g x + B/g y = -C/g
            a = A // g_ab
            b = B // g_ab
            c = -C // g_ab
            # Extended Euclid for a x0 + b y0 = 1
            def egcd(a, b):
                if b == 0:
                    return (a, 1, 0)
                g, x1, y1 = egcd(b, a % b)
                return (g, y1, x1 - (a // b) * y1)
            g_gcd, x0, y0 = egcd(a, b)
            if g_gcd != 1:
                continue
            x0 *= c
            y0 *= c
            # General solution: x = x0 + b t, y = y0 - a t
            t = rng.randint(-10, 10)
            nx = x0 + b * t
            ny = y0 - a * t
            if abs(nx) > 10**9 or abs(ny) > 10**9:
                continue
            new_pts[i] = [nx, ny]
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
    """Return 12 distinct integer points, |x|,|y| <= 10^9, not all collinear,
    minimizing ordinary lines. Uses Simulated Annealing."""
    
    best_pts = None
    best_score = float('inf')
    
    # Initial seeds
    seeds = [
        _make_seed_grid(),
    ]
    
    # Add some random seeds
    for _ in range(3):
        pts = [_random_point() for _ in range(12)]
        if _valid(pts):
            seeds.append(pts)
            
    # Run SA for each seed
    time_per_seed = 15.0
    
    for seed in seeds:
        if time.time() > _DEADLINE - 5:
            break
            
        pts, score = _simulate_annealing(seed, max_time=time_per_seed)
        if pts and score < best_score:
            best_score = score
            best_pts = pts
            
    # If we found a good solution, try to refine it further
    if best_pts is not None and best_score > 600 and time.time() < _DEADLINE - 20:
        pts, score = _simulate_annealing(best_pts, max_time=20.0)
        if pts and score < best_score:
            best_score = score
            best_pts = pts
            
    if best_pts is None:
        # Fallback
        best_pts = _make_seed_grid()
        
    # Final validation
    if not _valid(best_pts):
        best_pts = _make_seed_grid()
        
    return best_pts


if __name__ == "__main__":
    pts = construct()
    print("Points:", pts)
    print("Ordinary lines:", _count_ordinary(pts))
    print("3-point lines:", _count_3point_lines(pts))
    print("Objective:", _objective(pts))
    print("Distinct:", len(set(tuple(p) for p in pts)))
    print("Max coord:", max(abs(c) for p in pts for c in p))
    print("Collinear:", _collinear_check(pts))
