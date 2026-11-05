"""Direct construction: Parallel-Line Simulated Annealing for 12-point configuration.

Uses a structural constraint: 4 parallel lines, each with 3 points.
This reduces the search space and targets configurations with high collinearity.
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


def _parallel_sa(max_time=10.0):
    """
    Simulated Annealing constrained to 4 parallel lines (y=0,1,2,3).
    Each line has 3 points.
    Variables: x-coordinates of the 12 points.
    """
    rng = random.Random(int(time.time() * 1000) % 2**32)
    
    # Initialize: 4 lines, 3 points each.
    # x_coords[line_idx][point_idx]
    # y_coords are fixed: 0, 1, 2, 3
    # Start with a regular grid
    x_coords = []
    for i in range(4):
        row = []
        for j in range(3):
            row.append(i * 10 + j) # Spread out initially
        x_coords.append(row)
        
    def get_points():
        pts = []
        for i in range(4):
            for j in range(3):
                pts.append([x_coords[i][j], i])
        return pts

    cur_pts = get_points()
    if not _valid(cur_pts):
        # Try to fix if duplicates (unlikely with this init)
        return None, float('inf')
        
    cur_obj = _objective(cur_pts)
    best_pts = [p[:] for p in cur_pts]
    best_obj = cur_obj
    
    T = 50.0
    T_min = 0.1
    cooling_rate = 0.999
    
    start_time = time.time()
    
    while T > T_min and time.time() - start_time < max_time:
        new_x = [row[:] for row in x_coords]
        move_type = rng.random()
        
        if move_type < 0.4:
            # Move a single point's x-coordinate
            i = rng.randrange(4)
            j = rng.randrange(3)
            new_x[i][j] += rng.randint(-5, 5)
        elif move_type < 0.7:
            # Shift an entire line
            i = rng.randrange(4)
            shift = rng.randint(-5, 5)
            for j in range(3):
                new_x[i][j] += shift
        elif move_type < 0.9:
            # Scale a line (change spacing)
            i = rng.randrange(4)
            # Try to make them arithmetic progression?
            # Just perturb all 3
            for j in range(3):
                new_x[i][j] += rng.randint(-2, 2)
        else:
            # Swap two points on the same line (reordering)
            i = rng.randrange(4)
            j1, j2 = rng.sample(range(3), 2)
            new_x[i][j1], new_x[i][j2] = new_x[i][j2], new_x[i][j1]
            
        # Construct new points
        new_pts = []
        for i in range(4):
            for j in range(3):
                new_pts.append([new_x[i][j], i])
                
        if not _valid(new_pts):
            T *= cooling_rate
            continue
            
        new_obj = _objective(new_pts)
        delta = new_obj - cur_obj
        
        if delta < 0 or rng.random() < pow(2.71828, -delta / T):
            x_coords = new_x
            cur_pts = new_pts
            cur_obj = new_obj
            if cur_obj < best_obj:
                best_obj = cur_obj
                best_pts = [p[:] for p in cur_pts]
                
        T *= cooling_rate
        
    return best_pts, best_obj


def construct():
    """Return 12 distinct integer points, |x|,|y| <= 10^9, not all collinear,
    minimizing ordinary lines. Uses Parallel-Line SA."""
    
    best_pts = None
    best_score = float('inf')
    
    # Run multiple instances of Parallel SA with different seeds
    for _ in range(10):
        if time.time() > _DEADLINE - 10:
            break
            
        # We can vary the initial spread or seed
        # The function _parallel_sa uses a fixed internal seed based on time, 
        # so we need to force diversity.
        # Let's modify _parallel_sa to accept a seed or just run it multiple times.
        # Since _parallel_sa uses time-based seed, running it in a loop with small sleeps
        # or just relying on the randomness of the moves is fine.
        
        pts, score = _parallel_sa(max_time=10.0)
        if pts and score < best_score:
            best_score = score
            best_pts = pts
            
    # If we haven't found a very good solution, try a general SA as fallback
    # But the directive asks for structural change. The parallel constraint IS the change.
    # If parallel SA fails to find < 6 ordinary lines, it might be that the optimum
    # doesn't have 4 parallel lines.
    
    # Let's add a "Free" SA but with the same improved objective and moves
    # to ensure we don't get stuck if the parallel assumption is wrong.
    # However, to keep the code clean and focused on the structural change,
    # we will stick to the parallel approach but add a "Perturb Y" move
    # that allows the lines to become non-parallel?
    # No, that breaks the structure.
    
    # Let's try one more structural variant: 3 parallel lines of 4 points.
    # Modify _parallel_sa to support 3 lines of 4?
    # Let's just run the 4x3 version.
    
    if best_pts is None:
        # Fallback to a simple grid
        best_pts = [[i, j] for i in range(4) for j in range(3)]
        
    # Final validation
    if not _valid(best_pts):
        best_pts = [[i, j] for i in range(4) for j in range(3)]
        
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
