"""Rewritten program: search for 20 integer points with no four collinear and minimal largest no-three-collinear subset."""
import random
import itertools
import time
import math
from collections import defaultdict

def _no_four_collinear(points):
    """Check that no four points are collinear."""
    n = len(points)
    # For every pair of points, count how many other points lie on the same line
    # We need to ensure no line contains 4 or more points.
    # Efficient approach: for each pair, compute the line, then check all others.
    # But O(n^4) is fine for n=20.
    
    # Better: for each set of 3 points, check if they are collinear; if so, check if there's a 4th.
    # Actually, simplest: enumerate all lines determined by pairs, count points on each line.
    
    # Use a dictionary mapping normalized line representation to set of point indices.
    # A line can be represented by (A, B, C) where Ax + By + C = 0, normalized.
    
    def line_rep(i, j):
        x1, y1 = points[i]
        x2, y2 = points[j]
        A = y2 - y1
        B = x1 - x2
        C = x2 * y1 - x1 * y2
        # Normalize: divide by gcd of A, B, C, and ensure a canonical sign
        g = math.gcd(math.gcd(abs(A), abs(B)), abs(C)) if (A != 0 or B != 0) else 1
        if g == 0:
            g = 1
        A //= g
        B //= g
        C //= g
        # Canonical sign: first non-zero of (A, B, C) should be positive
        if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
            A, B, C = -A, -B, -C
        return (A, B, C)
    
    line_to_points = defaultdict(set)
    for i in range(n):
        for j in range(i + 1, n):
            rep = line_rep(i, j)
            line_to_points[rep].add(i)
            line_to_points[rep].add(j)
    
    for rep, pts in line_to_points.items():
        if len(pts) >= 4:
            return False
    return True

def _largest_no_three_collinear(points):
    """Find the largest subset of points with no three collinear, using branch and bound."""
    n = len(points)
    
    # Precompute: for each pair, which other points are collinear with them
    # collinear_with[i][j] = set of k such that points[i], points[j], points[k] are collinear
    collinear_with = [[set() for _ in range(n)] for _ in range(n)]
    
    def are_collinear(p1, p2, p3):
        return (p2[0] - p1[0]) * (p3[1] - p1[1]) == (p2[1] - p1[1]) * (p3[0] - p1[0])
    
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(n):
                if k != i and k != j:
                    if are_collinear(points[i], points[j], points[k]):
                        collinear_with[i][j].add(k)
                        collinear_with[j][i].add(k)
    
    # Branch and bound to find max subset with no three collinear
    best = [0]
    best_set = [set()]
    
    # State: current set of selected points, next index to consider
    # We use a recursive backtracking with pruning.
    
    # Sort points to help with pruning? Maybe by some heuristic.
    # Let's just use index order.
    
    def backtrack(selected, next_idx, remaining_potential):
        # remaining_potential: upper bound on how many more we can add
        if len(selected) + remaining_potential <= best[0]:
            return
        
        if next_idx >= n:
            if len(selected) > best[0]:
                best[0] = len(selected)
                best_set[0] = set(selected)
            return
        
        # Option 1: skip point next_idx
        # Estimate remaining potential: (n - next_idx - 1) + 1 (the current point if we take it)
        # Actually, remaining_potential should be an upper bound on points we can still add from next_idx onwards.
        # A simple bound: n - next_idx
        backtrack(selected, next_idx + 1, n - next_idx - 1)
        
        # Option 2: take point next_idx, if it doesn't create a triple
        p = next_idx
        conflict = False
        for i, j in itertools.combinations(selected, 2):
            if p in collinear_with[i][j]:
                conflict = True
                break
        if not conflict:
            new_selected = list(selected) + [p]
            backtrack(new_selected, next_idx + 1, n - next_idx - 1)
    
    backtrack([], 0, n)
    return best[0]

def construct():
    """Construct 20 distinct integer points with |x|,|y| <= 100, no four collinear,
    minimizing the largest no-three-collinear subset."""
    
    random.seed(42)
    
    best_score = 21
    best_points = None
    
    # Start with a known valid configuration and try to improve via local search
    # Initial: points on parabola y = x^2
    initial = [[x, x * x] for x in range(-10, 10)]
    
    if not _no_four_collinear(initial):
        # Should not happen, but just in case
        return initial
    
    current_points = [p[:] for p in initial]
    current_score = _largest_no_three_collinear(current_points)
    
    best_score = current_score
    best_points = [p[:] for p in current_points]
    
    # Local search: try to move individual points to reduce the score
    start_time = time.time()
    max_time = 100  # seconds, leave buffer for the evaluation
    
    iteration = 0
    while time.time() - start_time < max_time:
        iteration += 1
        improved = False
        
        # Try moving each point to a nearby location
        for i in range(len(current_points)):
            if time.time() - start_time > max_time:
                break
            
            x, y = current_points[i]
            # Try nearby points
            for dx in range(-5, 6):
                for dy in range(-5, 6):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = x + dx, y + dy
                    if abs(nx) > 100 or abs(ny) > 100:
                        continue
                    
                    new_points = [p[:] for p in current_points]
                    new_points[i] = [nx, ny]
                    
                    # Check distinctness
                    if len(set(map(tuple, new_points))) != 20:
                        continue
                    
                    # Check no four collinear
                    if not _no_four_collinear(new_points):
                        continue
                    
                    new_score = _largest_no_three_collinear(new_points)
                    
                    if new_score < current_score:
                        current_points = new_points
                        current_score = new_score
                        improved = True
                        if current_score < best_score:
                            best_score = current_score
                            best_points = [p[:] for p in current_points]
                        break
                else:
                    continue
                break
        
        if not improved:
            # Try random perturbations
            for _ in range(100):
                if time.time() - start_time > max_time:
                    break
                
                i = random.randint(0, 19)
                x, y = current_points[i]
                nx = random.randint(max(-100, x - 10), min(100, x + 10))
                ny = random.randint(max(-100, y - 10), min(100, y + 10))
                
                new_points = [p[:] for p in current_points]
                new_points[i] = [nx, ny]
                
                if len(set(map(tuple, new_points))) != 20:
                    continue
                if not _no_four_collinear(new_points):
                    continue
                
                new_score = _largest_no_three_collinear(new_points)
                
                if new_score < current_score:
                    current_points = new_points
                    current_score = new_score
                    if current_score < best_score:
                        best_score = current_score
                        best_points = [p[:] for p in current_points]
                    break
    
    return best_points

# EVOLVE-BLOCK-START
def construct():
    """Construct 20 distinct integer points with |x|,|y| <= 100, no four collinear,
    minimizing the largest no-three-collinear subset."""
    
    random.seed(42)
    
    best_score = 21
    best_points = None
    
    # Start with a known valid configuration and try to improve via local search
    # Initial: points on parabola y = x^2
    initial = [[x, x * x] for x in range(-10, 10)]
    
    if not _no_four_collinear(initial):
        return initial
    
    current_points = [p[:] for p in initial]
    current_score = _largest_no_three_collinear(current_points)
    
    best_score = current_score
    best_points = [p[:] for p in current_points]
    
    # Local search: try to move individual points to reduce the score
    start_time = time.time()
    max_time = 100  # seconds, leave buffer for the evaluation
    
    iteration = 0
    while time.time() - start_time < max_time:
        iteration += 1
        improved = False
        
        # Try moving each point to a nearby location
        for i in range(len(current_points)):
            if time.time() - start_time > max_time:
                break
            
            x, y = current_points[i]
            # Try nearby points
            for dx in range(-5, 6):
                for dy in range(-5, 6):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = x + dx, y + dy
                    if abs(nx) > 100 or abs(ny) > 100:
                        continue
                    
                    new_points = [p[:] for p in current_points]
                    new_points[i] = [nx, ny]
                    
                    # Check distinctness
                    if len(set(map(tuple, new_points))) != 20:
                        continue
                    
                    # Check no four collinear
                    if not _no_four_collinear(new_points):
                        continue
                    
                    new_score = _largest_no_three_collinear(new_points)
                    
                    if new_score < current_score:
                        current_points = new_points
                        current_score = new_score
                        improved = True
                        if current_score < best_score:
                            best_score = current_score
                            best_points = [p[:] for p in current_points]
                        break
                else:
                    continue
                break
        
        if not improved:
            # Try random perturbations
            for _ in range(100):
                if time.time() - start_time > max_time:
                    break
                
                i = random.randint(0, 19)
                x, y = current_points[i]
                nx = random.randint(max(-100, x - 10), min(100, x + 10))
                ny = random.randint(max(-100, y - 10), min(100, y + 10))
                
                new_points = [p[:] for p in current_points]
                new_points[i] = [nx, ny]
                
                if len(set(map(tuple, new_points))) != 20:
                    continue
                if not _no_four_collinear(new_points):
                    continue
                
                new_score = _largest_no_three_collinear(new_points)
                
                if new_score < current_score:
                    current_points = new_points
                    current_score = new_score
                    if current_score < best_score:
                        best_score = current_score
                        best_points = [p[:] for p in current_points]
                    break
    
    return best_points
# EVOLVE-BLOCK-END
