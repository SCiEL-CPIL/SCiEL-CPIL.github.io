"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools
from collections import defaultdict

# EVOLVE-BLOCK-START

def _count_exact4(points):
    """Count lines containing exactly 4 of the given points. Also verify no 5 collinear."""
    n = len(points)
    if n != 16:
        return 0
    # Check distinctness
    pts_set = set(tuple(p) for p in points)
    if len(pts_set) != n:
        return 0
    # Check bounds
    for p in points:
        if abs(p[0]) > 1000 or abs(p[1]) > 1000:
            return 0
    
    # Count points per line
    line_counts = defaultdict(int)
    for i in range(n):
        for j in range(i+1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            # Normalize direction
            g = _gcd(abs(dx), abs(dy))
            if g > 0:
                dx //= g
                dy //= g
            # Canonical form: ensure first non-zero component is positive
            if dx < 0 or (dx == 0 and dy < 0):
                dx = -dx
                dy = -dy
            # Line key: direction + a point on the line (normalized)
            # Use (dx, dy, x1 - dx*y1, y1 - dy*x1) style or just (dx,dy, offset)
            # A line can be identified by (dx, dy, c) where c = dx*y - dy*x (constant on line)
            # But we need a unique representation. Let's use the normalized direction and 
            # the "anchor" which is the point on the line closest to origin in some sense,
            # or simply use the cross product constant.
            # For line through (x1,y1) with direction (dx,dy):
            # Normal vector is (dy, -dx). Line equation: dy*(x-x1) - dx*(y-y1) = 0
            # => dy*x - dx*y = dy*x1 - dx*y1
            # Let A = dy, B = -dx, C = -(dy*x1 - dx*y1) = dx*y1 - dy*x1
            # Normalize (A, B, C) by gcd
            A = dy
            B = -dx
            C = dx*y1 - dy*x1
            g2 = _gcd(_gcd(abs(A), abs(B)), abs(C))
            if g2 > 0:
                A //= g2
                B //= g2
                C //= g2
            # Canonical sign: make first non-zero of (A,B) positive
            if A < 0 or (A == 0 and B < 0):
                A = -A
                B = -B
                C = -C
            key = (A, B, C)
            line_counts[key] += 1
    
    # Now count lines with exactly 4 points
    # But wait: line_counts counts the number of *pairs* that lie on each line.
    # If a line has k points, it contributes C(k,2) to the count.
    # We need to recover k from C(k,2).
    # C(k,2) = k*(k-1)/2
    # For k=4: C(4,2)=6
    # For k=5: C(5,2)=10
    # For k=3: C(3,2)=3
    # For k=2: C(2,2)=1
    # For k=6: C(6,2)=15
    
    # Build a map from pair-count to k
    # We need to find k such that k*(k-1)//2 == count
    # And check if any line has k>=5
    
    exact4 = 0
    for key, pair_count in line_counts.items():
        # Find k such that k*(k-1)/2 == pair_count
        # k is at most 16
        found_k = None
        for k in range(2, 17):
            if k*(k-1)//2 == pair_count:
                found_k = k
                break
        if found_k is None:
            # This shouldn't happen for valid integer geometry, but just in case
            continue
        if found_k >= 5:
            return 0  # Invalid: five or more collinear
        if found_k == 4:
            exact4 += 1
    
    return exact4

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def _try_local_search(points, time_limit):
    """
    Start from initial points and try to improve via local search.
    Replace one point at a time with a random valid point to increase score.
    """
    start_time = time.time()
    best_score = _count_exact4(points)
    best_points = [p[:] for p in points]
    
    # Precompute current lines for efficiency? Maybe too complex.
    # Just do simple hill climbing.
    
    for i in range(16):
        if time.time() - start_time > time_limit:
            break
        
        # Try to replace point i
        best_new_point = None
        best_new_score = best_score
        
        # Try some random replacements
        # Focus on points that are "bad" or try all?
        # Let's try random points in the grid
        for _ in range(50):
            if time.time() - start_time > time_limit:
                break
            
            new_x = random.randint(-1000, 1000)
            new_y = random.randint(-1000, 1000)
            
            # Check if this point is already in the set (other than i)
            new_pt = (new_x, new_y)
            if any(p[0] == new_x and p[1] == new_y for j, p in enumerate(points) if j != i):
                continue
            
            new_points = [p[:] for p in points]
            new_points[i] = [new_x, new_y]
            
            score = _count_exact4(new_points)
            if score > best_new_score:
                best_new_score = score
                best_new_point = [new_x, new_y]
        
        if best_new_point is not None:
            points[i] = best_new_point
            best_score = best_new_score
            best_points = [p[:] for p in points]
    
    return best_points, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin
    
    # Initial configuration: 4x4 grid
    initial_points = [[x, y] for x in range(4) for y in range(4)]
    initial_score = _count_exact4(initial_points)
    
    best_points = [p[:] for p in initial_points]
    best_score = initial_score
    
    # Try local search from the initial configuration
    if time.time() < deadline:
        improved_points, improved_score = _try_local_search(initial_points, deadline - time.time() - 10)
        if improved_score > best_score:
            best_score = improved_score
            best_points = improved_points
    
    # Try a few random restarts
    random.seed(42)
    for restart in range(10):
        if time.time() > deadline:
            break
        
        # Generate random points
        random_points = []
        for _ in range(16):
            while True:
                x = random.randint(-1000, 1000)
                y = random.randint(-1000, 1000)
                if (x, y) not in set(tuple(p) for p in random_points):
                    random_points.append([x, y])
                    break
        
        score = _count_exact4(random_points)
        if score > best_score:
            best_score = score
            best_points = [p[:] for p in random_points]
        
        # Local search from random start
        if time.time() < deadline - 10:
            improved_points, improved_score = _try_local_search(random_points, deadline - time.time() - 10)
            if improved_score > best_score:
                best_score = improved_score
                best_points = improved_points
    
    return best_points

# EVOLVE-BLOCK-END
