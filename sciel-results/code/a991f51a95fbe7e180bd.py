"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import random
from collections import defaultdict

def _gcd(a, b):
    a, b = abs(a), abs(b)
    while b:
        a, b = b, a % b
    return a

def _line_key(p1, p2):
    """Return a canonical key for the line through p1 and p2."""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    g = _gcd(dx, dy)
    if g == 0:
        return None
    dx //= g
    dy //= g
    # Canonical direction: first non-zero component positive
    if dx < 0 or (dx == 0 and dy < 0):
        dx, dy = -dx, -dy
    # Canonical point: smallest point on the line in lex order among our set is not needed;
    # we use the line equation: dy*x - dx*y = c
    c = dy * p1[0] - dx * p1[1]
    return (dx, dy, c)

def _count_exact_four(points):
    """Count lines containing exactly 4 of the given points. Return (count, valid)."""
    n = len(points)
    if n != 16:
        return 0, False
    # Check distinctness and bounds
    seen = set()
    for p in points:
        if p in seen:
            return 0, False
        seen.add(p)
        if abs(p[0]) > 1000 or abs(p[1]) > 1000:
            return 0, False
    
    # Map each line to the set of points on it
    line_points = defaultdict(set)
    for i in range(n):
        for j in range(i+1, n):
            key = _line_key(points[i], points[j])
            if key is None:
                continue
            line_points[key].add(i)
    
    count = 0
    for key, pts in line_points.items():
        if len(pts) == 4:
            count += 1
        elif len(pts) >= 5:
            return 0, False  # Five or more collinear: invalid
    return count, True

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s
    
    # The affine plane AG(2,4) has 16 points and 20 lines of size 4.
    # We want to embed this into the integer lattice such that collinearity is preserved
    # and no 5 points are collinear.
    
    # Strategy: Use a direct constructive embedding.
    # AG(2,4) can be represented as GF(4)^2. 
    # GF(4) = {0, 1, alpha, alpha+1} where alpha^2 = alpha + 1.
    # Points: (x, y) where x, y in GF(4).
    # Lines: sets of 4 points {P, P+v, P+2v, P+3v} for direction v.
    
    # We need to map GF(4)^2 -> Z^2 preserving collinearity.
    # A simple linear map won't work directly because GF(4) is not a subfield of Z.
    # However, we can use a specific known construction or search for a good embedding.
    
    # Known construction: The 4x4 grid has 10 lines of 4 (4 rows, 4 cols, 2 diags).
    # We can try to perturb the grid or use a different structure.
    
    # Let's try a randomized search with local improvement, guided by the structure of AG(2,4).
    # Actually, let's first try to find a static solution known in literature.
    # The maximum number of 4-point lines for 16 points is 20 (AG(2,4)).
    # Is there an integer realization?
    
    # Let's use a heuristic search:
    # 1. Start with a random set of 16 points.
    # 2. Evaluate score.
    # 3. Local search: move one point to a nearby location to improve score.
    
    # But random search is slow. Let's try a structured approach.
    # We can try to map the AG(2,4) structure to integer coordinates.
    # One known embedding uses coordinates based on powers of 2 or similar.
    
    # Let's try a specific candidate:
    # Points: (0,0), (1,0), (2,0), (4,0) - no, that's 4 collinear.
    # We need to avoid 5 collinear.
    
    # Let's try a grid-based perturbation.
    # Start with 4x4 grid: {(i, j) for i in 0..3, j in 0..3}. Score = 10.
    # Try to shift some points to create more 4-point lines.
    
    # Alternative: Use the fact that AG(2,4) lines correspond to solutions of linear equations over GF(4).
    # We can assign integer coordinates to GF(4) elements such that:
    # a + b + c + d = 0 in GF(4) iff the 4 corresponding integer points are collinear.
    # This is a strong constraint.
    
    # Let's try a specific mapping:
    # Map GF(4) to {0, 1, 2, 4} in Z? 
    # If we use x-coord from {0,1,2,4} and y-coord from {0,1,2,4}, we get 16 points.
    # Check collinearity:
    # Points: (0,0), (0,1), (0,2), (0,4) -> 4 collinear (vertical).
    # (1,0), (1,1), (1,2), (1,4) -> 4 collinear.
    # (2,0), (2,1), (2,2), (2,4) -> 4 collinear.
    # (4,0), (4,1), (4,2), (4,4) -> 4 collinear.
    # Rows: (0,0),(1,0),(2,0),(4,0) -> 4 collinear.
    # (0,1),(1,1),(2,1),(4,1) -> 4 collinear.
    # (0,2),(1,2),(2,2),(4,2) -> 4 collinear.
    # (0,4),(1,4),(2,4),(4,4) -> 4 collinear.
    # Diagonals? (0,0),(1,1),(2,2),(4,4) -> slope 1, 4 collinear.
    # (0,4),(1,3)? No, (1,3) not in set.
    # (0,2),(1,1),(2,0) -> slope -1, but (4,-2) not in set.
    # (0,4),(1,2)? No.
    # Let's count: 4 cols + 4 rows + 1 diag = 9.
    # Are there more?
    # (0,0),(2,1)? No.
    # This gives 9. Not better than 10.
    
    # Let's try {0,1,2,3} for x and {0,1,2,3} for y (standard grid). Score 10.
    
    # Let's try a randomized local search starting from the grid.
    
    best_points = [[i, j] for i in range(4) for j in range(4)]
    best_score, _ = _count_exact_four(best_points)
    
    # Local search
    current_points = [p[:] for p in best_points]
    current_score = best_score
    
    random.seed(42)
    
    while time.time() < deadline:
        # Pick a random point to move
        idx = random.randint(0, 15)
        old_x, old_y = current_points[idx]
        
        # Try moving to a nearby point
        found_improvement = False
        for dx in range(-5, 6):
            for dy in range(-5, 6):
                if dx == 0 and dy == 0:
                    continue
                new_x, new_y = old_x + dx, old_y + dy
                if abs(new_x) > 1000 or abs(new_y) > 1000:
                    continue
                # Check if new point is already in set
                new_point = (new_x, new_y)
                if new_point in set(tuple(p) for p in current_points if p != current_points[idx]):
                    continue
                
                # Create new set
                new_points = current_points[:]
                new_points[idx] = [new_x, new_y]
                
                score, valid = _count_exact_four(new_points)
                if valid and score > current_score:
                    current_points = new_points
                    current_score = score
                    if score > best_score:
                        best_score = score
                        best_points = [p[:] for p in current_points]
                    found_improvement = True
                    break
            if found_improvement:
                break
        
        # If no improvement, try a random jump
        if not found_improvement and random.random() < 0.1:
            # Random restart within bounds
            current_points = [[random.randint(-100, 100), random.randint(-100, 100)] for _ in range(16)]
            # Ensure distinct
            seen = set()
            valid = True
            for i in range(16):
                while tuple(current_points[i]) in seen:
                    current_points[i] = [random.randint(-100, 100), random.randint(-100, 100)]
                seen.add(tuple(current_points[i]))
            score, valid = _count_exact_four(current_points)
            if valid:
                current_score = score
                if score > best_score:
                    best_score = score
                    best_points = [p[:] for p in current_points]
    
    return best_points
# EVOLVE-BLOCK-END
