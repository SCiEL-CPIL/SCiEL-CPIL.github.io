import random
import math
import time
from fractions import Fraction
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    # Target: Maximize points in general position (no 3 collinear, no 4 concyclic)
    # with rational distances.
    # We use D=1 (integer coordinates) for simplicity.
    
    def is_square(n):
        if n < 0:
            return False
        r = math.isqrt(n)
        return r * r == n

    def are_collinear(p1, p2, p3):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        return (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1) == 0

    def are_concyclic(p1, p2, p3, p4):
        # Check if 4 points lie on a circle.
        # Using exact rational arithmetic.
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4
        
        # Determinant of:
        # | x1^2+y1^2  x1  y1  1 |
        # | x2^2+y2^2  x2  y2  1 |
        # | x3^2+y3^2  x3  y3  1 |
        # | x4^2+y4^2  x4  y4  1 |
        
        # To avoid huge numbers, we can check if p4 lies on the circle defined by p1,p2,p3.
        # Circle eq: A x^2 + A y^2 + B x + C y + D = 0
        # Subtract eq1 from eq2 and eq3 to eliminate D.
        
        x1sq = x1*x1 + y1*y1
        x2sq = x2*x2 + y2*y2
        x3sq = x3*x3 + y3*y3
        x4sq = x4*x4 + y4*y4
        
        # B(x2-x1) + C(y2-y1) = -(x2sq - x1sq)
        # B(x3-x1) + C(y3-y1) = -(x3sq - x1sq)
        
        a11 = x2 - x1
        a12 = y2 - y1
        b1 = x1sq - x2sq
        
        a21 = x3 - x1
        a22 = y3 - y1
        b2 = x1sq - x3sq
        
        det = a11 * a22 - a12 * a21
        if det == 0:
            # p1, p2, p3 collinear. Not a circle.
            # If p4 is also on the line, they are "concyclic" in the limit?
            # But we check collinearity separately.
            return False
            
        # Solve for B, C
        # B = (b1*a22 - b2*a12) / det
        # C = (a11*b2 - a21*b1) / det
        
        # Check p4:
        # x4sq + B*x4 + C*y4 + D = 0
        # D = -x1sq - B*x1 - C*y1
        # x4sq + B*x4 + C*y4 - x1sq - B*x1 - C*y1 = 0
        # B(x4-x1) + C(y4-y1) + (x4sq - x1sq) = 0
        
        # Substitute B, C:
        # (b1*a22 - b2*a12)(x4-x1) + (a11*b2 - a21*b1)(y4-y1) + det*(x4sq - x1sq) == 0 ?
        
        val1 = (b1 * a22 - b2 * a12) * (x4 - x1)
        val2 = (a11 * b2 - a21 * b1) * (y4 - y1)
        val3 = det * (x4sq - x1sq)
        
        return val1 + val2 + val3 == 0

    def check_set(points):
        n = len(points)
        # Check pairwise distances are rational (integers here)
        for i in range(n):
            for j in range(i+1, n):
                dx = points[i][0] - points[j][0]
                dy = points[i][1] - points[j][1]
                if not is_square(dx*dx + dy*dy):
                    return False
        
        # Check no 3 collinear
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    if are_collinear(points[i], points[j], points[k]):
                        return False
        
        # Check no 4 concyclic
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    for l in range(k+1, n):
                        if are_concyclic(points[i], points[j], points[k], points[l]):
                            return False
        return True

    # Generate a large pool of candidates
    # We will use a grid, but we will try multiple starting sets and search strategies.
    
    # Known small sets to start from
    start_sets = [
        [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)],
        [(0, 0), (3, 4), (0, 5), (3, 9), (6, 8)], # Check if this is valid?
        [(0, 0), (5, 0), (0, 12), (5, 12), (9, 16)] # Random guess
    ]
    
    # Validate start sets
    valid_starts = []
    for ss in start_sets:
        if check_set(ss):
            valid_starts.append(ss)
            
    if not valid_starts:
        # Fallback to the one from the original program
        valid_starts = [[(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]]
        
    best_set = valid_starts[0][:]
    
    # Candidate pool: Grid from -300 to 300
    # To speed up, we can pre-compute squared distances? No, points change.
    
    # We will perform a greedy search with random restarts on the candidate ordering.
    
    # To make it faster, we can filter candidates against the first point of the current set.
    
    def greedy_expand(start, candidates, max_iter=10000):
        current_set = start[:]
        # Shuffle candidates
        random.shuffle(candidates)
        
        # We will iterate through candidates and try to add them.
        # If we add one, we restart the candidate loop? No, that's expensive.
        # Just iterate once. If we want better, do multiple passes.
        
        # Optimization: Sort candidates by "potential"?
        # Let's just shuffle.
        
        for p in candidates:
            # Check if p is in current_set
            if p in current_set:
                continue
                
            # Check distances to all points in current_set
            valid_dist = True
            for q in current_set:
                dx = p[0] - q[0]
                dy = p[1] - q[1]
                if not is_square(dx*dx + dy*dy):
                    valid_dist = False
                    break
            
            if not valid_dist:
                continue
                
            # Check general position
            new_set = current_set + [p]
            if check_set(new_set):
                current_set = new_set
                # If we improved, we could try to find more.
                # For efficiency, we just continue with the same candidate list.
                # This might miss points that became valid only after adding p?
                # No, adding a point only makes it harder for others to be valid.
                # So a single pass is okay for a greedy approach.
        
        return current_set

    # Generate candidates
    # Using a grid is simple but maybe inefficient.
    # Let's use a grid of size 400x400.
    candidates = []
    for x in range(-300, 301):
        for y in range(-300, 301):
            candidates.append((x, y))
            
    # Remove points from start sets to avoid duplicates in pool
    # Actually, the check `if p in current_set` handles it.
    
    # Run multiple greedy expansions from different starts
    # and with different shuffles
    
    time_limit = time.time() + 110 # Leave 10s buffer
    
    for start in valid_starts:
        # Try a few different random orders
        for _ in range(5):
            if time.time() > time_limit:
                break
            result = greedy_expand(start, candidates)
            if len(result) > len(best_set):
                best_set = result[:]
                
    # If we found a set with > 5 points, we might try to expand it further with a different strategy?
    # For now, return the best found.
    
    # Format output
    points_out = []
    for x, y in best_set:
        points_out.append([[x, 1], [y, 1]])
        
    return {'D': 1, 'points': points_out}

# EVOLVE-BLOCK-END
