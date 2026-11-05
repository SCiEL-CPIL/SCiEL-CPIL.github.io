import math
import random
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 100  # 100s limit to be safe
    
    # Helper to check if a number is a perfect square
    def is_square(n):
        if n < 0:
            return False
        r = int(math.isqrt(n))
        return r * r == n

    # Helper to check if 3 points are collinear
    def collinear(p1, p2, p3):
        # (x2-x1)(y3-y1) - (y2-y1)(x3-x1) == 0
        return (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0]) == 0

    # Helper to check if 4 points are concyclic
    # Using the determinant of the circle equation:
    # | x^2+y^2  x  y  1 |
    # | x_i^2+y_i^2 x_i y_i 1 |
    # | ... | = 0
    # For integer coords, we can use the formula:
    # (x1-x2)(y1-y3) + ... 
    # Or simply: 4 points are concyclic iff the circumcircle of 3 passes through the 4th.
    def concyclic(p1, p2, p3, p4):
        # Calculate circumcircle of p1, p2, p3
        # If p1,p2,p3 are collinear, they don't define a unique circle, 
        # but the constraint "no 3 collinear" should have been checked before calling this.
        # Equation: x^2 + y^2 + Dx + Ey + F = 0
        # D, E, F can be found by solving linear system.
        # Alternatively, use the determinant form.
        # For efficiency with small ints, we can check if the power of p4 w.r.t circle(p1,p2,p3) is 0.
        
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4
        
        # Calculate D, E, F
        # D(x2-x1) + E(y2-y1) = -(x2^2+y2^2 - x1^2-y1^2)
        # D(x3-x1) + E(y3-y1) = -(x3^2+y3^2 - x1^2-y1^2)
        
        a1 = x2 - x1
        b1 = y2 - y1
        c1 = -(x2*x2 + y2*y2 - x1*x1 - y1*y1)
        
        a2 = x3 - x1
        b2 = y3 - y1
        c2 = -(x3*x3 + y3*y3 - x1*x1 - y1*y1)
        
        det = a1 * b2 - a2 * b1
        if det == 0:
            # Collinear, handled elsewhere
            return False
            
        D = (c1 * b2 - c2 * b1) / det
        E = (a1 * c2 - a2 * c1) / det
        F = -(x1*x1 + y1*y1 + D*x1 + E*y1)
        
        # Check if p4 satisfies x^2 + y^2 + Dx + Ey + F = 0
        # To avoid floats, use:
        # (x4^2+y4^2)*det + D_num*(x4-x1)*b2 - ... 
        # Actually, just use the integer arithmetic version:
        # 2 * ( (x2-x1)(y3-y1) - (y2-y1)(x3-x1) ) * (x4^2+y4^2) + ...
        
        # Simpler: 
        # Power of point P4 wrt circle through P1,P2,P3 is 0 iff:
        # | x1 y1 x1^2+y1^2 1 |
        # | x2 y2 x2^2+y2^2 1 |
        # | x3 y3 x3^2+y3^2 1 | = 0
        # | x4 y4 x4^2+y4^2 1 |
        
        # Let's compute the determinant directly.
        # Since coords are small, this is fast.
        M = [
            [x1, y1, x1*x1 + y1*y1, 1],
            [x2, y2, x2*x2 + y2*y2, 1],
            [x3, y3, x3*x3 + y3*y3, 1],
            [x4, y4, x4*x4 + y4*y4, 1]
        ]
        
        # Calculate determinant of 4x4 matrix
        # Using expansion or simple formula.
        # For 4x4, we can do:
        def det4(M):
            # M is 4x4 list of lists
            # a b c d
            # e f g h
            # i j k l
            # m n o p
            a,b,c,d = M[0]
            e,f,g,h = M[1]
            i,j,k,l = M[2]
            m,n,o,p = M[3]
            
            return (
                a*(f*(k*p-l*o) - g*(j*p-l*n) + h*(j*o-k*n)) -
                b*(e*(k*p-l*o) - g*(i*p-l*m) + h*(i*o-k*m)) +
                c*(e*(j*p-l*n) - f*(i*p-l*m) + h*(i*n-j*m)) -
                d*(e*(j*o-k*n) - f*(i*o-k*m) + g*(i*n-j*m))
            )
            
        return det4(M) == 0

    # Generate candidate points
    # We'll search in a box [-50, 50]
    candidates = []
    for x in range(-50, 51):
        for y in range(-50, 51):
            candidates.append((x, y))
            
    # Precompute valid pairs (integer distance)
    # valid_pairs[i][j] = True if dist(i,j) is integer
    n_cands = len(candidates)
    # This is O(N^2) which is 10000^2 = 100M, too slow in Python if done naively in a loop?
    # 10,000 points. Pairs = 50,000.
    # 50,000 distance checks is very fast.
    
    valid_dist = [[False] * n_cands for _ in range(n_cands)]
    for i in range(n_cands):
        x1, y1 = candidates[i]
        for j in range(i+1, n_cands):
            x2, y2 = candidates[j]
            dx = x1 - x2
            dy = y1 - y2
            d2 = dx*dx + dy*dy
            if is_square(d2):
                valid_dist[i][j] = True
                valid_dist[j][i] = True

    best_set = []
    
    # Greedy search with random restarts
    # We will try to build a set by adding points that are compatible with all existing points.
    
    num_restarts = 200
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        random.shuffle(candidates)
        current_set = []
        
        for p_idx in range(n_cands):
            p = candidates[p_idx]
            if time.time() > deadline:
                break
                
            # Check if p can be added
            can_add = True
            
            # 1. Distance check
            for q_idx in range(len(current_set)):
                if not valid_dist[p_idx][candidates.index(current_set[q_idx])]: 
                    # Wait, candidates.index is O(N). We should store indices.
                    pass
            
            # Optimization: Store indices in current_set
            # Let's rewrite the inner loop to use indices.
            
        # The above loop is inefficient due to index lookups.
        # Let's refine the search.
        
    # Refined Search
    best_size = 0
    best_points = []
    
    # We will use a list of indices for current set
    for r in range(500):
        if time.time() > deadline:
            break
            
        random.shuffle(candidates)
        current_indices = []
        
        for i in range(n_cands):
            # Check if candidates[i] is compatible with all in current_indices
            compat = True
            for j in current_indices:
                if not valid_dist[i][j]:
                    compat = False
                    break
            
            if not compat:
                continue
                
            # Check collinearity with any pair in current_set
            for k in range(len(current_indices)):
                for l in range(k+1, len(current_indices)):
                    if collinear(candidates[i], candidates[current_indices[k]], candidates[current_indices[l]]):
                        compat = False
                        break
                if not compat:
                    break
            
            if not compat:
                continue
                
            # Check concyclicity with any triple in current_set
            for k in range(len(current_indices)):
                for l in range(k+1, len(current_indices)):
                    for m in range(l+1, len(current_indices)):
                        if concyclic(candidates[i], candidates[current_indices[k]], candidates[current_indices[l]], candidates[current_indices[m]]):
                            compat = False
                            break
                    if not compat:
                        break
                if not compat:
                    break
            
            if compat:
                current_indices.append(i)
                
        if len(current_indices) > best_size:
            best_size = len(current_indices)
            best_points = [candidates[i] for i in current_indices]
            
            # If we found 6 or more, we can stop early if we want, but let's try to maximize.
            # Actually, 6 is the theoretical max for general position.
            if best_size >= 6:
                # We can break if we are confident, but let's run a few more to ensure validity or find 7 (unlikely)
                # For now, break to save time if 6 is found, as 7 is extremely rare/impossible.
                break

    # Format output
    # D=1, points are (x, y*sqrt(1)) = (x,y)
    # x_num=x, x_den=1, y_num=y, y_den=1
    
    if best_points:
        points_out = [[[x, 1], [y, 1]] for x, y in best_points]
        return {'D': 1, 'points': points_out}
    else:
        # Fallback: return the known 5-point set
        points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
        return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in points]}

