import time
import random
from math import isqrt
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10 second safety margin
    
    # Helper to check if a number is a perfect square
    def is_square(n):
        if n < 0:
            return False
        s = isqrt(n)
        return s * s == n

    # Helper to check collinearity of three points (integers)
    # (x2-x1)(y3-y1) - (x3-x1)(y2-y1) != 0
    def not_collinear(p1, p2, p3):
        return (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]) != 0

    # Helper to check if four points are concyclic
    # Determinant of the matrix:
    # | x^2+y^2  x  y  1 |
    # | ...      ... ... 1 |
    # Must be non-zero.
    def not_concyclic(p1, p2, p3, p4):
        # Expand determinant along the last column
        # Or use the formula:
        # | x1 y1 x1^2+y1^2 1 |
        # | x2 y2 x2^2+y2^2 1 | = 0
        # | x3 y3 x3^2+y3^2 1 |
        # | x4 y4 x4^2+y4^2 1 |
        
        def det4(m):
            # m is a 4x4 list of lists
            # Simple expansion for 4x4
            a, b, c, d = m[0]
            e, f, g, h = m[1]
            i, j, k, l = m[2]
            m4, n, o, p = m[3]
            
            # Using Sarrus-like rule or cofactor expansion
            # Cofactor expansion along first row
            def det3(sub):
                s1, s2, s3 = sub[0]
                t1, t2, t3 = sub[1]
                u1, u2, u3 = sub[2]
                return s1*(t2*u3 - t3*u2) - s2*(t1*u3 - t3*u1) + s3*(t1*u2 - t2*u1)
            
            return (a * det3([[f,g,h],[j,k,l],[n,o,p]]) - 
                    b * det3([[e,g,h],[i,k,l],[m4,o,p]]) + 
                    c * det3([[e,f,h],[i,j,l],[m4,n,p]]) - 
                    d * det3([[e,f,g],[i,j,k],[m4,n,o]]))

        pts = [p1, p2, p3, p4]
        matrix = []
        for x, y in pts:
            matrix.append([x, y, x*x + y*y, 1])
        
        return det4(matrix) != 0

    # Check if a point p can be added to set S
    def can_add(p, S):
        if p in S:
            return False
        for q in S:
            dx = p[0] - q[0]
            dy = p[1] - q[1]
            dist_sq = dx*dx + dy*dy
            if not is_square(dist_sq):
                return False
        for q, r in combinations(S, 2):
            if not not_collinear(p, q, r):
                return False
        for q, r, s in combinations(S, 3):
            if not not_concyclic(p, q, r, s):
                return False
        return True

    # Check if a set S is valid
    def is_valid_set(S):
        if len(S) < 3:
            return False
        for p, q, r in combinations(S, 3):
            if not not_collinear(p, q, r):
                return False
        for p, q, r, s in combinations(S, 4):
            if not not_concyclic(p, q, r, s):
                return False
        for p, q in combinations(S, 2):
            dx = p[0] - q[0]
            dy = p[1] - q[1]
            if not is_square(dx*dx + dy*dy):
                return False
        return True

    # Generate candidate points
    # We use a grid search around the origin
    max_coord = 200
    candidates = []
    for x in range(-max_coord, max_coord + 1):
        for y in range(-max_coord, max_coord + 1):
            candidates.append((x, y))
    
    # Shuffle candidates to avoid bias
    random.shuffle(candidates)

    best_set = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)] # Seed from prompt
    best_size = len(best_set)

    # Try to expand the seed
    current_set = list(best_set)
    
    # Greedy expansion
    for p in candidates:
        if time.time() > deadline:
            break
        if can_add(p, current_set):
            current_set.append(p)
            if len(current_set) > best_size:
                best_set = list(current_set)
                best_size = len(current_set)
            # If we reach a large size, we can stop early or continue to find more
            if best_size >= 8:
                break

    # If we didn't find a larger set with the seed, try randomized restarts
    if best_size < 6:
        for _ in range(100):
            if time.time() > deadline:
                break
            
            # Pick a random starting point
            p1 = random.choice(candidates)
            current_set = [p1]
            
            # Add random valid points
            for p in candidates:
                if time.time() > deadline:
                    break
                if p != p1 and can_add(p, current_set):
                    current_set.append(p)
                    if len(current_set) > best_size:
                        best_set = list(current_set)
                        best_size = len(current_set)
                    if best_size >= 8:
                        break
            if best_size >= 8:
                break

    # Format the result
    # D = 1
    # Points are [[x_num, x_den], [y_num, y_den]]
    result_points = []
    for x, y in best_set:
        result_points.append([[x, 1], [y, 1]])
        
    return {'D': 1, 'points': result_points}

# EVOLVE-BLOCK-END
