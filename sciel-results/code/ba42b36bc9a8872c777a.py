# EVOLVE-BLOCK-START
import itertools
import math
import random

def is_perfect_square(n):
    if n < 0:
        return False
    r = int(math.isqrt(n))
    return r * r == n

def dist_sq(p1, p2):
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return dx * dx + dy * dy

def are_collinear(p1, p2, p3):
    dx1 = p2[0] - p1[0]
    dy1 = p2[1] - p1[1]
    dx2 = p3[0] - p1[0]
    dy2 = p3[1] - p1[1]
    return dx1 * dy2 - dy1 * dx2 == 0

def are_concyclic(p1, p2, p3, p4):
    def row(p):
        return [p[0]*p[0] + p[1]*p[1], p[0], p[1], 1]
    
    r1 = row(p1)
    r2 = row(p2)
    r3 = row(p3)
    r4 = row(p4)
    
    def det4(m):
        sign = 1
        total = 0
        for col in range(4):
            minor = []
            for i in range(1, 4):
                minor.append([m[i][j] for j in range(4) if j != col])
            def det3(mat):
                return (mat[0][0]*(mat[1][1]*mat[2][2]-mat[1][2]*mat[2][1])
                      - mat[0][1]*(mat[1][0]*mat[2][2]-mat[1][2]*mat[2][0])
                      + mat[0][2]*(mat[1][0]*mat[2][1]-mat[1][1]*mat[2][0]))
            total += sign * m[0][col] * det3(minor)
            sign = -sign
        return total
    
    return det4([r1, r2, r3, r4]) == 0

def construct():
    # Fallback 5-point set
    fallback_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    
    # Try to find a 6-point set
    bound = 200
    candidates = []
    for x in range(-bound, bound + 1):
        for y in range(-bound, bound + 1):
            candidates.append((x, y))
            
    # Precompute distances and validity
    # To speed up, we can filter candidates that are not collinear with others?
    # Just use greedy with more restarts and a larger bound.
    
    import time
    start_time = time.time()
    timeout = 110 # 10 seconds for search, leave 110s buffer? No, 120s total.
    
    best_set = None
    
    # Use a randomized greedy approach
    random.seed(12345)
    
    for attempt in range(2000):
        if time.time() - start_time > 110:
            break
            
        shuffled = candidates[:]
        random.shuffle(shuffled)
        
        selected = []
        for p in shuffled:
            if len(selected) >= 6:
                break
            compatible = True
            for q in selected:
                if not is_perfect_square(dist_sq(p, q)):
                    compatible = False
                    break
            if not compatible:
                continue
            
            collinear_ok = True
            for i in range(len(selected) - 1):
                for j in range(i + 1, len(selected)):
                    if are_collinear(selected[i], selected[j], p):
                        collinear_ok = False
                        break
                if not collinear_ok:
                    break
            if not collinear_ok:
                continue
                
            concyclic_ok = True
            if len(selected) >= 3:
                for i in range(len(selected) - 2):
                    for j in range(i + 1, len(selected) - 1):
                        for k in range(j + 1, len(selected)):
                            if are_concyclic(selected[i], selected[j], selected[k], p):
                                concyclic_ok = False
                                break
                        if not concyclic_ok:
                            break
                    if not concyclic_ok:
                        break
            if concyclic_ok:
                selected.append(p)
                
        if len(selected) == 6:
            best_set = selected
            break
            
    if best_set is None:
        return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in fallback_points]}
        
    return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in best_set]}
# EVOLVE-BLOCK-END
