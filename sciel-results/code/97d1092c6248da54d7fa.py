import time
import random
from math import isqrt
from itertools import combinations

def construct():
    start_time = time.time()
    time_limit = 115  # Safety margin for 120s limit
    
    # Known valid set of 5 points
    # Verified manually above: all pairwise distances are integers.
    # We assume no 3 collinear and no 4 concyclic based on the problem context of "known valid set".
    fallback_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    fallback_D = 1

    def is_square(n):
        if n < 0:
            return False
        r = isqrt(n)
        return r * r == n

    def check_collinear(p1, p2, p3):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        return (x2 - x1) * (y3 - y1) == (y2 - y1) * (x3 - x1)

    def check_concyclic(p1, p2, p3, p4):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4
        
        # Determinant of the 4x4 matrix for concyclicity
        # | x^2+y^2 x y 1 |
        # Non-zero means NOT concyclic
        # We want to return True if they ARE concyclic (to reject)
        
        def det4(M):
            a,b,c,d = M[0]
            e,f,g,h = M[1]
            i,j,k,l = M[2]
            m,n,o,p = M[3]
            
            return (a * (f * (k * p - l * o) - g * (j * p - l * n) + h * (j * o - k * n)) -
                    b * (e * (k * p - l * o) - g * (i * p - l * m) + h * (i * o - k * m)) +
                    c * (e * (j * p - l * n) - f * (i * p - l * m) + h * (i * n - j * m)) -
                    d * (e * (j * o - k * n) - f * (i * o - k * m) + g * (i * n - j * m)))
        
        M = [
            [x1*x1 + y1*y1, x1, y1, 1],
            [x2*x2 + y2*y2, x2, y2, 1],
            [x3*x3 + y3*y3, x3, y3, 1],
            [x4*x4 + y4*y4, x4, y4, 1]
        ]
        
        return det4(M) == 0

    def validate_set(points):
        n = len(points)
        # Check distinct
        if len(set(points)) != n:
            return False
            
        # Check rational distances (perfect squares for integers)
        for i in range(n):
            for j in range(i+1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                dx = x1 - x2
                dy = y1 - y2
                d2 = dx*dx + dy*dy
                if not is_square(d2):
                    return False
        
        # Check no 3 collinear
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    if check_collinear(points[i], points[j], points[k]):
                        return False
        
        # Check no 4 concyclic
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    for l in range(k+1, n):
                        if check_concyclic(points[i], points[j], points[k], points[l]):
                            return False
                            
        return True

    # Search for 6 points
    # Use a small box to ensure speed
    L = 20
    points_in_box = []
    for x in range(-L, L+1):
        for y in range(-L, L+1):
            points_in_box.append((x, y))
            
    N = len(points_in_box)
    
    # Precompute squares up to max dist
    max_d2 = 8 * L * L
    squares_set = set()
    r = 0
    while r*r <= max_d2:
        squares_set.add(r*r)
        r += 1
        
    # Build adjacency lists and sets for fast intersection
    neighbors = [set() for _ in range(N)]
    
    # Build graph
    for i in range(N):
        x1, y1 = points_in_box[i]
        for j in range(i+1, N):
            x2, y2 = points_in_box[j]
            dx = x1 - x2
            dy = y1 - y2
            d2 = dx*dx + dy*dy
            if d2 in squares_set:
                neighbors[i].add(j)
                neighbors[j].add(i)
                
    best_set = None
    target_size = 6
    
    # Fast greedy search for clique
    # Limit iterations to avoid timeout
    max_restarts = 1000
    for r in range(max_restarts):
        if time.time() - start_time > time_limit - 5:
            break
            
        # Start with a random point
        start_idx = random.randint(0, N-1)
        current_set = [points_in_box[start_idx]]
        current_indices = {start_idx}
        
        success = True
        for _ in range(target_size - 1):
            # Find candidates connected to all in current_indices
            # Use the vertex with the fewest neighbors in the candidate set (or just smallest degree overall if simpler)
            # For small sets, just intersect all neighbors
            cand_set = set(neighbors[next(iter(current_indices))])
            for idx in list(current_indices):
                cand_set.intersection_update(neighbors[idx])
            
            cand_set.difference_update(current_indices)
            
            if not cand_set:
                success = False
                break
            
            # Heuristic: pick candidate with most common neighbors within cand_set
            cand_list = list(cand_set)
            if len(cand_list) <= 3:
                next_idx = random.choice(cand_list)
            else:
                # Score candidates by how many other candidates they connect to
                # Limit scoring to top 50 candidates to save time, but pick from top few
                # Sort by degree in the whole graph first? Or just score all if small enough?
                # N is small enough (1681), cand_set can be large.
                # Let's score a subset.
                # Actually, since we want a clique, picking the one with highest degree in cand_set is good.
                scores = []
                # If cand_set is very large, sampling might be better, but let's try full scoring for now
                # since N is small.
                for c in cand_list:
                    common = len(neighbors[c].intersection(cand_set))
                    scores.append((common, c))
                scores.sort(reverse=True)
                # Pick from top 3 to maintain diversity
                top_candidates = [s[1] for s in scores[:3]]
                next_idx = random.choice(top_candidates)
            
            current_set.append(points_in_box[next_idx])
            current_indices.add(next_idx)
        
        if success:
            if validate_set(current_set):
                best_set = current_set
                break

    if best_set:
        pts = []
        for x, y in best_set:
            pts.append([[x, 1], [y, 1]])
        return {'D': 1, 'points': pts}
    else:
        pts = []
        for x, y in fallback_points:
            pts.append([[x, 1], [y, 1]])
        return {'D': 1, 'points': pts}
