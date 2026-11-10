import time
import random
from math import isqrt
from itertools import combinations

def construct():
    start_time = time.time()
    time_limit = 115  # Safety margin for 120s limit (150s kill, but be safe)
    
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
        if len(set(points)) != n:
            return False
            
        for i in range(n):
            for j in range(i+1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                dx = x1 - x2
                dy = y1 - y2
                d2 = dx*dx + dy*dy
                if not is_square(d2):
                    return False
        
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    if check_collinear(points[i], points[j], points[k]):
                        return False
        
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    for l in range(k+1, n):
                        if check_concyclic(points[i], points[j], points[k], points[l]):
                            return False
                            
        return True

    # Known valid set of 5 points as fallback
    fallback_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    
    best_set = None
    best_size = 5
    
    # Try multiple box sizes, starting with smaller ones for speed
    for L in [10, 12, 14, 16, 18]:
        if time.time() - start_time > time_limit - 20:
            break
        
        points_in_box = []
        for x in range(-L, L+1):
            for y in range(-L, L+1):
                points_in_box.append((x, y))
                
        N = len(points_in_box)
        if N < 6:
            continue
            
        neighbors = [set() for _ in range(N)]
        
        max_d2 = 8 * L * L
        squares = set()
        r = 0
        while r*r <= max_d2:
            squares.add(r*r)
            r += 1
            
        for i in range(N):
            x1, y1 = points_in_box[i]
            for j in range(i+1, N):
                x2, y2 = points_in_box[j]
                dx = x1 - x2
                dy = y1 - y2
                d2 = dx*dx + dy*dy
                if d2 in squares:
                    neighbors[i].add(j)
                    neighbors[j].add(i)
        
        # Sort points by degree for better search
        degree_order = sorted(range(N), key=lambda i: len(neighbors[i]), reverse=True)
        
        for target_size in [6, 7, 8]:
            if time.time() - start_time > time_limit - 10:
                break
            
            max_restarts = 500
            found = False
            for restart in range(max_restarts):
                if time.time() - start_time > time_limit - 5:
                    break
                
                start_idx = degree_order[random.randint(0, min(50, N)-1)]
                current_indices = [start_idx]
                
                success = True
                for step in range(target_size - 1):
                    pivot = min(current_indices, key=lambda idx: len(neighbors[idx]))
                    cand_set = set(neighbors[pivot])
                    
                    for idx in current_indices:
                        if idx == pivot:
                            continue
                        cand_set.intersection_update(neighbors[idx])
                    
                    cand_set.difference_update(set(current_indices))
                    
                    if not cand_set:
                        success = False
                        break
                    
                    cand_list = list(cand_set)
                    if len(cand_list) <= 4:
                        next_idx = random.choice(cand_list)
                    else:
                        scores = []
                        for c in cand_list[:60]:
                            common = len(neighbors[c] & cand_set) - 1
                            scores.append((common, c))
                        scores.sort(reverse=True)
                        top_k = min(8, len(scores))
                        next_idx = random.choice([s[1] for s in scores[:top_k]])
                    
                    current_indices.append(next_idx)
                
                if success:
                    current_set = [points_in_box[idx] for idx in current_indices]
                    if validate_set(current_set):
                        if best_set is None or len(current_set) > len(best_set):
                            best_set = current_set
                            best_size = len(current_set)
                        found = True
                        if best_size >= target_size:
                            break
        
            if found and best_size >= target_size:
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
