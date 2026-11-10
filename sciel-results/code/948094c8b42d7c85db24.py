import time
import random
from math import isqrt

def construct():
    start_time = time.time()
    time_limit = 140  # Safety margin for 150s limit
    
    # Known valid set of 5 points
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

    # Use a smaller grid for faster search
    L = 30
    points_in_box = []
    for x in range(-L, L+1):
        for y in range(-L, L+1):
            points_in_box.append((x, y))
            
    N = len(points_in_box)
    
    # Precompute squares up to max dist
    max_d2 = 8 * L * L
    squares = set()
    r = 0
    while r*r <= max_d2:
        squares.add(r*r)
        r += 1
        
    # Build adjacency
    neighbors = [set() for _ in range(N)]
    
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
    
    # Convert to lists sorted by degree (ascending) for better pruning
    neighbors = [sorted(list(s)) for s in neighbors]
    degrees = [len(s) for s in neighbors]
    
    best_set = None
    target_size = 6
    
    # Greedy with random restarts and degree-based ordering
    num_restarts = 5000
    for r in range(num_restarts):
        if time.time() - start_time > time_limit - 5:
            break
            
        # Start with a random point, prefer lower degree points
        start_idx = random.randint(0, N-1)
        current_set = [points_in_box[start_idx]]
        current_indices = {start_idx}
        
        success = True
        for _ in range(target_size - 1):
            # Find candidates connected to all in current_indices
            # Use the point with minimum degree as pivot for efficiency
            pivot = min(current_indices, key=lambda idx: degrees[idx])
            cand_set = set(neighbors[pivot])
            
            for idx in current_indices:
                if idx == pivot:
                    continue
                cand_set.intersection_update(neighbors[idx])
            
            # Exclude those already in current_set
            cand_set.difference_update(current_indices)
            
            if not cand_set:
                success = False
                break
            
            # Pick random candidate from those with lower degree first
            cand_list = list(cand_set)
            # Sort by degree for better chances
            cand_list.sort(key=lambda idx: degrees[idx])
            
            # Pick from the first half (lower degree) for better pruning
            if len(cand_list) > 2:
                next_idx = random.choice(cand_list[:max(1, len(cand_list)//2)])
            else:
                next_idx = random.choice(cand_list)
            
            current_set.append(points_in_box[next_idx])
            current_indices.add(next_idx)
            
        if success:
            # Validate the set
            if validate_set(current_set):
                best_set = current_set
                break
    
    if best_set:
        # Convert to required format
        pts = []
        for x, y in best_set:
            pts.append([[x, 1], [y, 1]])
        return {'D': 1, 'points': pts}
    else:
        # Fallback to known 5 points
        pts = []
        for x, y in fallback_points:
            pts.append([[x, 1], [y, 1]])
        return {'D': 1, 'points': pts}
