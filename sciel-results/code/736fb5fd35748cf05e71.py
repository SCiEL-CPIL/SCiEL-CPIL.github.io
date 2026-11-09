import time
import random
import sys

sys.setrecursionlimit(10000)

def _are_collinear(p1, p2, p3):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    return (x2 - x1) * (y3 - y1) == (y2 - y1) * (x3 - x1)

def _has_four_collinear(points):
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                for l in range(k + 1, n):
                    if _are_collinear(points[i], points[j], points[k]) and \
                       _are_collinear(points[i], points[j], points[l]):
                        return True
    return False

def _max_no_three_collinear(points):
    n = len(points)
    
    # Precompute triple conflict masks
    # triple_mask[i][j] = bitmask of points k such that i, j, k are collinear
    triple_mask = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(n):
                if k != i and k != j and _are_collinear(points[i], points[j], points[k]):
                    triple_mask[i][j] |= (1 << k)
                    triple_mask[j][i] |= (1 << k)
    
    best = [0]
    
    def branch_and_bound(idx, current_mask, current_size):
        if idx >= n:
            if current_size > best[0]:
                best[0] = current_size
            return
        
        # Prune
        if current_size + (n - idx) <= best[0]:
            return
        
        # Try including point idx first (to find large solutions quickly)
        can_add = True
        cm = current_mask
        while cm:
            low = cm & (-cm)
            a = low.bit_length() - 1
            if triple_mask[idx][a] & current_mask:
                can_add = False
                break
            cm ^= low
        
        if can_add:
            branch_and_bound(idx + 1, current_mask | (1 << idx), current_size + 1)
        
        # Try excluding
        branch_and_bound(idx + 1, current_mask, current_size)
    
    # Greedy for initial lower bound
    current_mask = 0
    for i in range(n):
        can_add = True
        cm = current_mask
        while cm:
            low = cm & (-cm)
            a = low.bit_length() - 1
            if triple_mask[i][a] & current_mask:
                can_add = False
                break
            cm ^= low
        if can_add:
            current_mask |= (1 << i)
    best[0] = bin(current_mask).count('1')
    
    branch_and_bound(0, 0, 0)
    return best[0]

def _compute_score(points):
    return _max_no_three_collinear(points)

def _is_valid(points):
    if len(points) != 20:
        return False
    if len(set(map(tuple, points))) != 20:
        return False
    for x, y in points:
        if abs(x) > 100 or abs(y) > 100:
            return False
    if _has_four_collinear(points):
        return False
    return True

def _random_valid_point(occupied):
    while True:
        x = random.randint(-100, 100)
        y = random.randint(-100, 100)
        if (x, y) not in occupied:
            return [x, y]

def _count_collinear_triples(points):
    n = len(points)
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if _are_collinear(points[i], points[j], points[k]):
                    count += 1
    return count

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    random.seed(42)
    
    points = []
    occupied = set()
    for _ in range(20):
        while True:
            x = random.randint(-80, 80)
            y = random.randint(-80, 80)
            if (x, y) not in occupied:
                occupied.add((x, y))
                points.append([x, y])
                break
    
    attempts = 0
    while _has_four_collinear(points) and attempts < 2000:
        idx = random.randint(0, 19)
        old = points[idx][:]
        occupied.discard(tuple(old))
        new_pt = _random_valid_point(occupied)
        points[idx] = new_pt
        occupied.add(tuple(new_pt))
        attempts += 1
    
    if _has_four_collinear(points):
        return [[x, x * x] for x in range(-10, 10)]
    
    best_points = [p[:] for p in points]
    best_score = _compute_score(best_points)
    best_triples = _count_collinear_triples(best_points)
    
    iteration = 0
    while time.time() < deadline - 15:
        iteration += 1
        improved = False
        
        for idx in range(20):
            if time.time() > deadline - 15:
                break
            
            original = best_points[idx][:]
            occ = set(map(tuple, best_points))
            occ.discard(tuple(original))
            
            for trial in range(15):
                if time.time() > deadline - 15:
                    break
                
                new_pt = _random_valid_point(occ)
                new_points = [p[:] for p in best_points]
                new_points[idx] = new_pt
                
                if not _is_valid(new_points):
                    continue
                
                new_triples = _count_collinear_triples(new_points)
                if new_triples > best_triples:
                    new_score = _compute_score(new_points)
                    if new_score < best_score:
                        best_points = new_points
                        best_score = new_score
                        best_triples = new_triples
                        improved = True
                        break
                    elif new_score == best_score:
                        best_points = new_points
                        best_triples = new_triples
                        improved = True
                        break
        
        if not improved and iteration % 5 == 0:
            if time.time() > deadline - 25:
                break
            for _ in range(2):
                idx = random.randint(0, 19)
                old = best_points[idx][:]
                occ = set(map(tuple, best_points))
                occ.discard(tuple(old))
                new_pt = _random_valid_point(occ)
                best_points[idx] = new_pt
                occ.add(tuple(new_pt))
            
            if not _is_valid(best_points):
                continue
            
            new_score = _compute_score(best_points)
            if new_score < best_score:
                best_score = new_score
                best_triples = _count_collinear_triples(best_points)
    
    return best_points
