import time
import random

def are_collinear(p1, p2, p3):
    return (p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1])) == 0

def check_no_four_collinear(points):
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            count = 2
            for k in range(n):
                if k == i or k == j:
                    continue
                if are_collinear(points[i], points[j], points[k]):
                    count += 1
            if count >= 4:
                return False
    return True

def get_triples(points):
    n = len(points)
    triples = set()
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if are_collinear(points[i], points[j], points[k]):
                    triples.add((i, j, k))
    return triples

def max_independent_set_size(points, triples):
    n = len(points)
    best = 0
    
    # Map point index to list of (u, v) pairs that form a triple with it
    # If we pick i, we cannot pick both u and v.
    # This is a constraint on the remaining set.
    
    def solve(available_mask, forbidden_pairs, current_size):
        nonlocal best
        
        if current_size + bin(available_mask).count('1') <= best:
            return
            
        if available_mask == 0:
            if current_size > best:
                best = current_size
            return

        v = (available_mask & -available_mask).bit_length() - 1
        
        # Exclude v
        solve(available_mask & ~(1 << v), forbidden_pairs, current_size)
        
        # Include v
        new_available = available_mask & ~(1 << v)
        new_forbidden = set(forbidden_pairs)
        
        for (a, b, c) in triples:
            if a == v:
                new_forbidden.add((b, c))
            elif b == v:
                new_forbidden.add((a, c))
            elif c == v:
                new_forbidden.add((a, b))
                
        solve(new_available, new_forbidden, current_size + 1)

    solve((1 << n) - 1, set(), 0)
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    # Generate candidate pool from a small grid to increase triple density
    grid = []
    for x in range(-10, 11):
        for y in range(-10, 11):
            grid.append([x, y])
            
    random.shuffle(grid)
    
    points = []
    used = set()
    for p in grid:
        if (p[0], p[1]) not in used:
            points.append(p[:])
            used.add((p[0], p[1]))
            if len(points) == 20:
                break
                
    # Ensure validity
    while not check_no_four_collinear(points):
        # Perturb
        idx = random.randint(0, 19)
        # Pick a new point from grid not in used
        new_p = None
        for p in grid:
            if (p[0], p[1]) not in used:
                new_p = p
                break
        if new_p is None:
             # If grid exhausted, pick random
             new_p = [random.randint(-100, 100), random.randint(-100, 100)]
             
        used.discard(tuple(points[idx]))
        points[idx] = new_p[:]
        used.add((new_p[0], new_p[1]))
        
    triples = get_triples(points)
    current_score = max_independent_set_size(points, triples)
    best_score = current_score
    best_points = [p[:] for p in points]
    
    # Hill Climbing
    while time.time() < deadline:
        idx = random.randint(0, 19)
        # Pick a new point
        new_p = [random.randint(-100, 100), random.randint(-100, 100)]
        
        if (new_p[0], new_p[1]) in used:
            continue
            
        old_point = points[idx]
        old_tuple = tuple(old_point)
        points[idx] = new_p
        used.discard(old_tuple)
        used.add((new_p[0], new_p[1]))
        
        if check_no_four_collinear(points):
            triples = get_triples(points)
            new_score = max_independent_set_size(points, triples)
            
            if new_score < best_score:
                best_score = new_score
                best_points = [p[:] for p in points]
            else:
                points[idx] = old_point
                used.discard((new_p[0], new_p[1]))
                used.add(old_tuple)
        else:
            points[idx] = old_point
            used.discard((new_p[0], new_p[1]))
            used.add(old_tuple)
            
    return best_points
