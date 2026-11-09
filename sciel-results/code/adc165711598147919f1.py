import random
import time

def is_no4_collinear(points):
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            dx = points[j][0] - points[i][0]
            dy = points[j][1] - points[i][1]
            count = 2
            for k in range(n):
                if k == i or k == j:
                    continue
                if (points[j][0] - points[i][0]) * (points[k][1] - points[i][1]) == (points[j][1] - points[i][1]) * (points[k][0] - points[i][0]):
                    count += 1
            if count >= 4:
                return False
    return True

def compute_score(points):
    n = len(points)
    triples = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if (points[j][0] - points[i][0]) * (points[k][1] - points[i][1]) == (points[j][1] - points[i][1]) * (points[k][0] - points[i][0]):
                    triples.append((i, j, k))
    
    triple_lookup = {}
    for i in range(n):
        triple_lookup[i] = {}
    for (i, j, k) in triples:
        triple_lookup[i][j] = k
        triple_lookup[i][k] = j
        triple_lookup[j][i] = k
        triple_lookup[j][k] = i
        triple_lookup[k][i] = j
        triple_lookup[k][j] = i
        
    best = 0
    
    def bb(used, available, count):
        nonlocal best
        if count + bin(available).count('1') <= best:
            return
        
        if available == 0:
            if count > best:
                best = count
            return
        
        # Heuristic: Pick a point from available.
        # To improve pruning, pick a point with high degree in the remaining hypergraph?
        # For now, pick the first one.
        i_bit = available & -available
        i = i_bit.bit_length() - 1
        
        # Option 1: Skip i
        bb(used, available ^ i_bit, count)
        
        # Option 2: Include i
        new_used = used | i_bit
        new_available = available ^ i_bit
        
        # Prune new_available:
        # If j is in new_available, and there exists k in used such that i, j, k are collinear,
        # then j cannot be added (because i and k are already in the set).
        temp_avail = new_available
        while temp_avail:
            j_bit = temp_avail & -temp_avail
            j = j_bit.bit_length() - 1
            if j in triple_lookup[i]:
                k = triple_lookup[i][j]
                if used & (1 << k):
                    new_available &= ~j_bit
            temp_avail ^= j_bit
            
        bb(new_used, new_available, count + 1)
        
    bb(0, (1 << n) - 1, 0)
    return best

def construct():
    start_time = time.time()
    deadline = 115 # 115 seconds
    
    # Initial configuration: Random valid points
    def random_valid_points():
        while True:
            pts = []
            seen = set()
            while len(pts) < 20:
                x = random.randint(-15, 15)
                y = random.randint(-15, 15)
                if (x, y) not in seen:
                    seen.add((x, y))
                    pts.append([x, y])
            if is_no4_collinear(pts):
                return pts

    best_points = random_valid_points()
    best_score = compute_score(best_points)
    
    # Local Search
    while time.time() < deadline:
        # Pick a random index to swap
        idx = random.randint(0, 19)
        old_point = best_points[idx]
        
        # Generate a new random point
        # Use a smaller range to increase collinearity chances
        new_point = [random.randint(-15, 15), random.randint(-15, 15)]
        
        # Check if new point is distinct
        if any(p[0] == new_point[0] and p[1] == new_point[1] for p in best_points):
            continue
            
        candidate = best_points.copy()
        candidate[idx] = new_point
        
        if not is_no4_collinear(candidate):
            continue
            
        score = compute_score(candidate)
        
        if score < best_score:
            best_score = score
            best_points = candidate
        elif score == best_score and random.random() < 0.1:
            # Simulated annealing style: accept equal score with low prob to escape local minima
            best_points = candidate

    return best_points
