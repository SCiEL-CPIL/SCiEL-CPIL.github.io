"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START
def _collinear(p1, p2, p3):
    (x1, y1), (x2, y2), (x3, y3) = p1, p2, p3
    return (x2 - x1) * (y3 - y1) == (x3 - x1) * (y2 - y1)

def _no_four_collinear(points):
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            count = 2
            for k in range(n):
                if k != i and k != j:
                    if _collinear(points[i], points[j], points[k]):
                        count += 1
                        if count >= 4:
                            return False
    return True

def _precompute_triples(points):
    n = len(points)
    triples = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if _collinear(points[i], points[j], points[k]):
                    triples.append((i, j, k))
    return triples

def _max_indep_bnb(n, triples, time_limit=4.0):
    start_time = time.time()
    best_size = 0
    triple_masks = [(1 << i) | (1 << j) | (1 << k) for (i, j, k) in triples]
    degree = [0] * n
    for (i, j, k) in triples:
        degree[i] += 1
        degree[j] += 1
        degree[k] += 1
    order = sorted(range(n), key=lambda x: -degree[x])
    point_triples = [[] for _ in range(n)]
    for tm in triple_masks:
        for p in range(n):
            if tm & (1 << p):
                point_triples[p].append(tm)
    
    def bnb(idx, cur_mask, cur_size):
        nonlocal best_size
        if time.time() - start_time > time_limit:
            return
        if cur_size + (n - idx) <= best_size:
            return
        if idx == n:
            if cur_size > best_size:
                best_size = cur_size
            return
        p = order[idx]
        new_mask = cur_mask | (1 << p)
        valid = True
        for tm in point_triples[p]:
            if (new_mask & tm) == tm:
                valid = False
                break
        if valid:
            bnb(idx + 1, new_mask, cur_size + 1)
        bnb(idx + 1, cur_mask, cur_size)
    
    bnb(0, 0, 0)
    return best_size

def _score(points):
    n = len(points)
    triples = _precompute_triples(points)
    return _max_indep_bnb(n, triples, time_limit=4.0)

def _initial_config():
    """Grid-based initial config with many 3-collinear lines."""
    random.seed(42)
    points = []
    for i in range(5):
        for j in range(4):
            x = i * 18 + random.randint(-2, 2)
            y = j * 18 + random.randint(-2, 2)
            points.append([x, y])
    if _no_four_collinear(points):
        return points
    # Fallback
    return [[x, x * x] for x in range(-10, 10)]

def construct():
    deadline = time.time() + 110
    random.seed(123)
    
    points = _initial_config()
    best_points = [p[:] for p in points]
    best_score = _score(best_points)
    
    # Local search: replace one point at a time
    while time.time() < deadline - 15:
        i = random.randint(0, 19)
        new_x = random.randint(-100, 100)
        new_y = random.randint(-100, 100)
        if [new_x, new_y] in best_points:
            continue
        
        candidate = [p[:] for p in best_points]
        candidate[i] = [new_x, new_y]
        
        if not _no_four_collinear(candidate):
            continue
        
        new_score = _score(candidate)
        if new_score < best_score:
            best_score = new_score
            best_points = candidate
    
    return best_points
# EVOLVE-BLOCK-END
