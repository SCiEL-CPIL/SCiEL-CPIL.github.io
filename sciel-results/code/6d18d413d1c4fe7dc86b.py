import time
import random
import itertools

def _collinear(p1, p2, p3):
    return (p2[0] - p1[0]) * (p3[1] - p1[1]) == (p2[1] - p1[1]) * (p3[0] - p1[0])

def _no_four_collinear(points):
    n = len(points)
    # Check all 4-subsets? No, check lines.
    # For each pair, count points on line.
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

def _calc_score(points):
    n = len(points)
    # Find all collinear triples
    triples = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if _collinear(points[i], points[j], points[k]):
                    triples.append((i, j, k))
    
    # If no triples, score is n
    if not triples:
        return n
        
    # Check subsets from n down to 1
    for size in range(n, 0, -1):
        for comb in itertools.combinations(range(n), size):
            comb_set = set(comb)
            valid = True
            for t in triples:
                if t[0] in comb_set and t[1] in comb_set and t[2] in comb_set:
                    valid = False
                    break
            if valid:
                return size
    return 0

def construct():
    deadline = time.time() + 110
    best_score = 20
    best_points = None
    
    # Initial random points
    points = []
    while len(points) < 20:
        p = [random.randint(-100, 100), random.randint(-100, 100)]
        if p not in points:
            points.append(p)
            
    # Repair if necessary
    while not _no_four_collinear(points):
        # Find a 4-collinear set and move one point
        # Simplified: just randomize one point
        idx = random.randint(0, 19)
        while True:
            new_p = [random.randint(-100, 100), random.randint(-100, 100)]
            if new_p not in points:
                points[idx] = new_p
                break
        if time.time() > deadline - 10:
            break

    if _no_four_collinear(points):
        score = _calc_score(points)
        best_score = score
        best_points = points[:]
    
    current_points = best_points
    current_score = best_score
    
    while time.time() < deadline - 5:
        idx = random.randint(0, 19)
        new_p = [random.randint(-100, 100), random.randint(-100, 100)]
        if new_p in current_points:
            continue
        new_points = current_points[:]
        new_points[idx] = new_p
        
        if _no_four_collinear(new_points):
            new_score = _calc_score(new_points)
            if new_score < current_score:
                current_points = new_points
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_points = new_points[:]
            elif new_score == current_score:
                # Random walk
                if random.random() < 0.5:
                    current_points = new_points
        # else: invalid, ignore

    return best_points
