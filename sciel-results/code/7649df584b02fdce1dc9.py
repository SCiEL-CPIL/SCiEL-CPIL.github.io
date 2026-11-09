"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
from itertools import combinations

def _are_collinear(p1, p2, p3):
    return (p1[0]*(p2[1]-p3[1]) + p2[0]*(p3[1]-p1[1]) + p3[0]*(p1[1]-p2[1])) == 0

def _count_triples(points):
    """Count the number of collinear triples in the point set."""
    count = 0
    for i, j, k in combinations(range(len(points)), 3):
        if _are_collinear(points[i], points[j], points[k]):
            count += 1
    return count

def _has_four_collinear(points):
    """Check if any four points are collinear."""
    n = len(points)
    for i, j, k, l in combinations(range(n), 4):
        if _are_collinear(points[i], points[j], points[k]) and \
           _are_collinear(points[i], points[j], points[l]) and \
           _are_collinear(points[i], points[k], points[l]):
            return True
    return False

def _max_no_three_collinear(points):
    """Find the largest subset with no three collinear using branch and bound."""
    n = len(points)
    best = [0]
    
    # Precompute collinearity: collinear[i][j][k] = True if i,j,k are collinear
    collinear = [[[False]*n for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                if _are_collinear(points[i], points[j], points[k]):
                    collinear[i][j][k] = True
                    collinear[i][k][j] = True
                    collinear[j][i][k] = True
                    collinear[j][k][i] = True
                    collinear[k][i][j] = True
                    collinear[k][j][i] = True
    
    def is_valid_subset(subset):
        """Check if no three points in subset are collinear."""
        for i in range(len(subset)):
            for j in range(i+1, len(subset)):
                for k in range(j+1, len(subset)):
                    if collinear[subset[i]][subset[j]][subset[k]]:
                        return False
        return True
    
    def branch_and_bound(subset, remaining, depth):
        if not remaining:
            if len(subset) > best[0]:
                best[0] = len(subset)
            return
        
        # Upper bound: current size + remaining
        if len(subset) + len(remaining) <= best[0]:
            return
        
        for idx, point in enumerate(remaining):
            # Try adding this point
            new_subset = subset + [point]
            if is_valid_subset(new_subset):
                branch_and_bound(new_subset, remaining[idx+1:], depth+1)
            # Try not adding this point
            branch_and_bound(subset, remaining[idx+1:], depth+1)
    
    branch_and_bound([], list(range(n)), 0)
    return best[0]

def _find_valid_initial():
    """Find a valid initial 20-point set with no four collinear."""
    while True:
        points = []
        for _ in range(20):
            while True:
                x = random.randint(-100, 100)
                y = random.randint(-100, 100)
                p = [x, y]
                if p not in points:
                    break
            points.append(p)
        if not _has_four_collinear(points):
            return points

def _try_move_point(points, idx, deadline):
    """Try to move point at idx to a new position that improves the configuration."""
    current_triples = _count_triples(points)
    best_new_point = None
    best_new_triples = current_triples
    
    # Try some random new positions
    for _ in range(50):
        if time.time() > deadline:
            break
        x = random.randint(-100, 100)
        y = random.randint(-100, 100)
        new_p = [x, y]
        if new_p in points:
            continue
        
        new_points = points[:idx] + [new_p] + points[idx+1:]
        if _has_four_collinear(new_points):
            continue
        
        new_triples = _count_triples(new_points)
        if new_triples > best_new_triples:
            best_new_triples = new_triples
            best_new_point = new_p
    
    if best_new_point is not None:
        points[idx] = best_new_point
        return True
    return False

def construct():
    start_time = time.time()
    deadline = start_time + 100  # Leave 20s safety margin
    
    # Start with a valid initial configuration
    points = _find_valid_initial()
    best_points = points[:]
    best_score = _max_no_three_collinear(points)
    
    # Local search: try to improve by moving individual points
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for idx in range(20):
            if time.time() > deadline:
                break
            if _try_move_point(points, idx, deadline):
                improved = True
                score = _max_no_three_collinear(points)
                if score < best_score:
                    best_score = score
                    best_points = points[:]
    
    # If we still have time, try a few more random restarts
    while time.time() < deadline - 10:
        points = _find_valid_initial()
        for _ in range(20):
            if time.time() > deadline - 10:
                break
            for idx in range(20):
                if time.time() > deadline - 10:
                    break
                _try_move_point(points, idx, deadline - 10)
        
        score = _max_no_three_collinear(points)
        if score < best_score:
            best_score = score
            best_points = points[:]
    
    return best_points
# EVOLVE-BLOCK-END
