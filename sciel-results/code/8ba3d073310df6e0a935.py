"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _count_collinear_triples(points):
    """Count the number of collinear triples among the points."""
    n = len(points)
    triples = 0
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                x3, y3 = points[k]
                if (x2 - x1) * (y3 - y1) == (y2 - y1) * (x3 - x1):
                    triples += 1
    return triples

def _has_four_collinear(points):
    """Check if any four points are collinear."""
    n = len(points)
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                x3, y3 = points[k]
                for l in range(k+1, n):
                    x4, y4 = points[l]
                    if (x2-x1)*(y3-y1) == (y2-y1)*(x3-x1) and \
                       (x2-x1)*(y4-y1) == (y2-y1)*(x4-x1):
                        return True
    return False

def _max_3free_subset(points, deadline=None):
    """Branch and bound to find the largest subset with no three collinear."""
    n = len(points)
    best = [0]
    best_set = [set()]
    
    # Precompute collinearity: collinear[i][j][k] = True if i,j,k are collinear
    collinear = [[[False]*n for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            for k in range(j+1, n):
                x3, y3 = points[k]
                if (x2-x1)*(y3-y1) == (y2-y1)*(x3-x1):
                    collinear[i][j][k] = True
                    collinear[j][i][k] = True
                    collinear[k][i][j] = True
                    collinear[i][k][j] = True
                    collinear[j][k][i] = True
                    collinear[k][j][i] = True
    
    def is_valid_3free(subset_indices):
        """Check if a set of indices has no three collinear."""
        idx_list = sorted(subset_indices)
        for i in range(len(idx_list)):
            for j in range(i+1, len(idx_list)):
                for k in range(j+1, len(idx_list)):
                    if collinear[idx_list[i]][idx_list[j]][idx_list[k]]:
                        return False
        return True
    
    def dfs(idx, current, current_size):
        if deadline and time.time() > deadline:
            return
        if idx >= n:
            if current_size > best[0]:
                best[0] = current_size
                best_set[0] = set(current)
            return
        
        # Pruning: even if we take all remaining, can we beat best?
        if current_size + (n - idx) <= best[0]:
            return
        
        # Try excluding idx
        dfs(idx + 1, current, current_size)
        
        # Try including idx
        # Check if adding idx creates a collinear triple with any two in current
        can_add = True
        for i, ci in enumerate(current):
            for j in range(i+1, len(current)):
                cj = current[j]
                if collinear[idx][ci][cj]:
                    can_add = False
                    break
            if not can_add:
                break
        
        if can_add:
            current.append(idx)
            dfs(idx + 1, current, current_size + 1)
            current.pop()
    
    dfs(0, [], 0)
    return best[0]

def _generate_candidate(seed):
    """Generate a candidate set of 20 points with many collinear triples."""
    rng = random.Random(seed)
    
    # Strategy: use a structured pattern with lines of 3
    # Place points on a grid but with perturbations
    # Try to create many 3-point lines
    
    # Base grid: 5x4 grid
    base_points = []
    for row in range(4):
        for col in range(5):
            base_points.append([col * 20, row * 20])
    
    # Perturb to avoid 4-collinear while keeping many 3-collinear
    points = []
    for p in base_points:
        x = p[0] + rng.randint(-5, 5)
        y = p[1] + rng.randint(-5, 5)
        # Ensure within bounds
        x = max(-100, min(100, x))
        y = max(-100, min(100, y))
        points.append([x, y])
    
    # Remove duplicates
    seen = set()
    unique_points = []
    for p in points:
        key = tuple(p)
        if key not in seen:
            seen.add(key)
            unique_points.append(p)
    
    # If we lost points due to duplicates, add new ones
    while len(unique_points) < 20:
        x = rng.randint(-100, 100)
        y = rng.randint(-100, 100)
        key = (x, y)
        if key not in seen:
            seen.add(key)
            unique_points.append([x, y])
    
    return unique_points[:20]

def _local_search(points, deadline):
    """Improve a candidate via local search."""
    points = [p[:] for p in points]
    current_triples = _count_collinear_triples(points)
    
    while time.time() < deadline:
        improved = False
        for i in range(20):
            if time.time() > deadline:
                break
            orig = points[i][:]
            for _ in range(20):
                if time.time() > deadline:
                    break
                new_x = max(-100, min(100, orig[0] + random.randint(-3, 3)))
                new_y = max(-100, min(100, orig[1] + random.randint(-3, 3)))
                points[i] = [new_x, new_y]
                
                # Check validity
                if len(set(tuple(p) for p in points)) != 20:
                    continue
                if _has_four_collinear(points):
                    continue
                
                new_triples = _count_collinear_triples(points)
                if new_triples > current_triples:
                    current_triples = new_triples
                    improved = True
                    break
            if points[i] != orig and not improved:
                points[i] = orig
        
        if not improved:
            break
    
    return points

def construct():
    start_time = time.time()
    deadline = start_time + 120  # Safety margin before 150s limit
    
    best_points = None
    best_score = 20  # Worst case
    best_triples = -1
    
    # Generate and evaluate candidates
    for seed in range(1000):
        if time.time() > deadline - 30:  # Leave 30s for final evaluation
            break
        
        points = _generate_candidate(seed)
        
        # Check basic validity
        if len(set(tuple(p) for p in points)) != 20:
            continue
        if _has_four_collinear(points):
            continue
        
        triples = _count_collinear_triples(points)
        
        if triples > best_triples:
            best_triples = triples
            best_points = [p[:] for p in points]
        
        # Local search on promising candidates
        if triples > 50 and time.time() < deadline - 60:
            improved_points = _local_search(points, deadline - 30)
            if len(set(tuple(p) for p in improved_points)) == 20 and not _has_four_collinear(improved_points):
                improved_triples = _count_collinear_triples(improved_points)
                if improved_triples > best_triples:
                    best_triples = improved_triples
                    best_points = [p[:] for p in improved_points]
    
    # If we found good candidates, evaluate the best one's exact score
    if best_points is not None and time.time() < deadline - 5:
        score = _max_3free_subset(best_points, deadline - 2)
        if score < best_score:
            best_score = score
    
    # Return the best points found
    if best_points is not None:
        return best_points
    else:
        # Fallback: parabola
        return [[x, x * x] for x in range(-10, 10)]
# EVOLVE-BLOCK-END
