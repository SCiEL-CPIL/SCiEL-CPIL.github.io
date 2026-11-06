"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16

# EVOLVE-BLOCK-START
import itertools
import random
import math

def _sq_dist(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2

def _build_conflict_graph():
    """Build conflict graph: edge (i,j) if exists k such that {i,j,k} is isosceles."""
    points = [(x, y) for x in range(N) for y in range(N)]
    n = len(points)
    # conflict[i][j] = True if i and j cannot both be in a no-isosceles set
    conflict = [[False] * n for _ in range(n)]
    
    for i in range(n):
        for j in range(i + 1, n):
            di = _sq_dist(points[i], points[j])
            # Check if there exists k such that triangle (i,j,k) is isosceles
            # Isosceles means two of the three distances are equal
            # dist(i,j) = di
            # For each k: dist(i,k) == di OR dist(j,k) == di OR dist(i,k) == dist(j,k)
            found = False
            for k in range(n):
                if k == i or k == j:
                    continue
                dik = _sq_dist(points[i], points[k])
                djk = _sq_dist(points[j], points[k])
                if dik == di or djk == di or dik == djk:
                    found = True
                    break
            if found:
                conflict[i][j] = True
                conflict[j][i] = True
    
    return points, conflict

def _greedy_maximal(points, conflict):
    """Greedy construction in a given order."""
    chosen = []
    chosen_set = set()
    for i in range(len(points)):
        can_add = True
        for j in chosen_set:
            if conflict[i][j]:
                can_add = False
                break
        if can_add:
            chosen.append(points[i])
            chosen_set.add(i)
    return chosen

def _is_maximal(chosen_indices, points, conflict):
    """Check if the set is maximal (no point can be added)."""
    chosen_set = set(chosen_indices)
    for i in range(len(points)):
        if i in chosen_set:
            continue
        can_add = True
        for j in chosen_set:
            if conflict[i][j]:
                can_add = False
                break
        if can_add:
            return False
    return True

def _local_search(chosen_indices, points, conflict, max_iter=5000):
    """Try to find a smaller maximal set by removing points and re-adding."""
    current = set(chosen_indices)
    best = set(chosen_indices)
    best_size = len(current)
    
    for iteration in range(max_iter):
        # Try removing each point and see if we can maintain maximality
        improved = False
        for i in list(current):
            trial = current - {i}
            if not trial:
                continue
            # Check if trial is still an independent set (it is, since it's a subset)
            # Check if trial is maximal
            if _is_maximal(trial, points, conflict):
                current = trial
                if len(current) < best_size:
                    best = set(current)
                    best_size = len(current)
                improved = True
                break
        if not improved:
            break
    
    return list(best)

def construct():
    random.seed(42)
    points, conflict = _build_conflict_graph()
    n = len(points)
    
    # Try multiple random orderings and take the best
    best_set = None
    best_size = n + 1
    
    # First try greedy with natural order
    greedy = _greedy_maximal(points, conflict)
    greedy_indices = [i for i, p in enumerate(points) if p in greedy]
    current = _local_search(greedy_indices, points, conflict)
    if len(current) < best_size:
        best_size = len(current)
        best_set = [points[i] for i in current]
    
    # Try several random orderings
    for trial in range(20):
        order = list(range(n))
        random.shuffle(order)
        # Greedy in this order
        chosen = []
        chosen_set = set()
        for idx in order:
            can_add = True
            for j in chosen_set:
                if conflict[idx][j]:
                    can_add = False
                    break
            if can_add:
                chosen.append(idx)
                chosen_set.add(idx)
        
        current = _local_search(chosen, points, conflict)
        if len(current) < best_size:
            best_size = len(current)
            best_set = [points[i] for i in current]
    
    return best_set
# EVOLVE-BLOCK-END

