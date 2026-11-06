import time
import random
from itertools import combinations

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

def _max_indep_bnb(points, triples, time_limit=2.0):
    start_time = time.time()
    n = len(points)
    best_size = 0
    
    point_triples = [[] for _ in range(n)]
    for idx, (i, j, k) in enumerate(triples):
        point_triples[i].append(idx)
        point_triples[j].append(idx)
        point_triples[k].append(idx)
        
    degree = [len(pt) for pt in point_triples]
    order = sorted(range(n), key=lambda x: -degree[x])
    orig_idx = [0] * n
    for new_idx, old_idx in enumerate(order):
        orig_idx[new_idx] = old_idx
    
    def bnb(idx, selected_mask, cur_size):
        nonlocal best_size
        if time.time() - start_time > time_limit:
            return
        if cur_size + (n - idx) <= best_size:
            return
        if idx == n:
            if cur_size > best_size:
                best_size = cur_size
            return
            
        p_orig = orig_idx[idx]
        new_mask = selected_mask | (1 << p_orig)
        valid = True
        for t_idx in point_triples[p_orig]:
            i, j, k = triples[t_idx]
            if (new_mask & (1 << i)) and (new_mask & (1 << j)) and (new_mask & (1 << k)):
                valid = False
                break
                
        if valid:
            bnb(idx + 1, new_mask, cur_size + 1)
        bnb(idx + 1, selected_mask, cur_size)

    bnb(0, 0, 0)
    return best_size

def _score(points, time_limit=2.0):
    triples = _precompute_triples(points)
    return _max_indep_bnb(points, triples, time_limit)

def _count_triples_involving(points, idx):
    """Count collinear triples involving point idx."""
    count = 0
    n = len(points)
    for i in range(n):
        if i == idx:
            continue
        for j in range(i + 1, n):
            if j == idx:
                continue
            if _collinear(points[idx], points[i], points[j]):
                count += 1
    return count

def _greedy_construct():
    """
    Greedy construction that maximizes collinear triples.
    Start with a valid base, then add points one by one,
    choosing positions that create the most new collinear triples.
    """
    random.seed(42)
    points = []
    
    # Start with a few points on a line to seed triples
    # Place 3 points collinearly
    base_points = [[-30, 0], [0, 0], [30, 0]]
    points = [p[:] for p in base_points]
    
    # Add remaining 17 points greedily
    for _ in range(17):
        best_pos = None
        best_triple_gain = -1
        best_total_triples = -1
        
        # Try many random candidate positions
        for _ in range(500):
            cx = random.randint(-100, 100)
            cy = random.randint(-100, 100)
            
            # Check distinctness
            if [cx, cy] in points:
                continue
            
            candidate = [p[:] for p in points]
            candidate.append([cx, cy])
            
            # Check no four collinear
            if not _no_four_collinear(candidate):
                continue
            
            # Count total triples
            triples = _precompute_triples(candidate)
            total = len(triples)
            
            if total > best_total_triples:
                best_total_triples = total
                best_pos = [cx, cy]
                best_triple_gain = total
        
        if best_pos is not None:
            points.append(best_pos)
        else:
            # Fallback: random position
            while True:
                cx = random.randint(-100, 100)
                cy = random.randint(-100, 100)
                if [cx, cy] not in points:
                    candidate = [p[:] for p in points]
                    candidate.append([cx, cy])
                    if _no_four_collinear(candidate):
                        points.append([cx, cy])
                        break
    
    return points

def _simulated_annealing(points, deadline):
    """
    Simulated annealing to minimize the max independent set size.
    Uses fast incremental scoring via triple counting.
    """
    current_points = [p[:] for p in points]
    current_score = _score(current_points, time_limit=2.0)
    best_points = [p[:] for p in current_points]
    best_score = current_score
    
    T = 10.0
    T_min = 0.1
    cooling = 0.995
    
    while time.time() < deadline - 15 and T > T_min:
        # Pick a point to move
        idx = random.randint(0, 19)
        
        # Try several candidate moves
        for _ in range(3):
            new_x = random.randint(-100, 100)
            new_y = random.randint(-100, 100)
            
            if [new_x, new_y] in current_points:
                continue
            
            candidate = [p[:] for p in current_points]
            candidate[idx] = [new_x, new_y]
            
            if not _no_four_collinear(candidate):
                continue
            
            # Fast score: use triple count as proxy, then exact if promising
            cand_triples = _precompute_triples(candidate)
            cand_triple_count = len(cand_triples)
            
            # If triple count is significantly higher, do exact scoring
            if cand_triple_count >= 5 or random.random() < 0.3:
                cand_score = _score(candidate, time_limit=1.0)
            else:
                # Approximate: more triples generally means smaller independent set
                # Use a rough estimate
                covered = set()
                for i, j, k in cand_triples:
                    covered.add(i)
                    covered.add(j)
                    covered.add(k)
                cand_score = 20 - len(covered) + 5  # rough estimate
            
            # Simulated annealing acceptance
            if cand_score < current_score:
                current_score = cand_score
                current_points = candidate
                if cand_score < best_score:
                    best_score = cand_score
                    best_points = [p[:] for p in current_points]
            else:
                # Accept with probability exp(-delta/T)
                delta = cand_score - current_score
                if random.random() < pow(2.718281828, -delta / T):
                    current_score = cand_score
                    current_points = candidate
            
            T *= cooling
            if time.time() > deadline - 15:
                break
    
    return best_points, best_score

def _local_search_refine(points, deadline):
    """
    Local search: try swapping pairs of points to reduce independent set size.
    """
    current_points = [p[:] for p in points]
    current_score = _score(current_points, time_limit=2.0)
    best_points = [p[:] for p in current_points]
    best_score = current_score
    
    no_improve_count = 0
    
    while time.time() < deadline - 10 and no_improve_count < 50:
        # Try swapping two points
        i, j = random.sample(range(20), 2)
        
        # Generate candidate swap: move point i to a new position near point j
        # or just swap their positions
        candidate = [p[:] for p in current_points]
        
        # Try moving point i to a random position
        for _ in range(2):
            new_x = random.randint(-100, 100)
            new_y = random.randint(-100, 100)
            
            if [new_x, new_y] in candidate:
                continue
            
            candidate = [p[:] for p in current_points]
            candidate[i] = [new_x, new_y]
            
            if not _no_four_collinear(candidate):
                continue
            
            cand_score = _score(candidate, time_limit=1.0)
            
            if cand_score < current_score:
                current_score = cand_score
                current_points = candidate
                if cand_score < best_score:
                    best_score = cand_score
                    best_points = [p[:] for p in current_points]
                no_improve_count = 0
                break
        else:
            no_improve_count += 1
    
    return best_points, best_score

def construct():
    deadline = time.time() + 130
    random.seed(123)
    
    # Phase 1: Greedy construction to create many collinear triples
    points = _greedy_construct()
    
    # Ensure validity
    if len(points) != 20 or not _no_four_collinear(points):
        # Fallback
        points = [[i, i*i] for i in range(-10, 10)]
    
    # Phase 2: Simulated annealing
    best_points, best_score = _simulated_annealing(points, deadline)
    
    # Phase 3: Local search refinement
    best_points, best_score = _local_search_refine(best_points, deadline)
    
    # Final validation
    if len(best_points) != 20 or not _no_four_collinear(best_points):
        # Return a known valid configuration
        return [[i, i*i] for i in range(-10, 10)]
    
    return best_points
