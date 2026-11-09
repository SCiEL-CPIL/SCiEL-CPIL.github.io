"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

_START_TIME = time.time()
_DEADLINE = _START_TIME + 85  # 5s safety margin before 90s kill

def _score(values, edges):
    """Compute raw score: max(distinct sums, distinct products)."""
    sums = set()
    prods = set()
    for u, v in edges:
        a = values[u]
        b = values[v]
        sums.add(a + b)
        prods.add(a * b)
    return max(len(sums), len(prods))

def _incremental_score_change(values, edges_set, remove_edge, add_edge):
    """
    Compute the change in score when removing one edge and adding another.
    Returns (new_score, new_sums_set, new_prods_set) or None if infeasible.
    """
    u1, v1 = remove_edge
    u2, v2 = add_edge
    
    # Current sums and products
    sums = set()
    prods = set()
    for u, v in edges_set:
        a = values[u]
        b = values[v]
        sums.add(a + b)
        prods.add(a * b)
    
    # Remove the old edge's contribution
    old_sum = values[u1] + values[v1]
    old_prod = values[u1] * values[v1]
    
    # Count how many edges have this sum/product
    sum_count = {s: 0 for s in sums}
    prod_count = {p: 0 for p in prods}
    for u, v in edges_set:
        a = values[u]
        b = values[v]
        s = a + b
        p = a * b
        sum_count[s] += 1
        prod_count[p] += 1
    
    # Remove old edge
    sum_count[old_sum] -= 1
    prod_count[old_prod] -= 1
    if sum_count[old_sum] == 0:
        del sum_count[old_sum]
    if prod_count[old_prod] == 0:
        del prod_count[old_prod]
    
    # Add new edge
    new_sum = values[u2] + values[v2]
    new_prod = values[u2] * values[v2]
    
    if new_sum in sum_count:
        sum_count[new_sum] += 1
    else:
        sum_count[new_sum] = 1
    if new_prod in prod_count:
        prod_count[new_prod] += 1
    else:
        prod_count[new_prod] = 1
    
    new_score = max(len(sum_count), len(prod_count))
    return new_score

def _hill_climb_edges(values, m, n, max_iter=50000):
    """Hill climbing with incremental scoring for edge selection."""
    all_edges = list(combinations(range(n), 2))
    random.shuffle(all_edges)
    
    # Start with a random set of m edges
    edges = set(random.sample(all_edges, m))
    
    current_score = _score(values, list(edges))
    best_score = current_score
    best_edges = set(edges)
    no_improve_count = 0
    
    for iteration in range(max_iter):
        if time.time() > _DEADLINE:
            break
        if current_score <= 2:
            break
        
        # Try a random swap
        u, v = random.choice(list(edges))
        candidates = [e for e in all_edges if e not in edges]
        if not candidates:
            break
        x, y = random.choice(candidates)
        
        new_score = _incremental_score_change(values, edges, (u, v), (x, y))
        
        if new_score < current_score:
            edges.discard((u, v))
            edges.add((x, y))
            current_score = new_score
            no_improve_count = 0
            if current_score < best_score:
                best_score = current_score
                best_edges = set(edges)
        else:
            no_improve_count += 1
            if no_improve_count > 2000:
                break
    
    return best_score, [list(e) for e in best_edges]

def _try_value_set(values, m, n):
    """Try a value set with hill climbing."""
    if time.time() > _DEADLINE:
        return None
    
    best_score = m  # worst case
    best_edges = None
    
    # Multiple restarts
    for restart in range(5):
        if time.time() > _DEADLINE:
            break
        
        score, edges = _hill_climb_edges(values, m, n, max_iter=10000)
        
        if score < best_score:
            best_score = score
            best_edges = edges
            if best_score <= 2:
                break
    
    if best_edges is None:
        return None
    
    return best_score, values, best_edges

def construct():
    n = 14
    m = 35
    max_val = 10000
    
    best_score = m  # worst case: all distinct
    best_result = None
    
    # Candidate value sets to try
    candidates = []
    
    # 1. Small consecutive integers
    candidates.append(list(range(1, n + 1)))
    
    # 2. Small even numbers
    candidates.append(list(range(2, 2 * n + 1, 2)))
    
    # 3. Highly composite-like small numbers
    candidates.append([1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20])
    
    # 4. Numbers with many divisors
    candidates.append([1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 12, 15, 16, 18])
    # Fix duplicates
    candidates[-1] = [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20]
    
    # 5. Mixed small values with lots of collisions
    candidates.append([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
    
    # 6. Try some random small value sets
    random.seed(42)
    for _ in range(30):
        vals = sorted(random.sample(range(1, 60), n))
        candidates.append(vals)
    
    # 7. Arithmetic progressions
    for start in range(1, 6):
        for step in range(1, 4):
            vals = [start + i * step for i in range(n)]
            if max(vals) <= max_val:
                candidates.append(vals)
    
    # 8. Numbers that are multiples of small numbers
    for base in [2, 3, 4, 5, 6]:
        vals = [base * (i + 1) for i in range(n)]
        if max(vals) <= max_val and len(set(vals)) == n:
            candidates.append(vals)
    
    # 9. Mix of small and medium numbers
    for _ in range(20):
        small = random.sample(range(1, 20), 7)
        medium = random.sample(range(20, 100), 7)
        vals = sorted(small + medium)
        if len(set(vals)) == n and max(vals) <= max_val:
            candidates.append(vals)
    
    for vals in candidates:
        if time.time() > _DEADLINE:
            break
        # Validate
        if len(set(vals)) != n or min(vals) < 1 or max(vals) > max_val:
            continue
        # Ensure all are Python int
        if not all(isinstance(v, int) and not isinstance(v, bool) for v in vals):
            continue
        
        result = _try_value_set(vals, m, n)
        if result is None:
            continue
        score, values, edges = result
        if score < best_score:
            best_score = score
            best_result = (values, edges)
            if best_score <= 2:
                break
    
    if best_result is None:
        # Fallback
        values = list(range(1, n + 1))
        edges = [list(e) for e in list(combinations(range(n), 2))[:m]]
    else:
        values, edges = best_result
    
    # Ensure types are correct
    values = [int(v) for v in values]
    edges = [[int(u), int(v)] for u, v in edges]
    
    return {'values': values, 'edges': edges}
# EVOLVE-BLOCK-END
