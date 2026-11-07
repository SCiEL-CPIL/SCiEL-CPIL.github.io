"""Rewritten program: structural change to a two-phase search over value sets and edge selections."""
# EVOLVE-BLOCK-START
import time
import random
import math
from itertools import combinations

_DEADLINE = 85.0  # safety margin before 90s kill

def _raw_score(values, edges):
    """Compute max(distinct sums, distinct products) over edges."""
    if not edges:
        return 0
    sums = set()
    prods = set()
    for i, j in edges:
        s = values[i] + values[j]
        p = values[i] * values[j]
        sums.add(s)
        prods.add(p)
    return max(len(sums), len(prods))

def _greedy_edges_for_values(values, target_k):
    """
    Given a list of 14 distinct positive integers, select 35 edges to minimize
    max(distinct sums, distinct products). Use a greedy heuristic:
    - Start with empty edge set.
    - At each step, add the edge that causes the smallest increase in
      max(distinct_sums, distinct_products), preferring edges that reuse
      existing sums/products.
    - If we can't reach target_k with greedy, just return the best found.
    """
    n = len(values)
    all_edges = list(combinations(range(n), 2))
    
    # Precompute sum and product for each edge
    edge_data = []
    for i, j in all_edges:
        s = values[i] + values[j]
        p = values[i] * values[j]
        edge_data.append((i, j, s, p))
    
    best_edges = []
    best_score = float('inf')
    
    # Greedy: repeatedly pick the edge that minimizes the resulting score
    used = set()
    sums_set = set()
    prods_set = set()
    
    for _ in range(35):
        best_edge = None
        best_new_score = float('inf')
        best_new_sums = None
        best_new_prods = None
        
        for (i, j, s, p) in edge_data:
            if (i, j) in used:
                continue
            new_sums = sums_set | {s}
            new_prods = prods_set | {p}
            score = max(len(new_sums), len(new_prods))
            if score < best_new_score:
                best_new_score = score
                best_edge = (i, j)
                best_new_sums = new_sums
                best_new_prods = new_prods
                # Early exit if we find a score of 1 (impossible unless all same)
                if best_new_score <= len(sums_set) and best_new_score <= len(prods_set):
                    break
        
        if best_edge is None:
            break
        
        used.add(best_edge)
        sums_set = best_new_sums
        prods_set = best_new_prods
        best_edges.append(list(best_edge))
    
    if len(best_edges) < 35:
        # Fill remaining with any unused edges
        for (i, j, s, p) in edge_data:
            if len(best_edges) >= 35:
                break
            if (i, j) not in used:
                best_edges.append([i, j])
                used.add((i, j))
    
    return best_edges[:35]

def _local_search_edges(values, edges, max_iters=20000):
    """
    Local search: try swapping one edge for another to reduce the score.
    """
    n = len(values)
    all_edges = list(combinations(range(n), 2))
    edge_set = set(tuple(e) for e in edges)
    
    def compute_score(edge_list):
        sums = set()
        prods = set()
        for i, j in edge_list:
            sums.add(values[i] + values[j])
            prods.add(values[i] * values[j])
        return max(len(sums), len(prods))
    
    current_score = compute_score(edges)
    best_edges = list(edges)
    best_score = current_score
    no_improve = 0
    iters = 0
    
    while no_improve < 500 and iters < max_iters:
        iters += 1
        # Pick a random edge to remove
        idx = random.randint(0, len(best_edges) - 1)
        removed = tuple(best_edges[idx])
        
        # Try adding a random unused edge
        candidates = [e for e in all_edges if e not in edge_set and e != removed]
        if not candidates:
            no_improve += 1
            continue
        add_edge = random.choice(candidates)
        
        new_edges = list(best_edges)
        new_edges[idx] = list(add_edge)
        new_score = compute_score(new_edges)
        
        if new_score < best_score:
            best_score = new_score
            best_edges = new_edges
            edge_set.remove(removed)
            edge_set.add(add_edge)
            no_improve = 0
        else:
            no_improve += 1
    
    return best_edges, best_score

def _try_value_set(values, time_left):
    """
    For a given value set, run greedy + local search to find good edges.
    Returns (edges, score).
    """
    start = time.time()
    edges = _greedy_edges_for_values(values, 100)
    score = _raw_score(values, edges)
    
    # Local search with time check
    remaining = time_left - (time.time() - start)
    if remaining > 0.5:
        edges, score = _local_search_edges(values, edges, max_iters=10000)
    
    return edges, score

def _generate_candidate_value_sets():
    """
    Generate diverse candidate value sets of 14 distinct positive integers <= 10000.
    """
    candidates = []
    
    # 1. Powers of 2: 2^0 to 2^13
    candidates.append([2**i for i in range(14)])
    
    # 2. Small consecutive: 1..14
    candidates.append(list(range(1, 15)))
    
    # 3. Geometric-ish: roughly 1.5^i
    vals = [1]
    for i in range(13):
        vals.append(min(10000, int(vals[-1] * 1.5) + 1))
    # Ensure distinct and <=10000
    vals = sorted(set(vals))
    while len(vals) < 14:
        vals.append(vals[-1] + 1)
    vals = vals[:14]
    candidates.append(vals)
    
    # 4. Primes
    primes = []
    x = 2
    while len(primes) < 14:
        if all(x % p != 0 for p in primes):
            primes.append(x)
        x += 1
    candidates.append(primes)
    
    # 5. Random sets with structure: mix of small and large
    random.seed(42)
    for seed in range(5):
        random.seed(seed)
        vals = set()
        # Pick 7 small, 7 large
        small_pool = list(range(1, 50))
        large_pool = list(range(500, 10001))
        vals.update(random.sample(small_pool, 7))
        vals.update(random.sample(large_pool, 7))
        if len(vals) == 14:
            candidates.append(sorted(vals))
    
    # 6. Arithmetic progression with various starts/steps
    for start, step in [(1, 10), (10, 50), (100, 700), (5, 30), (20, 600)]:
        vals = [start + i * step for i in range(14)]
        if all(1 <= v <= 10000 for v in vals) and len(set(vals)) == 14:
            candidates.append(vals)
    
    # 7. Two-cluster: 7 near 10, 7 near 9000
    random.seed(99)
    cluster1 = random.sample(range(1, 100), 7)
    cluster2 = random.sample(range(8000, 10001), 7)
    candidates.append(sorted(cluster1 + cluster2))
    
    # 8. Fibonacci-like
    fib = [1, 1]
    for i in range(12):
        fib.append(fib[-1] + fib[-2])
    fib = fib[:14]
    # Ensure distinct
    if len(set(fib)) == 14 and all(v <= 10000 for v in fib):
        candidates.append(fib)
    
    # 9. Exponential base 3
    vals = [3**i for i in range(14)]
    if all(v <= 10000 for v in vals) and len(set(vals)) == 14:
        candidates.append(vals)
    
    # 10. Mixed: some small primes, some large composites
    random.seed(123)
    small_primes = [2, 3, 5, 7, 11, 13, 17]
    large_comps = random.sample(range(1000, 10001), 7)
    candidates.append(sorted(small_primes + large_comps))
    
    return candidates

def construct():
    start_time = time.time()
    
    def time_up():
        return (time.time() - start_time) > _DEADLINE
    
    # Generate candidate value sets
    candidates = _generate_candidate_value_sets()
    
    best_edges = None
    best_score = float('inf')
    best_values = None
    
    # Shuffle candidates for diversity
    random.seed(7)
    random.shuffle(candidates)
    
    for vals in candidates:
        if time_up():
            break
        
        # Validate
        if len(vals) != 14:
            continue
        if len(set(vals)) != 14:
            continue
        if not all(isinstance(v, int) and 1 <= v <= 10000 for v in vals):
            continue
        
        edges, score = _try_value_set(vals, _DEADLINE - (time.time() - start_time))
        
        if score < best_score:
            best_score = score
            best_edges = edges
            best_values = list(vals)
        
        # Check time
        if time_up():
            break
    
    # If we didn't find anything (shouldn't happen), use default
    if best_values is None:
        best_values = list(range(1, 15))
        best_edges = [list(e) for e in list(combinations(range(14), 2))[:35]]
    
    # Final validation
    assert len(best_values) == 14
    assert len(set(best_values)) == 14
    assert all(isinstance(v, int) and 1 <= v <= 10000 for v in best_values)
    assert len(best_edges) == 35
    
    edge_tuples = set()
    for e in best_edges:
        assert len(e) == 2
        i, j = e
        assert isinstance(i, int) and isinstance(j, int)
        assert 0 <= i < 14 and 0 <= j < 14
        assert i != j
        t = (min(i, j), max(i, j))
        assert t not in edge_tuples
        edge_tuples.add(t)
    
    # Convert to pure Python types
    result_values = [int(v) for v in best_values]
    result_edges = [[int(i), int(j)] for i, j in best_edges]
    
    return {'values': result_values, 'edges': result_edges}
# EVOLVE-BLOCK-END
