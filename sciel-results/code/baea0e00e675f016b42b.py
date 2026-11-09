import time
import random
import math
from itertools import combinations

# Deadline guard: 90s limit, use 85s for safety
START_TIME = time.time()
DEADLINE = START_TIME + 85.0

def _check_time():
    return time.time() > DEADLINE

def _score(values, edges):
    """Calculate the raw score: max(distinct sums, distinct products)."""
    sums = set()
    prods = set()
    for u, v in edges:
        a = values[u]
        b = values[v]
        sums.add(a + b)
        prods.add(a * b)
    return max(len(sums), len(prods))

def _optimize_edges(values, n_edges=35, max_iters=5000):
    """
    Find the best set of n_edges edges for a given set of values
    using Simulated Annealing.
    """
    n = len(values)
    all_edges = list(combinations(range(n), 2))
    num_all = len(all_edges)
    
    # Precompute sums and products for all possible edges
    edge_props = {}
    for e in all_edges:
        u, v = e
        edge_props[e] = (values[u] + values[v], values[u] * values[v])
        
    # Initialize with a greedy approach
    # Sort edges by how 'common' their sum and product are
    sum_counts = {}
    prod_counts = {}
    for s, p in edge_props.values():
        sum_counts[s] = sum_counts.get(s, 0) + 1
        prod_counts[p] = prod_counts.get(p, 0) + 1
        
    def edge_priority(e):
        s, p = edge_props[e]
        return (sum_counts[s], prod_counts[p])
        
    sorted_edges = sorted(all_edges, key=edge_priority, reverse=True)
    
    current_edges = set(sorted_edges[:n_edges])
    current_score = _score(values, list(current_edges))
    
    best_edges = set(current_edges)
    best_score = current_score
    
    # Simulated Annealing parameters
    T = 10.0
    cooling = 0.995
    min_T = 0.1
    
    iter_count = 0
    while T > min_T and iter_count < max_iters:
        if _check_time():
            break
            
        # Pick a random edge to remove
        u, v = random.choice(list(current_edges))
        
        # Pick a random edge to add (not currently in set)
        # Efficiently pick from non-edges
        # For small n, just sampling is fine
        cand_e = None
        attempts = 0
        while cand_e is None or cand_e in current_edges:
            cand_e = random.choice(all_edges)
            attempts += 1
            if attempts > 100:
                break
                
        if cand_e is None or cand_e in current_edges:
            continue
            
        # Calculate new score
        # We can optimize this by tracking sets, but recalculating is fast for n=14
        # Let's try a faster incremental update if possible, or just recalc if it's cheap enough
        # Recalculation: 35 edges * 2 ops = 70 ops. Very cheap.
        
        new_edges = list(current_edges)
        new_edges.remove((u, v))
        new_edges.append(cand_e)
        
        new_score = _score(values, new_edges)
        
        delta = new_score - current_score
        
        if delta < 0 or random.random() < math.exp(-delta / T):
            current_edges.discard((u, v))
            current_edges.add(cand_e)
            current_score = new_score
            
            if current_score < best_score:
                best_score = current_score
                best_edges = set(current_edges)
                
        T *= cooling
        iter_count += 1
        
    return best_score, list(best_edges)

def _generate_multiplicative_values():
    """
    Generate value sets with strong multiplicative structure to maximize
    product collisions. Use combinations of small prime powers.
    """
    candidates = []
    n = 14
    
    # Strategy 1: Powers of 2 and 3 only
    # 1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 27, 32, 36
    candidates.append([1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 27, 32, 36])
    
    # Strategy 2: Include 5 as well
    # 1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20
    candidates.append([1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20])
    
    # Strategy 3: Heavier on 2s and 3s
    # 1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 27, 32, 48
    candidates.append([1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 27, 32, 48])
    
    # Strategy 4: Add more 5s
    # 1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 20, 24
    candidates.append([1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 20, 24])
    
    # Strategy 5: Very dense small numbers
    # 1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 14, 15, 16, 18
    candidates.append([1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 14, 15, 16, 18])
    
    # Strategy 6: Include 7
    # 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 15, 16
    candidates.append([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 15, 16])
    
    # Strategy 7: Mix of prime powers up to 5
    # 1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 20, 25
    candidates.append([1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 20, 25])
    
    # Strategy 8: Focus on 2,3,5 with more variety
    # 1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 18, 20, 24
    candidates.append([1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 18, 20, 24])
    
    # Strategy 9: Very small range
    # 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 15, 16
    candidates.append([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 15, 16])
    
    # Strategy 10: Random search over multiplicative structures
    random.seed(123)
    for _ in range(30):
        # Pick 14 numbers from 1..40 that have many small factors
        pool = []
        for i in range(1, 41):
            # Count number of divisors
            div_count = sum(1 for d in range(1, i+1) if i % d == 0)
            pool.append((div_count, i))
        pool.sort(reverse=True)
        # Take top 20, then sample 14
        top_pool = [x[1] for x in pool[:20]]
        if len(top_pool) >= 14:
            vals = random.sample(top_pool, 14)
            vals.sort()
            if len(set(vals)) == 14:
                candidates.append(vals)
    
    # Deduplicate
    unique_candidates = []
    seen = set()
    for c in candidates:
        key = tuple(sorted(c))
        if key not in seen:
            seen.add(key)
            unique_candidates.append(list(c))
            
    return unique_candidates

def _local_search_values(values, edges, max_iters=2000):
    """
    Local search on values: replace one value at a time with a small integer,
    keeping the same edge set, to see if we can improve the score.
    """
    n = len(values)
    best_vals = values[:]
    best_edges = edges[:]
    best_score = _score(values, edges)
    
    T = 5.0
    cooling = 0.995
    min_T = 0.5
    
    iter_count = 0
    while T > min_T and iter_count < max_iters:
        if _check_time():
            break
            
        # Pick a random vertex to change
        idx = random.randint(0, n - 1)
        old_val = best_vals[idx]
        
        # Try a few candidate new values
        found_improvement = False
        for _ in range(20):
            new_val = random.randint(1, 50)
            if new_val == old_val or new_val in best_vals:
                continue
                
            # Create new values
            new_vals = best_vals[:]
            new_vals[idx] = new_val
            
            # Score with same edges
            new_score = _score(new_vals, best_edges)
            
            if new_score < best_score or (new_score == best_score and random.random() < 0.5):
                best_vals = new_vals
                best_score = new_score
                found_improvement = True
                break
        
        if not found_improvement:
            T *= cooling
            
        iter_count += 1
        
    return best_score, best_vals, best_edges

def construct():
    n = 14
    m = 35
    
    candidates = _generate_multiplicative_values()
    
    best_score = float('inf')
    best_values = None
    best_edges = None
    
    # Iterate through candidates
    for vals in candidates:
        if _check_time():
            break
            
        # Ensure valid values
        if len(set(vals)) != n:
            continue
        if min(vals) < 1 or max(vals) > 10000:
            continue
            
        score, edges = _optimize_edges(vals, n_edges=m, max_iters=3000)
        
        if score < best_score:
            best_score = score
            best_values = vals
            best_edges = edges
            
            # If we hit a very low score, we can stop early
            if best_score <= 5:
                break
                
    # If we have time, try to improve the best values found
    if best_values is not None and not _check_time():
        current_vals = best_values[:]
        current_edges = best_edges
        current_score = best_score
        
        # Local search on values
        new_score, new_vals, new_edges = _local_search_values(current_vals, current_edges, max_iters=1500)
        
        if new_score < current_score:
            current_score = new_score
            best_score = new_score
            best_values = new_vals
            best_edges = new_edges
            
        # Try a few more rounds of local search with different seeds
        for seed in [42, 123, 456]:
            if _check_time():
                break
            random.seed(seed)
            new_score, new_vals, new_edges = _local_search_values(current_vals, current_edges, max_iters=500)
            if new_score < best_score:
                best_score = new_score
                best_values = new_vals
                best_edges = new_edges
                current_vals = new_vals
                current_edges = new_edges

    # Final validation
    if best_values is None:
        # Fallback
        best_values = [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20]
        best_edges = list(combinations(range(n), 2))[:m]
        
    # Ensure types
    values = [int(v) for v in best_values]
    edges = [[int(u), int(v)] for u, v in best_edges]
    
    # Sanity checks
    assert len(values) == n
    assert len(set(values)) == n
    assert all(isinstance(v, int) and not isinstance(v, bool) for v in values)
    assert min(values) >= 1 and max(values) <= 10000
    assert len(edges) == m
    assert all(u != v for u, v in edges)
    assert len(set(tuple(sorted(e)) for e in edges)) == m
    
    return {'values': values, 'edges': edges}
