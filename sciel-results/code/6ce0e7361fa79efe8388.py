"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _raw_score(values, edges):
    """Compute the raw score: max(distinct sums, distinct products)."""
    sums = set()
    prods = set()
    for u, v in edges:
        su = values[u]
        sv = values[v]
        sums.add(su + sv)
        prods.add(su * sv)
    return max(len(sums), len(prods))

def _greedy_edge_selection(values, n, m, deadline):
    """
    Greedily select m edges to minimize the score.
    Strategy: Start with empty graph. At each step, add the edge that causes
    the smallest increase in max(distinct_sums, distinct_prods).
    Ties are broken randomly to allow exploration.
    """
    all_edges = list(combinations(range(n), 2))
    random.shuffle(all_edges)
    
    current_sums = set()
    current_prods = set()
    selected_edges = []
    
    for step in range(m):
        if time.time() > deadline:
            break
        best_edge = None
        best_score = float('inf')
        best_sums = None
        best_prods = None
        
        for u, v in all_edges:
            if (u, v) in selected_edges or (v, u) in selected_edges:
                continue
            
            su = values[u]
            sv = values[v]
            s = su + sv
            p = su * sv
            
            # Estimate new score
            new_sums_count = len(current_sums) + (0 if s in current_sums else 1)
            new_prods_count = len(current_prods) + (0 if p in current_prods else 1)
            new_score = max(new_sums_count, new_prods_count)
            
            if new_score < best_score:
                best_score = new_score
                best_edge = (u, v)
                best_sums = current_sums | {s}
                best_prods = current_prods | {p}
            elif new_score == best_score and random.random() < 0.1:
                best_edge = (u, v)
                best_sums = current_sums | {s}
                best_prods = current_prods | {p}
        
        if best_edge is None:
            break
        
        selected_edges.append(best_edge)
        current_sums = best_sums
        current_prods = best_prods
    
    return selected_edges

def _generate_value_candidates(n, max_val, deadline):
    """
    Generate candidate value sets.
    Focus on small values and highly composite numbers to maximize collisions.
    """
    candidates = []
    
    # Strategy 1: Small consecutive values
    candidates.append(list(range(1, n + 1)))
    
    # Strategy 2: Powers of 2 (maximize product collisions)
    candidates.append([2**i for i in range(n) if 2**i <= max_val])
    while len(candidates[-1]) < n:
        candidates[-1].append(max_val)
    
    # Strategy 3: Highly composite numbers (many divisors)
    # 1, 2, 3, 4, 6, 8, 12, 16, 18, 20, 24, 30, 32, 36, 40, 42, 48, 56, 60, 72, 80, 84, 90, 96, 108, 120, ...
    highly_composite = [1, 2, 3, 4, 6, 8, 12, 16, 18, 20, 24, 30, 32, 36, 40, 42, 48, 56, 60, 72, 80, 84, 90, 96, 108, 120, 144, 168, 180, 240, 360, 720, 840, 1260, 1680, 2520, 5040]
    candidates.append(highly_composite[:n])
    
    # Strategy 4: Mix of small primes and composites
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43]
    composites = [4, 6, 8, 9, 10, 12, 14, 15, 16, 18, 20, 21, 22, 24]
    mixed = []
    for i in range(n):
        if i % 2 == 0:
            mixed.append(primes[i // 2])
        else:
            mixed.append(composites[i // 2])
    candidates.append(mixed)
    
    # Strategy 5: Random small values
    for _ in range(10):
        random_vals = random.sample(range(1, 100), n)
        candidates.append(random_vals)
    
    # Strategy 6: Geometric progression with small ratio
    for ratio in [2, 3]:
        vals = [1]
        for i in range(1, n):
            vals.append(vals[-1] * ratio)
        if vals[-1] <= max_val:
            candidates.append(vals)
    
    # Strategy 7: Factorial-based
    factorials = [1]
    for i in range(1, n + 1):
        factorials.append(factorials[-1] * i)
    if factorials[n - 1] <= max_val:
        candidates.append(factorials[:n])
    
    # Deduplicate
    unique_candidates = []
    seen = set()
    for c in candidates:
        key = tuple(sorted(c))
        if key not in seen:
            seen.add(key)
            unique_candidates.append(c)
    
    return unique_candidates

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s out of 90s
    
    n = 14
    m = 35
    max_val = 10000
    
    best_values = None
    best_edges = None
    best_score = float('inf')
    
    # Generate candidate value sets
    candidates = _generate_value_candidates(n, max_val, deadline)
    
    for values in candidates:
        if time.time() > deadline:
            break
        
        # Ensure values are distinct positive integers <= max_val
        if len(set(values)) != n or any(v < 1 or v > max_val for v in values):
            continue
        
        # Try greedy edge selection with different random seeds
        for seed in range(5):
            if time.time() > deadline:
                break
            
            random.seed(seed)
            edges = _greedy_edge_selection(values, n, m, deadline)
            
            if len(edges) < m:
                continue
            
            score = _raw_score(values, edges)
            
            if score < best_score:
                best_score = score
                best_values = values[:]
                best_edges = [list(e) for e in edges]
    
    # If we didn't find a good solution, fall back to simple approach
    if best_values is None:
        best_values = list(range(1, n + 1))
        all_edges = list(combinations(range(n), 2))
        random.shuffle(all_edges)
        best_edges = [list(e) for e in all_edges[:m]]
        best_score = _raw_score(best_values, best_edges)
    
    # Final polish: try local search on the best solution
    if time.time() < deadline - 5:
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            current_score = _raw_score(best_values, best_edges)
            
            # Try swapping edges
            for i in range(len(best_edges)):
                if time.time() > deadline - 2:
                    break
                
                for u, v in combinations(range(n), 2):
                    if (u, v) in best_edges or (v, u) in best_edges:
                        continue
                    
                    # Try adding this edge and removing one existing edge
                    for j in range(len(best_edges)):
                        new_edges = best_edges[:]
                        new_edges[j] = [u, v]
                        
                        # Check for duplicates
                        edge_set = set()
                        valid = True
                        for e in new_edges:
                            key = tuple(sorted(e))
                            if key in edge_set:
                                valid = False
                                break
                            edge_set.add(key)
                        
                        if not valid:
                            continue
                        
                        new_score = _raw_score(best_values, new_edges)
                        
                        if new_score < current_score:
                            best_edges = new_edges
                            current_score = new_score
                            improved = True
                            break
                
                if improved:
                    break
            
            # Try changing one value
            for i in range(n):
                if time.time() > deadline - 2:
                    break
                
                old_val = best_values[i]
                for new_val in range(1, min(max_val + 1, 100)):
                    if new_val == old_val:
                        continue
                    
                    if new_val in best_values:
                        continue
                    
                    best_values[i] = new_val
                    new_score = _raw_score(best_values, best_edges)
                    
                    if new_score < current_score:
                        current_score = new_score
                        improved = True
                    else:
                        best_values[i] = old_val
    
    # Ensure types are correct
    best_values = [int(v) for v in best_values]
    best_edges = [[int(u), int(v)] for u, v in best_edges]
    
    return {'values': best_values, 'edges': best_edges}
# EVOLVE-BLOCK-END
