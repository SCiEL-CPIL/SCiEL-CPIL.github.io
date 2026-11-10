"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

P = {'n': 14, 'm': 35, 'maximum': 10000}

def _raw_score(values, edges):
    sums = set()
    prods = set()
    for i, j in edges:
        s = values[i] + values[j]
        p = values[i] * values[j]
        sums.add(s)
        prods.add(p)
    return max(len(sums), len(prods))

def _all_pairs(n):
    return list(combinations(range(n), 2))

def _greedy_edges(values, all_edges, m):
    # Pick edges that maximize collision (minimize new distinct sums/prods)
    # Start with all edges, greedily add edges that add fewest new distinct values
    sums = set()
    prods = set()
    selected = []
    remaining = list(all_edges)
    
    for _ in range(m):
        best_edge = None
        best_score = float('inf')
        for e in remaining:
            i, j = e
            s = values[i] + values[j]
            p = values[i] * values[j]
            new_sums = 0 if s in sums else 1
            new_prods = 0 if p in prods else 1
            score = new_sums + new_prods
            if score < best_score:
                best_score = score
                best_edge = e
            elif score == best_score:
                # tie-break: prefer edges with lower values to keep things small
                pass
        if best_edge is None:
            break
        selected.append(best_edge)
        remaining.remove(best_edge)
        i, j = best_edge
        sums.add(values[i] + values[j])
        prods.add(values[i] * values[j])
    
    return selected

def _local_search_edges(values, edges, all_edges, iters=200):
    # Try to improve by swapping edges
    current_score = _raw_score(values, edges)
    edge_set = set(tuple(sorted(e)) for e in edges)
    
    for _ in range(iters):
        improved = False
        for _trial in range(50):
            # Pick a random edge to remove
            e_out = random.choice(edges)
            # Pick a random edge not in current set to add
            candidates = [e for e in all_edges if tuple(sorted(e)) not in edge_set]
            if not candidates:
                break
            e_in = random.choice(candidates)
            
            new_edges = [e for e in edges if e != e_out] + [e_in]
            new_score = _raw_score(values, new_edges)
            
            if new_score < current_score:
                current_score = new_score
                edges = new_edges
                edge_set.discard(tuple(sorted(e_out)))
                edge_set.add(tuple(sorted(e_in)))
                improved = True
                break
        
        if not improved:
            break
    
    return edges, current_score

def _gen_value_set(n, maximum):
    # Generate a structured value set
    # Use products of small primes to create multiplicative structure
    # This creates many collisions in products
    base_primes = [2, 3, 5]
    candidates = set()
    
    # Generate numbers of form 2^a * 3^b * 5^c
    for a in range(0, 15):
        for b in range(0, 10):
            for c in range(0, 7):
                v = (2**a) * (3**b) * (5**c)
                if 1 <= v <= maximum:
                    candidates.add(v)
    
    candidates = sorted(candidates)
    if len(candidates) < n:
        # Fallback to random
        return random.sample(range(1, maximum + 1), n)
    
    # Pick n values with good spread but structured
    # Use a subset that maximizes collision potential
    # Heuristic: pick values that are close in log scale
    import math
    log_candidates = [(math.log(v), v) for v in candidates]
    log_candidates.sort()
    
    # Pick evenly spaced in log space
    step = len(log_candidates) // n
    selected = [log_candidates[i * step][1] for i in range(n)]
    
    return selected

def _gen_random_value_set(n, maximum):
    return random.sample(range(1, maximum + 1), n)

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s deadline, 10s safety margin
    
    n = P['n']
    m = P['m']
    maximum = P['maximum']
    all_edges = _all_pairs(n)
    
    best_score = float('inf')
    best_values = None
    best_edges = None
    
    # Try multiple strategies
    strategies = []
    
    # Strategy 1: Structured value set (multiplicative)
    strategies.append(('structured', _gen_value_set(n, maximum)))
    
    # Strategy 2: Small values (1 to 14)
    strategies.append(('small', list(range(1, n + 1))))
    
    # Strategy 3: Geometric-like
    strategies.append(('geo', [2**i for i in range(n)] if 2**(n-1) <= maximum else list(range(1, n+1))))
    
    # Strategy 4: Divisor-rich
    divisor_rich = []
    for v in range(1, maximum + 1):
        d = sum(1 for i in range(1, int(v**0.5) + 1) if v % i == 0)
        divisor_rich.append((d, v))
    divisor_rich.sort(reverse=True)
    strategies.append(('divisor', [v for _, v in divisor_rich[:n]]))
    
    # Add some random sets
    for _ in range(5):
        strategies.append(('random', _gen_random_value_set(n, maximum)))
    
    for strategy_name, values in strategies:
        if time.time() > deadline:
            break
        
        # Ensure values are distinct and valid
        if len(set(values)) != n or any(v < 1 or v > maximum for v in values):
            continue
        
        # Get initial edges via greedy
        edges = _greedy_edges(values, all_edges, m)
        score = _raw_score(values, edges)
        
        if score < best_score:
            best_score = score
            best_values = list(values)
            best_edges = list(edges)
        
        # Local search on edges
        if time.time() < deadline - 5:
            edges, score = _local_search_edges(values, edges, all_edges, iters=100)
            if score < best_score:
                best_score = score
                best_values = list(values)
                best_edges = list(edges)
    
    # Try to improve values with local search if time permits
    if best_values is not None and time.time() < deadline - 10:
        for _ in range(50):
            if time.time() > deadline - 2:
                break
            # Perturb one value
            new_values = list(best_values)
            idx = random.randint(0, n - 1)
            old_v = new_values[idx]
            new_v = random.randint(1, maximum)
            if new_v in new_values:
                continue
            new_values[idx] = new_v
            
            # Re-optimize edges
            edges = _greedy_edges(new_values, all_edges, m)
            score = _raw_score(new_values, edges)
            
            if score < best_score:
                best_score = score
                best_values = new_values
                best_edges = edges
                # Local search
                if time.time() < deadline - 5:
                    edges, score = _local_search_edges(new_values, edges, all_edges, iters=50)
                    if score < best_score:
                        best_score = score
                        best_values = list(new_values)
                        best_edges = list(edges)
    
    if best_values is None:
        # Fallback
        best_values = list(range(1, n + 1))
        best_edges = [list(e) for e in all_edges[:m]]
    
    # Ensure types are correct
    best_values = [int(v) for v in best_values]
    best_edges = [[int(i), int(j)] for i, j in best_edges]
    
    return {'values': best_values, 'edges': best_edges}
# EVOLVE-BLOCK-END
