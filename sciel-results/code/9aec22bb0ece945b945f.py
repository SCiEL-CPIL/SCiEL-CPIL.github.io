"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

# Deterministic seed for reproducibility within a run
_SEED = 42

def _score(values, edges):
    """Compute the objective: max(distinct sums, distinct products)."""
    sums = set()
    prods = set()
    for u, v in edges:
        a = values[u]
        b = values[v]
        sums.add(a + b)
        prods.add(a * b)
    return max(len(sums), len(prods))

def _random_values(n=14, maximum=10000):
    """Generate n distinct positive integers <= maximum."""
    while True:
        vals = random.sample(range(1, maximum + 1), n)
        # Ensure they are distinct (sample guarantees this)
        return vals

def _random_edges(n=14, m=35):
    """Generate m random distinct edges from all possible pairs."""
    all_edges = list(combinations(range(n), 2))
    return [list(e) for e in random.sample(all_edges, m)]

def _local_search(values, edges, max_iters=5000, deadline=None):
    """
    Local search: try swapping values and swapping edges to improve the score.
    """
    best_score = _score(values, edges)
    best_values = values[:]
    best_edges = [e[:] for e in edges]
    
    n = len(values)
    all_possible_edges = list(combinations(range(n), 2))
    
    iter_count = 0
    while iter_count < max_iters:
        if deadline is not None and time.time() > deadline:
            break
        iter_count += 1
        
        current_score = _score(best_values, best_edges)
        
        # Try moving a value
        improved = False
        for _ in range(20):  # Limit attempts per iteration to keep it fast
            i = random.randint(0, n - 1)
            old_val = best_values[i]
            # Pick a new random value
            new_val = random.randint(1, 10000)
            if new_val == old_val:
                continue
            # Check distinctness
            if new_val in best_values:
                continue
            
            test_values = best_values[:]
            test_values[i] = new_val
            test_score = _score(test_values, best_edges)
            
            if test_score < current_score:
                best_values[i] = new_val
                current_score = test_score
                improved = True
                break
        
        if improved and current_score < best_score:
            best_score = current_score
            best_values = best_values[:]
            continue
        
        # Try swapping an edge
        improved = False
        for _ in range(10):
            # Pick a random current edge
            if not best_edges:
                break
            idx = random.randint(0, len(best_edges) - 1)
            old_edge = best_edges[idx]
            
            # Pick a random non-edge
            edge_set = set(tuple(sorted(e)) for e in best_edges)
            while True:
                cand = random.choice(all_possible_edges)
                if tuple(sorted(cand)) not in edge_set:
                    break
            
            test_edges = best_edges[:]
            test_edges[idx] = list(cand)
            test_score = _score(best_values, test_edges)
            
            if test_score < current_score:
                best_edges[idx] = list(cand)
                current_score = test_score
                improved = True
                break
        
        if improved and current_score < best_score:
            best_score = current_score
            best_edges = best_edges[:]
            continue
        
        # If no improvement in a while, restart with new random values
        if iter_count % 500 == 0:
            best_values = _random_values()
            best_edges = _random_edges()
            best_score = _score(best_values, best_edges)
    
    return best_values, best_edges, best_score

def _greedy_edge_selection(values, max_edges=35, deadline=None):
    """
    Greedily select edges to minimize the number of distinct sums/products.
    """
    n = len(values)
    all_edges = list(combinations(range(n), 2))
    random.shuffle(all_edges)
    
    selected_edges = []
    sums = set()
    prods = set()
    
    for u, v in all_edges:
        if len(selected_edges) >= max_edges:
            break
        if deadline is not None and time.time() > deadline:
            break
        
        s = values[u] + values[v]
        p = values[u] * values[v]
        
        # Prefer edges that reuse existing sums/products
        if s in sums or p in prods:
            selected_edges.append([u, v])
            sums.add(s)
            prods.add(p)
        elif len(sums) + len(prods) < 100:  # Keep it small
            selected_edges.append([u, v])
            sums.add(s)
            prods.add(p)
    
    return selected_edges

def construct():
    random.seed(_SEED)
    
    deadline = time.time() + 85  # Safety margin before 90s limit
    
    best_overall = None
    best_score = float('inf')
    
    # Strategy 1: Random restarts with local search
    while time.time() < deadline - 5:  # Leave 5s for final checks
        values = _random_values()
        edges = _random_edges()
        
        # Quick greedy refinement of edges
        edges = _greedy_edge_selection(values, deadline=deadline)
        
        # Local search
        v, e, s = _local_search(values, edges, max_iters=2000, deadline=deadline)
        
        if s < best_score:
            best_score = s
            best_overall = (v, e)
            
            # If we hit a very low score, we can stop early
            if s <= 30:
                break
    
    # Strategy 2: Try structured value sets
    if time.time() < deadline - 5:
        # Try powers of 2
        for base in [2, 3]:
            vals = []
            x = base
            while len(vals) < 14 and x <= 10000:
                vals.append(x)
                x *= base
            if len(vals) == 14:
                edges = _greedy_edge_selection(vals, deadline=deadline)
                v, e, s = _local_search(vals, edges, max_iters=1000, deadline=deadline)
                if s < best_score:
                    best_score = s
                    best_overall = (v, e)
    
    # Strategy 3: Try arithmetic progressions
    if time.time() < deadline - 5:
        for start in [1, 2, 3, 5, 10]:
            for diff in [2, 3, 5, 7, 10, 15, 20, 50, 100]:
                vals = []
                x = start
                while len(vals) < 14 and x <= 10000:
                    vals.append(x)
                    x += diff
                if len(vals) == 14:
                    edges = _greedy_edge_selection(vals, deadline=deadline)
                    v, e, s = _local_search(vals, edges, max_iters=500, deadline=deadline)
                    if s < best_score:
                        best_score = s
                        best_overall = (v, e)
    
    if best_overall is None:
        # Fallback
        values = list(range(1, 15))
        edges = [list(e) for e in list(combinations(range(14), 2))[:35]]
    else:
        values, edges = best_overall
    
    # Ensure types are correct
    values = [int(v) for v in values]
    edges = [[int(u), int(v)] for u, v in edges]
    
    return {'values': values, 'edges': edges}

# EVOLVE-BLOCK-END
