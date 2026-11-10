"""Optimized program for small sums and products along selected graph edges."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

P = {'n': 14, 'm': 35, 'maximum': 10000}

def _score(values, edges):
    """Compute the raw value: max(#distinct sums, #distinct products)."""
    sums = set()
    prods = set()
    for i, j in edges:
        vi = values[i]
        vj = values[j]
        sums.add(vi + vj)
        prods.add(vi * vj)
    return max(len(sums), len(prods))

def _optimize_graph_for_values(values, initial_edges, max_iter=20000, timeout=5.0):
    """
    Local search on the graph structure for fixed values.
    Tries to remove one edge and add another to reduce the score.
    """
    n = len(values)
    all_possible_edges = set(combinations(range(n), 2))
    
    # Convert initial edges to a set of tuples for fast lookup
    current_edges = set(tuple(e) for e in initial_edges)
    
    # Ensure we have exactly m edges
    if len(current_edges) != P['m']:
        # If not, just take a random subset or superset? 
        # The task requires exactly 35.
        # Let's assume initial_edges is valid.
        pass
        
    current_score = _score(values, [list(e) for e in current_edges])
    best_edges = set(current_edges)
    best_score = current_score
    
    start_time = time.time()
    iter_count = 0
    no_improve = 0
    max_no_improve = 2000
    
    while iter_count < max_iter:
        if time.time() - start_time > timeout:
            break
            
        # Pick a random edge to remove
        if not current_edges:
            break
        edge_to_remove = random.choice(list(current_edges))
        
        # Pick a random edge to add (not currently in graph)
        # To speed up, we can pre-compute non-edges, but it changes.
        # Simple approach: try random pairs until we find a non-edge.
        # With 91 possible and 35 present, probability of non-edge is ~60%.
        new_edge = None
        for _ in range(10): # 10 tries to find a non-edge
            u, v = random.sample(range(n), 2)
            candidate = (min(u, v), max(u, v))
            if candidate not in current_edges:
                new_edge = candidate
                break
        
        if new_edge is None:
            continue
            
        # Perform swap
        current_edges.remove(edge_to_remove)
        current_edges.add(new_edge)
        
        # Evaluate
        new_score = _score(values, [list(e) for e in current_edges])
        iter_count += 1
        
        if new_score < best_score:
            best_score = new_score
            best_edges = set(current_edges)
            no_improve = 0
        else:
            # Revert
            current_edges.remove(new_edge)
            current_edges.add(edge_to_remove)
            no_improve += 1
            
        if no_improve > max_no_improve:
            break
            
    return [list(e) for e in best_edges], best_score

def _optimize_values_for_edges(edges, initial_values, max_iter=10000, timeout=5.0):
    """
    Local search on values for fixed edges.
    """
    n = len(initial_values)
    current = list(initial_values)
    best_score = _score(current, edges)
    best_values = list(current)
    
    start_time = time.time()
    iter_count = 0
    no_improve = 0
    max_no_improve = 2000
    
    while iter_count < max_iter:
        if time.time() - start_time > timeout:
            break
        
        # Pick two vertices to swap values
        i, j = random.sample(range(n), 2)
        
        # Swap
        current[i], current[j] = current[j], current[i]
        
        new_score = _score(current, edges)
        iter_count += 1
        
        if new_score < best_score:
            best_score = new_score
            best_values = list(current)
            no_improve = 0
        else:
            current[i], current[j] = current[j], current[i]
            no_improve += 1
            
        if no_improve > max_no_improve:
            break
            
    return best_values, best_score

def _generate_candidate_values():
    """Generate a diverse set of candidate value vectors."""
    candidates = []
    n = P['n']
    
    # 1. Powers of 2
    candidates.append([2**i for i in range(n)])
    
    # 2. Arithmetic Progressions
    for d in [1, 2, 3, 5, 7, 10]:
        candidates.append([1 + i*d for i in range(n)])
        
    # 3. Primes
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43]
    candidates.append(primes)
    
    # 4. Small integers
    candidates.append(list(range(1, n+1)))
    
    # 5. Randomized (several seeds)
    for seed in range(5):
        rng = random.Random(seed)
        vals = sorted(rng.sample(range(1, 10001), n))
        candidates.append(vals)
        
    # 6. Mixed: Small range, dense
    candidates.append([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
    
    # 7. Geometric-ish
    candidates.append([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192])
    
    return candidates

def construct():
    start = time.time()
    deadline = start + 85.0  # Safety margin for 90s limit
    
    n = P['n']
    m = P['m']
    
    best_values = None
    best_edges = None
    best_score = float('inf')
    
    # Strategy 1: Optimize Graph for Fixed Values
    value_candidates = _generate_candidate_values()
    
    # Initial edges for graph optimization: Random or Circulant
    # We'll use a few different starting graphs
    start_graphs = []
    
    # Circulant graph
    circ_edges = set()
    for i in range(n):
        for s in [1, 2, 3, 4, 5]:
            j = (i + s) % n
            if j != i:
                a, b = min(i, j), max(i, j)
                circ_edges.add((a, b))
    if len(circ_edges) == m:
        start_graphs.append([list(e) for e in circ_edges])
    else:
        # Adjust if circulant doesn't have exactly 35
        # 14 * 5 = 70 directed, 35 undirected. So it should be exactly 35.
        pass
        
    # Random graphs
    for seed in range(3):
        rng = random.Random(seed)
        all_edges = list(combinations(range(n), 2))
        rng.shuffle(all_edges)
        start_graphs.append([list(e) for e in all_edges[:m]])
        
    # Iterate through value candidates and start graphs
    # We need to manage time carefully.
    # Total candidates: ~10 values, ~4 graphs = 40 combinations.
    # Time per combination: ~2s. Total 80s. Tight.
    # Let's reduce.
    
    # Select top 5 value candidates and top 2 start graphs
    selected_values = value_candidates[:5]
    selected_graphs = start_graphs[:2]
    
    for vals in selected_values:
        if time.time() > deadline - 10:
            break
        if len(set(vals)) != n or any(v < 1 or v > 10000 for v in vals):
            continue
            
        for init_edges in selected_graphs:
            if time.time() > deadline - 5:
                break
            try:
                opt_edges, sc = _optimize_graph_for_values(vals, init_edges, max_iter=5000, timeout=3.0)
                if sc < best_score:
                    best_score = sc
                    best_values = list(vals)
                    best_edges = opt_edges
            except Exception:
                continue
                
    # Strategy 2: Optimize Values for Fixed Edges (using best edges found or good fixed edges)
    # Use the best edges found so far, or a default good edge set
    fixed_edges_for_val_opt = best_edges if best_edges else start_graphs[0]
    
    # Try to improve values for this specific edge set
    if time.time() < deadline - 10:
        # Use a few good value starts
        val_starts = [list(range(1, n+1)), [2**i for i in range(n)], [1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41]]
        for v_start in val_starts:
            if time.time() > deadline - 5:
                break
            if len(set(v_start)) != n or any(v < 1 or v > 10000 for v in v_start):
                continue
            try:
                opt_vals, sc = _optimize_values_for_edges(fixed_edges_for_val_opt, v_start, max_iter=5000, timeout=3.0)
                if sc < best_score:
                    best_score = sc
                    best_values = list(opt_vals)
                    # Edges remain fixed_edges_for_val_opt
            except Exception:
                continue

    # Fallback
    if best_values is None:
        best_values = list(range(1, n + 1))
        best_edges = [list(e) for e in list(combinations(range(n), 2))[:m]]
        best_score = _score(best_values, best_edges)
    
    # Final validation
    assert len(best_values) == n
    assert len(set(best_values)) == n
    assert all(isinstance(v, int) and 1 <= v <= 10000 for v in best_values)
    assert len(best_edges) == m
    edge_set = set()
    for e in best_edges:
        assert len(e) == 2
        a, b = e
        assert isinstance(a, int) and isinstance(b, int)
        assert a != b
        assert 0 <= a < n and 0 <= b < n
        key = (min(a, b), max(a, b))
        assert key not in edge_set
        edge_set.add(key)
    
    return {'values': [int(v) for v in best_values], 'edges': [[int(a), int(b)] for a, b in best_edges]}
# EVOLVE-BLOCK-END
