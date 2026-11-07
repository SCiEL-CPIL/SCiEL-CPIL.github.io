"""Search-based: minimize max(distinct_sums, distinct_products) over 35 edges of 14 vertices."""
# EVOLVE-BLOCK-START
import time
import itertools
import random

DEADLINE = 85.0  # safety margin before 90s kill

def _raw_score(values, edges):
    """Compute max(#distinct sums, #distinct products) over edges."""
    sums = set()
    prods = set()
    for u, v in edges:
        a, b = values[u], values[v]
        sums.add(a + b)
        prods.add(a * b)
    return max(len(sums), len(prods))

def _all_edges(n):
    return list(itertools.combinations(range(n), 2))

def _greedy_edge_selection(values, target_m):
    """
    Greedy edge selection: pick edges that minimize growth in distinct sums and products.
    Returns a list of edges.
    """
    n = len(values)
    all_e = _all_edges(n)
    
    # Precompute sums and products for all edges
    edge_data = []
    for u, v in all_e:
        a, b = values[u], values[v]
        edge_data.append((u, v, a + b, a * b))
    
    sums = set()
    prods = set()
    edges = []
    remaining = list(range(len(all_e)))
    
    # Sort initially? No, greedy picks best at each step.
    # To speed up, we can sort by 'potential' but dynamic state matters.
    
    for _ in range(target_m):
        best_idx = -1
        best_score = float('inf')
        best_sums = None
        best_prods = None
        best_u, best_v, best_s, best_p = -1, -1, 0, 0
        
        for idx in remaining:
            u, v, s, p = edge_data[idx]
            
            # Calculate new distinct counts
            # We don't want to create new sets every time, so we check membership
            new_sums_count = len(sums) + (0 if s in sums else 1)
            new_prods_count = len(prods) + (0 if p in prods else 1)
            
            current_max = max(new_sums_count, new_prods_count)
            
            # Tie-breaker: prefer adding to existing sets if possible?
            # Or minimize the sum of counts?
            # Let's minimize the max, then minimize the sum.
            tie_breaker = new_sums_count + new_prods_count
            
            if current_max < best_score or (current_max == best_score and tie_breaker < best_score):
                best_score = current_max
                best_idx = idx
                best_u, best_v, best_s, best_p = u, v, s, p
                # We don't update sums/prods sets yet, just track the hypothetical state
                # Actually, we need to know the *new* sets for tie breaking if scores are equal?
                # No, just the counts are enough for the max metric.
        
        if best_idx == -1:
            break
            
        # Add the chosen edge
        u, v, s, p = edge_data[best_idx]
        sums.add(s)
        prods.add(p)
        edges.append([u, v])
        remaining.remove(best_idx)
        
    return edges

def _local_search(values, edges, max_iters=5000, timeout=None):
    """
    Local search: swap edges to improve score.
    """
    n = len(values)
    all_e = set(_all_edges(n))
    current_edges = set(tuple(e) for e in edges)
    current_score = _raw_score(values, list(current_edges))
    
    # Precompute sums and products for all edges
    edge_props = {}
    for u, v in all_e:
        a, b = values[u], values[v]
        edge_props[(u, v)] = (a + b, a * b)
        edge_props[(v, u)] = (a + b, a * b)
        
    start_time = time.time()
    
    for _ in range(max_iters):
        if timeout and time.time() > timeout:
            break
            
        if not current_edges:
            break
            
        # Pick a random edge to remove
        e_to_remove = random.choice(list(current_edges))
        # Pick a random edge to add
        candidates = list(all_e - current_edges)
        if not candidates:
            break
        e_to_add = random.choice(candidates)
        
        # Ensure no loops or duplicates (handled by set logic)
        # Ensure undirected: (u,v) is same as (v,u)
        
        new_edges = current_edges - {e_to_remove} | {e_to_add}
        
        # Calculate new score
        # Optimization: incremental update
        # Remove e_to_remove's contribution
        # Add e_to_add's contribution
        
        # Simple re-evaluation is O(M), M=35. 35*2 ops. Very fast.
        # But we can do better.
        
        sums = set()
        prods = set()
        for u, v in new_edges:
            s, p = edge_props[(u, v)]
            sums.add(s)
            prods.add(p)
            
        new_score = max(len(sums), len(prods))
        
        if new_score < current_score:
            current_edges = new_edges
            current_score = new_score
            
    return list(current_edges), current_score

def construct():
    start_time = time.time()
    best_score = float('inf')
    best_result = None
    
    n = 14
    m = 35
    max_val = 10000
    
    # Generate diverse value sets
    value_sets = []
    
    # 1. Arithmetic Progression
    value_sets.append(list(range(1, n+1)))
    value_sets.append(list(range(1, max_val, max_val//n))[:n])
    
    # 2. Geometric Progression
    gp = [1]
    while len(gp) < n:
        gp.append(gp[-1] * 2)
    value_sets.append([min(x, max_val) for x in gp][:n])
    # Ensure distinctness
    seen = set()
    unique_gp = []
    for x in gp:
        if x not in seen:
            seen.add(x)
            unique_gp.append(x)
    value_sets.append(unique_gp[:n])
    
    # 3. Random sets
    random.seed(42)
    for _ in range(5):
        vals = random.sample(range(1, max_val+1), n)
        value_sets.append(vals)
        
    # 4. "Collision" sets: try to make many sums equal
    # e.g., pairs (1, 100), (2, 99), etc.
    # Let's create a set with many additive collisions
    colliding = []
    target_sum = 200
    for i in range(1, 11):
        if i < target_sum - i and i not in colliding and (target_sum - i) not in colliding:
            colliding.append(i)
            colliding.append(target_sum - i)
    # Fill rest
    while len(colliding) < n:
        x = random.randint(1, max_val)
        if x not in colliding:
            colliding.append(x)
    value_sets.append(colliding[:n])
    
    # 5. "Collision" sets: try to make many products equal
    # Factors of a highly composite number
    # 2520 = 2^3 * 3^2 * 5 * 7
    # Pairs: (1,2520), (2,1260), (3,840), (4,630), (5,504), (6,420), (7,360), (8,315), (9,280), (10,252)
    # All <= 10000
    factors = [1, 2520, 2, 1260, 3, 840, 4, 630, 5, 504, 6, 420, 7, 360]
    value_sets.append(factors[:n])
    
    # Remove duplicates in value sets
    unique_value_sets = []
    seen_sets = set()
    for vs in value_sets:
        # Normalize for comparison
        key = tuple(sorted(vs))
        if key not in seen_sets:
            seen_sets.add(key)
            unique_value_sets.append(vs)
            
    # For each value set, run greedy + local search
    for vs in unique_value_sets:
        if time.time() > DEADLINE:
            break
            
        # Greedy selection
        edges = _greedy_edge_selection(vs, m)
        score = _raw_score(vs, edges)
        
        if score < best_score:
            best_score = score
            best_result = {'values': vs, 'edges': edges}
            
        # Local search
        # Allocate remaining time proportionally or fixed chunk
        # Let's give 5 seconds per value set for local search if time permits
        ls_timeout = start_time + 5.0 
        if time.time() < DEADLINE - 5.0:
            edges_ls, score_ls = _local_search(vs, edges, max_iters=10000, timeout=ls_timeout)
            if score_ls < best_score:
                best_score = score_ls
                best_result = {'values': vs, 'edges': edges_ls}
                
    # If we still have time, do more random restarts on the best value set
    if best_result and time.time() < DEADLINE - 5.0:
        best_vs = best_result['values']
        # Run local search on best result with more iterations
        edges_ls, score_ls = _local_search(best_vs, best_result['edges'], max_iters=50000, timeout=DEADLINE - 2.0)
        if score_ls < best_score:
            best_score = score_ls
            best_result = {'values': best_vs, 'edges': edges_ls}
            
    # Final check
    if best_result is None:
        # Fallback
        values = list(range(1, n+1))
        edges = [list(e) for e in list(itertools.combinations(range(n), 2))[:m]]
        best_result = {'values': values, 'edges': edges}
        
    # Ensure types are correct
    best_result['values'] = [int(v) for v in best_result['values']]
    best_result['edges'] = [[int(u), int(v)] for u, v in best_result['edges']]
    
    return best_result

# EVOLVE-BLOCK-END
