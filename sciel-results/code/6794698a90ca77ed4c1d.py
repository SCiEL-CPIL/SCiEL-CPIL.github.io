"""Search-based: minimize max(distinct_sums, distinct_products) over 35 edges of 14 vertices."""
# EVOLVE-BLOCK-START
import time
import itertools
import random
from collections import defaultdict

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

def _greedy_edges(values, target_m):
    """Greedy edge selection: prioritize edges that reuse existing sums/products."""
    n = len(values)
    all_e = _all_edges(n)
    sums = set()
    prods = set()
    edges = []
    edge_set = set()
    
    # Precompute sum/prod for all edges to speed up greedy
    edge_props = {}
    for u, v in all_e:
        a, b = values[u], values[v]
        edge_props[(u, v)] = (a + b, a * b)
        
    for _ in range(target_m):
        best_edge = None
        best_score = float('inf')
        
        for (u, v) in all_e:
            if (u, v) in edge_set:
                continue
            
            s, p = edge_props[(u, v)]
            new_s = 1 if s not in sums else 0
            new_p = 1 if p not in prods else 0
            
            # We want to minimize the resulting max(count_sums, count_prods)
            # Current counts: len(sums), len(prods)
            # New counts: len(sums)+new_s, len(prods)+new_p
            # Score is max of these.
            # To make it faster, we can just check if it adds 0 new items first.
            # If it adds 0, it's ideal. If it adds 1 to the smaller set, it's good.
            
            cur_s = len(sums)
            cur_p = len(prods)
            
            # If both are new, it increases both.
            # If one is new, it increases the max if that set is currently the max or tied.
            
            # Simple heuristic score: number of new items
            # Tie-break: prefer edges that connect to high-degree nodes? 
            # For now, just minimize new items.
            score = new_s + new_p
            
            # Secondary: if score is same, prefer edges that are part of larger collision groups?
            # We don't have that info easily here without pre-computing groups.
            # Let's stick to minimizing new items.
            
            if score < best_score:
                best_score = score
                best_edge = (u, v)
                
        if best_edge is None:
            break
        
        u, v = best_edge
        edges.append((u, v))
        edge_set.add((u, v))
        s, p = edge_props[(u, v)]
        sums.add(s)
        prods.add(p)
        
    return edges

def _local_search_edges(values, edges, max_iters=1000):
    """Local search: try swapping edges to reduce distinct sums/products."""
    n = len(values)
    all_e = _all_edges(n)
    edge_set = set(edges)
    current_score = _raw_score(values, edges)
    
    # Precompute edge props
    edge_props = {}
    for u, v in all_e:
        edge_props[(u, v)] = (values[u] + values[v], values[u] * values[v])
        
    for _ in range(max_iters):
        if time.time() > DEADLINE - 2:
            break
            
        improved = False
        # Shuffle edges to avoid bias
        edge_indices = list(range(len(edges)))
        random.shuffle(edge_indices)
        
        for idx in edge_indices:
            u, v = edges[idx]
            s_rm, p_rm = edge_props[(u, v)]
            
            # Temporarily remove edge
            # We need to know the state of sums/prods without this edge
            # Instead of rebuilding sets every time (expensive), we can track counts.
            # But for simplicity and robustness in this short time, let's do a quick check.
            # Actually, rebuilding 35 edges is fast.
            
            sums = set()
            prods = set()
            for i, (a, b) in enumerate(edges):
                if i == idx:
                    continue
                sa, pa = edge_props[(a, b)]
                sums.add(sa)
                prods.add(pa)
                
            # Try adding each non-edge
            best_add = None
            best_new_score = current_score
            
            for (x, y) in all_e:
                if (x, y) in edge_set:
                    continue
                
                s_add, p_add = edge_props[(x, y)]
                
                # New sets
                # If s_add in sums, len(sums) stays same. Else +1.
                new_sums_count = len(sums) + (0 if s_add in sums else 1)
                new_prods_count = len(prods) + (0 if p_add in prods else 1)
                
                new_score = max(new_sums_count, new_prods_count)
                
                if new_score < best_new_score:
                    best_new_score = new_score
                    best_add = (x, y)
            
            if best_add is not None and best_new_score < current_score:
                # Apply swap
                edges[idx] = best_add
                edge_set.discard((u, v))
                edge_set.add(best_add)
                current_score = best_new_score
                improved = True
                break # Restart search with new state
                
        if not improved:
            # Try 2-opt: remove 2, add 2? 
            # This is O(N^2 * M^2) which might be too slow.
            # Let's just break if no 1-swap improvement.
            break
            
    return edges, current_score

def _evaluate(values):
    """Evaluate a value set with greedy edge selection plus local search polish."""
    edges = _greedy_edges(values, 35)
    if len(edges) < 35:
        return float('inf'), values, edges
    # Polish with local search
    edges, score = _local_search_edges(values, edges, max_iters=500)
    return score, values, edges

def _try_value_set(values):
    """Try a specific value set."""
    if len(set(values)) != 14:
        return None
    if any(not isinstance(v, int) or v < 1 or v > 10000 for v in values):
        return None
    score, vals, edges = _evaluate(values)
    return score, vals, edges

def _generate_candidate_values():
    """Generate value sets optimized for product collisions."""
    candidates = []
    
    # Strategy: Values are multiples of g.
    # Base values v_i should be small and structured to minimize distinct v_i*v_j.
    
    # 1. Arithmetic Progressions scaled by g
    for g in range(2, 15):
        for a in range(1, 20):
            for d in range(1, 10):
                base_vals = [a + i * d for i in range(14)]
                if max(base_vals) > 10000 // g:
                    continue
                vals = [g * v for v in base_vals]
                if len(set(vals)) == 14 and max(vals) <= 10000:
                    candidates.append(vals)
                    
    # 2. Consecutive integers scaled by g
    for g in range(2, 15):
        for start in range(1, 20):
            base_vals = list(range(start, start + 14))
            if max(base_vals) > 10000 // g:
                continue
            vals = [g * v for v in base_vals]
            if len(set(vals)) == 14 and max(vals) <= 10000:
                candidates.append(vals)
                
    # 3. Small random sets scaled by g
    random.seed(42)
    for g in range(2, 12):
        for _ in range(200):
            # Sample 14 distinct small integers
            # Max base value should be around 10000/g
            max_base = 10000 // g
            if max_base < 14:
                continue
            base_vals = random.sample(range(1, max_base + 1), 14)
            vals = [g * v for v in base_vals]
            if len(set(vals)) == 14:
                candidates.append(vals)
                
    # 4. Pure small integers (g=1) - less likely to have product collisions but worth checking
    # APs
    for a in range(1, 20):
        for d in range(1, 10):
            vals = [a + i * d for i in range(14)]
            if max(vals) <= 10000 and len(set(vals)) == 14:
                candidates.append(vals)
                
    return candidates

def construct():
    start_time = time.time()
    best_score = float('inf')
    best_result = None
    
    # Strategy 1: Try collision-optimized value sets
    candidates = _generate_candidate_values()
    
    # Sort candidates? Maybe by "density" of small values?
    # Smaller values -> smaller products -> more collisions?
    # Let's just iterate.
    
    for vals in candidates:
        if time.time() > DEADLINE - 5:
            break
            
        result = _try_value_set(vals)
        if result:
            s, v, e = result
            if s < best_score:
                best_score = s
                best_result = (v, e)
                
    # Strategy 2: If we have time, do a random search for value sets
    if time.time() < DEADLINE - 10:
        random.seed(123)
        for _ in range(500):
            if time.time() > DEADLINE - 10:
                break
                
            # Generate a random set with some structure
            # Mix of small numbers and multiples
            g = random.choice([1, 2, 3, 4, 5, 6])
            max_val = 10000 // g
            if max_val < 14:
                continue
                
            vals = []
            # Pick some small numbers
            for _ in range(7):
                v = random.randint(1, min(50, max_val))
                if v not in vals:
                    vals.append(v)
            # Pick some larger numbers
            while len(vals) < 14:
                v = random.randint(1, max_val)
                if v not in vals:
                    vals.append(v)
                    
            vals = [g * v for v in vals]
            if len(set(vals)) != 14 or max(vals) > 10000:
                continue
                
            result = _try_value_set(vals)
            if result:
                s, v, e = result
                if s < best_score:
                    best_score = s
                    best_result = (v, e)
                    
    if best_result is None:
        # Fallback
        vals = list(range(1, 15))
        edges = list(itertools.combinations(range(14), 2))[:35]
        return {'values': vals, 'edges': [list(e) for e in edges]}
        
    vals, edges = best_result
    # Ensure proper types
    vals = [int(v) for v in vals]
    edges = [list(e) for e in edges]
    return {'values': vals, 'edges': edges}

# EVOLVE-BLOCK-END
