"""Structural redesign: fix values to smooth numbers, optimize edge subset via fast local search."""
import time
import itertools
import random
import math

DEADLINE = 80.0  # safety margin before 90s kill

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

def _fast_local_search(values, edges, all_e, max_iters=200):
    """
    Fast local search for edge subset optimization.
    Uses precomputed sum/prod arrays for O(1) lookups.
    Tries single-edge swaps to reduce max(distinct_sums, distinct_prods).
    """
    n = len(values)
    m = len(edges)
    edge_set = set(edges)
    
    # Precompute sum and product for all edges
    sum_map = {}
    prod_map = {}
    for (u, v) in all_e:
        a, b = values[u], values[v]
        sum_map[(u, v)] = a + b
        prod_map[(u, v)] = a * b
    
    # Current state
    sums = set()
    prods = set()
    for (u, v) in edges:
        sums.add(sum_map[(u, v)])
        prods.add(prod_map[(u, v)])
    
    current_score = max(len(sums), len(prods))
    
    # For fast removal, track counts
    sum_counts = {}
    prod_counts = {}
    for (u, v) in edges:
        s = sum_map[(u, v)]
        p = prod_map[(u, v)]
        sum_counts[s] = sum_counts.get(s, 0) + 1
        prod_counts[p] = prod_counts.get(p, 0) + 1
    
    improved = True
    iters = 0
    while improved and iters < max_iters:
        improved = False
        iters += 1
        
        for idx, (u, v) in enumerate(edges):
            s_remove = sum_map[(u, v)]
            p_remove = prod_map[(u, v)]
            
            # Try removing this edge
            sum_counts[s_remove] -= 1
            if sum_counts[s_remove] == 0:
                del sum_counts[s_remove]
            prod_counts[p_remove] -= 1
            if prod_counts[p_remove] == 0:
                del prod_counts[p_remove]
            
            base_sum_count = len(sum_counts)
            base_prod_count = len(prod_counts)
            
            # Try adding each non-edge
            best_new_edge = None
            best_new_score = current_score
            
            for (x, y) in all_e:
                if (x, y) in edge_set:
                    continue
                
                s_add = sum_map[(x, y)]
                p_add = prod_map[(x, y)]
                
                new_sum_count = base_sum_count + (1 if s_add not in sum_counts else 0)
                new_prod_count = base_prod_count + (1 if p_add not in prod_counts else 0)
                
                new_score = max(new_sum_count, new_prod_count)
                
                if new_score < best_new_score:
                    best_new_score = new_score
                    best_new_edge = (x, y)
            
            # Restore removed edge
            sum_counts[s_remove] = sum_counts.get(s_remove, 0) + 1
            prod_counts[p_remove] = prod_counts.get(p_remove, 0) + 1
            
            if best_new_edge is not None and best_new_score < current_score:
                # Perform the swap
                edge_set.discard((u, v))
                edge_set.add(best_new_edge)
                edges[idx] = best_new_edge
                
                # Update counts
                s_add = sum_map[best_new_edge]
                p_add = prod_map[best_new_edge]
                sum_counts[s_add] = sum_counts.get(s_add, 0) + 1
                prod_counts[p_add] = prod_counts.get(p_add, 0) + 1
                
                current_score = best_new_score
                improved = True
                break
        
        if not improved:
            break
    
    return edges, current_score

def _greedy_edges(values, target_m):
    """Greedy edge selection with fast scoring."""
    n = len(values)
    all_e = _all_edges(n)
    
    # Precompute
    sum_map = {}
    prod_map = {}
    for (u, v) in all_e:
        a, b = values[u], values[v]
        sum_map[(u, v)] = a + b
        prod_map[(u, v)] = a * b
    
    sums = set()
    prods = set()
    edges = []
    edge_set = set()
    
    for _ in range(target_m):
        best_edge = None
        best_score = float('inf')
        
        for (u, v) in all_e:
            if (u, v) in edge_set:
                continue
            
            s = sum_map[(u, v)]
            p = prod_map[(u, v)]
            
            new_s = 1 if s not in sums else 0
            new_p = 1 if p not in prods else 0
            
            score = max(len(sums) + new_s, len(prods) + new_p)
            
            if score < best_score:
                best_score = score
                best_edge = (u, v)
        
        if best_edge is None:
            break
        
        u, v = best_edge
        edges.append((u, v))
        edge_set.add((u, v))
        sums.add(sum_map[(u, v)])
        prods.add(prod_map[(u, v)])
    
    return edges

def _generate_smooth_values():
    """Generate smooth number value sets (highly composite, many common factors)."""
    candidates = []
    
    # Strategy 1: Multiples of a highly composite number
    for g in [6, 12, 24, 30, 36, 48, 60, 72, 84, 96, 120, 144, 168, 180, 240]:
        for base in range(1, 10):
            for d in range(1, 5):
                vals = [g * (base + i * d) for i in range(14)]
                if max(vals) <= 10000 and len(set(vals)) == 14 and min(vals) >= 1:
                    candidates.append(vals)
    
    # Strategy 2: Geometric progression with rational ratio
    for g in [2, 4, 6, 8, 10, 12, 15, 20, 24, 30]:
        for base in range(2, 8):
            for num in [3, 4, 5, 6]:
                for den in [2, 3]:
                    if num == den:
                        continue
                    vals = []
                    cur = g * base
                    for _ in range(14):
                        vals.append(cur)
                        cur = (cur * num) // den
                    if max(vals) <= 10000 and len(set(vals)) == 14 and min(vals) >= 1:
                        candidates.append(vals)
    
    # Strategy 3: Powers of 2 times small integers
    for g in [1, 2, 3, 4, 5, 6, 8, 10, 12]:
        for offset in range(0, 4):
            vals = [g * (2**i + offset) for i in range(14)]
            if max(vals) <= 10000 and len(set(vals)) == 14 and min(vals) >= 1:
                candidates.append(vals)
    
    # Strategy 4: Quadratic sequences
    for g in [2, 4, 6, 8, 10, 12, 15, 20]:
        for a in range(1, 6):
            for b in range(1, 6):
                vals = [g * (a * i * i + b * i + 1) for i in range(14)]
                if max(vals) <= 10000 and len(set(vals)) == 14 and min(vals) >= 1:
                    candidates.append(vals)
    
    return candidates

def _try_value_set(values, all_e):
    """Try a specific value set: greedy + local search."""
    if len(set(values)) != 14:
        return None
    if any(not isinstance(v, int) or v < 1 or v > 10000 for v in values):
        return None
    
    edges = _greedy_edges(values, 35)
    if len(edges) < 35:
        return None
    
    edges, score = _fast_local_search(values, edges, all_e, max_iters=100)
    return score, values, edges

def construct():
    start_time = time.time()
    best_score = float('inf')
    best_result = None
    
    n = 14
    all_e = _all_edges(n)
    
    # Phase 1: Try structured smooth value sets
    structured = _generate_smooth_values()
    random.shuffle(structured)
    
    for vals in structured:
        if time.time() > DEADLINE - 10:
            break
        result = _try_value_set(vals, all_e)
        if result:
            s, v, e = result
            if s < best_score:
                best_score = s
                best_result = (list(v), list(e))
    
    # Phase 2: Simulated annealing on value sets
    if time.time() < DEADLINE - 15:
        random.seed(42)
        
        if best_result:
            current_vals = list(best_result[0])
            current_score = best_score
        else:
            current_vals = random.sample(range(1, 10001), 14)
            result = _try_value_set(current_vals, all_e)
            if result:
                current_score, current_vals, _ = result
            else:
                current_score = float('inf')
        
        temp = 3.0
        cooling = 0.93
        
        while time.time() < DEADLINE - 15 and temp > 0.05:
            new_vals = list(current_vals)
            num_mods = random.randint(1, 3)
            for _ in range(num_mods):
                idx = random.randint(0, 13)
                if random.random() < 0.6:
                    delta = random.randint(-30, 30)
                    new_vals[idx] = max(1, min(10000, new_vals[idx] + delta))
                else:
                    new_vals[idx] = random.randint(1, 10000)
            
            if len(set(new_vals)) != 14:
                temp *= cooling
                continue
            
            result = _try_value_set(new_vals, all_e)
            if not result:
                temp *= cooling
                continue
            
            new_score, _, _ = result
            delta = new_score - current_score
            
            if delta < 0 or random.random() < math.exp(-delta / temp):
                current_vals = new_vals
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_result = (list(new_vals), result[2])
            
            temp *= cooling
    
    # Phase 3: Additional local search on best result
    if best_result and time.time() < DEADLINE - 5:
        vals, edges = best_result
        for _ in range(3):
            if time.time() > DEADLINE - 5:
                break
            edges, score = _fast_local_search(vals, edges, all_e, max_iters=50)
            if score < best_score:
                best_score = score
                best_result = (list(vals), list(edges))
    
    if best_result is None:
        vals = list(range(1, 15))
        edges = list(itertools.combinations(range(14), 2))[:35]
        return {'values': vals, 'edges': [list(e) for e in edges]}
    
    vals, edges = best_result
    vals = [int(v) for v in vals]
    edges = [list(e) for e in edges]
    return {'values': vals, 'edges': edges}
