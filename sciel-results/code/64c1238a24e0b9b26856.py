import time
import random
from itertools import combinations

def _evaluate(values, edges):
    """Calculate the raw value: max(distinct sums, distinct products)"""
    sums = set()
    prods = set()
    for u, v in edges:
        s = values[u] + values[v]
        p = values[u] * values[v]
        sums.add(s)
        prods.add(p)
    return max(len(sums), len(prods))

def _greedy_select_edges(values, m):
    """
    Greedily select m edges to minimize max(distinct_sums, distinct_products).
    """
    n = len(values)
    all_edges = list(combinations(range(n), 2))
    
    # Precompute sums and products for all edges
    edge_data = []
    for u, v in all_edges:
        s = values[u] + values[v]
        p = values[u] * values[v]
        edge_data.append((u, v, s, p))
        
    selected_edges = []
    seen_sums = set()
    seen_prods = set()
    
    # We want to pick m edges.
    # Heuristic: pick edges that add the fewest new distinct sums and products.
    # Score an edge by (new_sums, new_prods).
    # To balance, we might prioritize reducing the max.
    
    remaining = list(range(len(all_edges)))
    
    for _ in range(m):
        best_idx = -1
        best_score = (float('inf'), float('inf'))
        
        for idx in remaining:
            u, v, s, p = edge_data[idx]
            new_s = 0 if s in seen_sums else 1
            new_p = 0 if p in seen_prods else 1
            
            # We want to minimize the increase in the max count.
            # Current counts: len(seen_sums), len(seen_prods)
            # New counts: len(seen_sums) + new_s, len(seen_prods) + new_p
            # We want to minimize max(new_sum_count, new_prod_count)
            
            new_max = max(len(seen_sums) + new_s, len(seen_prods) + new_p)
            
            # Tie-breaker: prefer edges with smaller products/sums? Or just deterministic.
            # Let's use new_max as primary, and (new_s, new_p) as secondary to keep counts balanced.
            score = (new_max, new_s, new_p)
            
            if score < best_score:
                best_score = score
                best_idx = idx
                
        if best_idx == -1:
            break
            
        u, v, s, p = edge_data[best_idx]
        selected_edges.append([u, v])
        seen_sums.add(s)
        seen_prods.add(p)
        remaining.remove(best_idx)
        
    return selected_edges

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Leave 5s margin
    
    # Candidate value sets
    # Set 1: Highly composite numbers based on 2 and 3
    set1 = [2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 32, 36, 48, 72]
    
    # Set 2: Include 1? No, 1 creates unique products equal to the other value.
    # Set 3: Randomly perturb set 1?
    # Let's try a few structured sets.
    
    value_sets = [
        [2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 32, 36, 48, 72],
        [2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 32, 36, 54, 72], # Swap 48 for 54?
        [2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 32, 40, 48, 72], # Swap 36 for 40?
        [2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 32, 36, 48, 96], # Swap 72 for 96?
    ]
    
    best_result = None
    best_score = float('inf')
    
    for values in value_sets:
        if time.time() > deadline:
            break
            
        # Ensure distinct and <= 10000
        if len(set(values)) != 14 or max(values) > 10000:
            continue
            
        edges = _greedy_select_edges(values, 35)
        
        if len(edges) < 35:
            continue
            
        score = _evaluate(values, edges)
        
        if score < best_score:
            best_score = score
            best_result = {'values': values, 'edges': edges}
            
        # Local Search: Try to improve the edge set
        # We have 85s, so we can afford some local search.
        # Simple swap: remove an edge, add another.
        
        current_edges = [tuple(e) for e in edges]
        current_edge_set = set(current_edges)
        current_score = score
        
        # Precompute data for all edges again for speed in local search
        n = 14
        all_edges = list(combinations(range(n), 2))
        edge_data = {}
        for u, v in all_edges:
            s = values[u] + values[v]
            p = values[u] * values[v]
            edge_data[(u, v)] = (s, p)
            
        # Limit local search steps based on time
        steps = 0
        max_steps = 1000 # Adjust based on time
        
        while time.time() < deadline - 5 and steps < max_steps:
            steps += 1
            improved = False
            
            # Pick a random edge from current set to remove
            rem_idx = random.randint(0, len(current_edges) - 1)
            rem_edge = current_edges[rem_idx]
            
            # Pick a random edge from all edges not in current set to add
            candidates = [e for e in all_edges if e not in current_edge_set]
            if not candidates:
                break
            add_edge = random.choice(candidates)
            
            # Try the swap
            new_edges = current_edges.copy()
            new_edges[rem_idx] = add_edge
            new_edge_set = set(new_edges)
            
            # Evaluate
            sums = set()
            prods = set()
            for u, v in new_edges:
                s, p = edge_data[(u, v)]
                sums.add(s)
                prods.add(p)
            new_score = max(len(sums), len(prods))
            
            if new_score < current_score:
                current_edges = new_edges
                current_edge_set = new_edge_set
                current_score = new_score
                improved = True
                max_steps += 1000 # Reward improvement by allowing more steps
                
            if time.time() > deadline - 5:
                break
        
        if current_score < best_score:
            best_score = current_score
            best_result = {
                'values': values, 
                'edges': [list(e) for e in current_edges]
            }
            
    if best_result is None:
        # Fallback to baseline-like
        values = list(range(1, 15))
        edges = [list(e) for e in list(combinations(range(14), 2))[:35]]
        best_result = {'values': values, 'edges': edges}
        
    # Ensure types are correct
    best_result['values'] = [int(v) for v in best_result['values']]
    best_result['edges'] = [[int(u), int(v)] for u, v in best_result['edges']]
    
    return best_result
