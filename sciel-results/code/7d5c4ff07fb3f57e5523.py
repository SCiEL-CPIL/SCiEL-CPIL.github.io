import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    n = 14
    m = 35
    
    # Precompute all possible edges (indices)
    all_edge_indices = list(combinations(range(n), 2))
    
    best_score = float('inf')
    best_values = None
    best_edges = None
    
    # Generate candidate value sets
    candidates = []
    
    # 1. Consecutive small ints
    candidates.append(list(range(1, 15)))
    
    # 2. Highly composite-ish
    candidates.append([1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 27, 32, 36])
    
    # 3. Random small ints
    random.seed(42)
    for _ in range(10):
        vals = set()
        while len(vals) < n:
            if random.random() < 0.8:
                v = random.randint(1, 30)
            else:
                v = random.randint(1, 100)
            vals.add(v)
        candidates.append(list(vals))
        
    # 4. Arithmetic progression
    for start in [1, 2, 3]:
        for step in [1, 2, 3]:
            vals = [start + i * step for i in range(n)]
            if max(vals) <= 10000 and len(set(vals)) == n:
                candidates.append(vals)
                
    # 5. Multiplicative structure
    candidates.append([2**i for i in range(14)])
    candidates.append([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192])
    
    # 6. Mixed small
    for _ in range(10):
        vals = random.sample(range(1, 50), n)
        candidates.append(vals)
        
    # 7. Specific collision-prone sets
    # e.g. 1,2,3,4,5,6,7,8,9,10,11,12,13,14 is already there.
    # Try 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 16
    candidates.append([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 16])
    candidates.append([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16])

    for vals in candidates:
        if time.time() > deadline:
            break
            
        # Validate
        if len(set(vals)) != n or any(v < 1 or v > 10000 for v in vals):
            continue
            
        # Calculate time remaining
        time_remaining = deadline - time.time()
        if time_remaining < 1.0:
            break
            
        # Allocate iterations based on time remaining
        # Assume 10000 iters per second (very conservative, actually much faster)
        # Let's just use a fixed large number and let the inner loop check time.
        max_iters = 100000 
        
        score, edges = _optimize_edges(vals, all_edge_indices, m, max_iters, deadline)
        
        if score < best_score:
            best_score = score
            best_values = vals
            best_edges = edges
            
    # Fallback
    if best_values is None:
        best_values = list(range(1, 15))
        best_edges = [list(e) for e in all_edge_indices[:m]]
        best_score = _score(best_values, best_edges)
        
    return {'values': [int(v) for v in best_values], 'edges': [list(map(int, e)) for e in best_edges]}

def _score(values, edges):
    sums = set()
    prods = set()
    for u, v in edges:
        sums.add(values[u] + values[v])
        prods.add(values[u] * values[v])
    return max(len(sums), len(prods))

def _optimize_edges(values, all_edge_indices, m, max_iters, deadline):
    n = len(values)
    
    # Precompute sum and product for each edge
    edge_sum = {}
    edge_prod = {}
    for u, v in all_edge_indices:
        edge_sum[(u, v)] = values[u] + values[v]
        edge_prod[(u, v)] = values[u] * values[v]
        
    # Initial random selection
    current_edges = set(random.sample(all_edge_indices, m))
    
    # Initialize counters
    sum_counts = {}
    prod_counts = {}
    
    for e in current_edges:
        s = edge_sum[e]
        p = edge_prod[e]
        sum_counts[s] = sum_counts.get(s, 0) + 1
        prod_counts[p] = prod_counts.get(p, 0) + 1
        
    distinct_sums = len(sum_counts)
    distinct_prods = len(prod_counts)
    current_score = max(distinct_sums, distinct_prods)
    
    best_score = current_score
    best_edges = set(current_edges)
    
    iters = 0
    
    while iters < max_iters:
        # Check time
        if time.time() > deadline - 1.0:
            break
            
        # Pick an edge to remove
        e_remove = random.choice(list(current_edges))
        
        # Pick an edge to add
        while True:
            e_add = random.choice(all_edge_indices)
            if e_add not in current_edges:
                break
                
        # Calculate delta
        s_rem = edge_sum[e_remove]
        p_rem = edge_prod[e_remove]
        s_add = edge_sum[e_add]
        p_add = edge_prod[e_add]
        
        # Apply removal
        sum_counts[s_rem] -= 1
        if sum_counts[s_rem] == 0:
            del sum_counts[s_rem]
            distinct_sums -= 1
            
        prod_counts[p_rem] -= 1
        if prod_counts[p_rem] == 0:
            del prod_counts[p_rem]
            distinct_prods -= 1
            
        # Apply addition
        if s_add in sum_counts:
            sum_counts[s_add] += 1
        else:
            sum_counts[s_add] = 1
            distinct_sums += 1
            
        if p_add in prod_counts:
            prod_counts[p_add] += 1
        else:
            prod_counts[p_add] = 1
            distinct_prods += 1
            
        current_edges.remove(e_remove)
        current_edges.add(e_add)
        
        new_score = max(distinct_sums, distinct_prods)
        
        if new_score < best_score:
            best_score = new_score
            best_edges = set(current_edges)
            
        # Accept if better or equal
        if new_score > current_score:
            # Revert
            # Remove added
            sum_counts[s_add] -= 1
            if sum_counts[s_add] == 0:
                del sum_counts[s_add]
                distinct_sums -= 1
            prod_counts[p_add] -= 1
            if prod_counts[p_add] == 0:
                del prod_counts[p_add]
                distinct_prods -= 1
                
            current_edges.remove(e_add)
            current_edges.add(e_remove)
            
            # Re-add removed
            if s_rem in sum_counts:
                sum_counts[s_rem] += 1
            else:
                sum_counts[s_rem] = 1
                distinct_sums += 1
            if p_rem in prod_counts:
                prod_counts[p_rem] += 1
            else:
                prod_counts[p_rem] = 1
                distinct_prods += 1
                
            current_score = max(distinct_sums, distinct_prods)
        else:
            current_score = new_score
            
        iters += 1
        
    return best_score, [list(e) for e in best_edges]
