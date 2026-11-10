import random
import time
from itertools import combinations

def _calc_score(vals, edges):
    sums = set()
    prods = set()
    for u, v in edges:
        sums.add(vals[u] + vals[v])
        prods.add(vals[u] * vals[v])
    return max(len(sums), len(prods))

def _optimize_edges(vals, initial_edges, time_limit):
    n = len(vals)
    m = len(initial_edges)
    all_edges = set(combinations(range(n), 2))
    
    # Convert initial edges to set for fast lookup
    edges = set(initial_edges)
    current_score = _calc_score(vals, list(edges))
    
    start_time = time.time()
    
    while time.time() - start_time < time_limit:
        # Generate a random swap
        # Pick an edge to remove
        if not edges:
            continue
        e_remove = random.choice(list(edges))
        
        # Pick a non-edge to add
        # To avoid generating all non-edges every time, we can sample
        # But n=14, total edges 91. m=35. Non-edges = 56.
        # Generating list of non-edges is cheap.
        non_edges = [e for e in all_edges if e not in edges]
        if not non_edges:
            continue
        e_add = random.choice(non_edges)
        
        # Perform swap
        edges.remove(e_remove)
        edges.add(e_add)
        
        new_score = _calc_score(vals, list(edges))
        
        # Acceptance criterion
        if new_score < current_score:
            current_score = new_score
        elif new_score == current_score:
            # Accept with some probability to explore
            if random.random() < 0.3:
                current_score = new_score
            else:
                # Revert
                edges.remove(e_add)
                edges.add(e_remove)
        else:
            # Revert
            edges.remove(e_add)
            edges.add(e_remove)
            
    return list(edges), current_score

def _optimize_values(edges, initial_vals, time_limit):
    n = len(initial_vals)
    max_val = 10000
    vals = list(initial_vals)
    current_score = _calc_score(vals, edges)
    
    start_time = time.time()
    
    while time.time() - start_time < time_limit:
        # Pick a vertex to change
        idx = random.randint(0, n-1)
        old_val = vals[idx]
        
        # Generate new value
        # Bias towards neighbors or random
        if random.random() < 0.5:
            # Neighbor
            step = random.randint(-10, 10)
            new_val = old_val + step
        else:
            # Random
            new_val = random.randint(1, max_val)
            
        # Bounds check
        if new_val < 1 or new_val > max_val:
            continue
            
        # Distinctness check
        if new_val in vals:
            continue
            
        # Apply
        vals[idx] = new_val
        new_score = _calc_score(vals, edges)
        
        # Acceptance
        if new_score < current_score:
            current_score = new_score
        elif new_score == current_score:
            if random.random() < 0.3:
                current_score = new_score
            else:
                vals[idx] = old_val
        else:
            vals[idx] = old_val
            
    return vals, current_score

def construct():
    start_time = time.time()
    deadline = start_time + 55 # Safety margin
    
    n = 14
    m = 35
    
    # Initial candidates
    candidates = []
    candidates.append(list(range(1, n+1))) # AP
    candidates.append([2**i for i in range(n)]) # GP
    for _ in range(10):
        candidates.append(random.sample(range(1, 10001), n))
    
    best_score = float('inf')
    best_result = None
    
    # We will run a series of restarts or a long local search
    # Given the small size, we can afford a decent amount of search.
    
    # Strategy:
    # 1. For each initial value set, optimize edges.
    # 2. Then optimize values.
    # 3. Alternate a few times.
    
    for init_vals in candidates:
        if time.time() > deadline:
            break
            
        # Initialize edges randomly
        all_edges = list(combinations(range(n), 2))
        init_edges = random.sample(all_edges, m)
        
        # Phase 1: Optimize edges for these values
        time_per_phase = (deadline - time.time()) / 4.0
        if time_per_phase < 1.0:
            time_per_phase = 1.0
            
        edges, score = _optimize_edges(init_vals, init_edges, time_per_phase)
        
        # Phase 2: Optimize values for these edges
        vals, score = _optimize_values(edges, init_vals, time_per_phase)
        
        # Phase 3: Re-optimize edges with new values
        if time.time() < deadline:
            edges, score = _optimize_edges(vals, edges, time_per_phase)
            
        # Phase 4: Re-optimize values
        if time.time() < deadline:
            vals, score = _optimize_values(edges, vals, time_per_phase)
            
        if score < best_score:
            best_score = score
            best_result = {
                'values': [int(v) for v in vals],
                'edges': [[int(u), int(v)] for u, v in edges]
            }
            
    # If no result found (shouldn't happen), return a valid default
    if best_result is None:
        vals = list(range(1, n+1))
        edges = [list(e) for e in list(combinations(range(n), 2))[:m]]
        best_result = {
            'values': vals,
            'edges': edges
        }
        
    return best_result
