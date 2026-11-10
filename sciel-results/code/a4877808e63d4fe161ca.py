import time
import random
from itertools import combinations

P = {'n': 14, 'm': 35, 'maximum': 10000}

def greedy_select_edges(values):
    n = len(values)
    # Precompute all 91 edges with their sums and products
    all_edges = []
    for i in range(n):
        for j in range(i+1, n):
            s = values[i] + values[j]
            p = values[i] * values[j]
            all_edges.append((i, j, s, p))
    
    chosen_edges = []
    chosen_sums = set()
    chosen_prods = set()
    
    remaining = list(range(len(all_edges)))
    
    for _ in range(P['m']):
        best_idx = -1
        best_score = float('inf')
        best_delta_sum = float('inf')
        best_delta_prod = float('inf')
        
        for idx in remaining:
            i, j, s, p = all_edges[idx]
            
            delta_s = 0 if s in chosen_sums else 1
            delta_p = 0 if p in chosen_prods else 1
            
            new_k_s = len(chosen_sums) + delta_s
            new_k_p = len(chosen_prods) + delta_p
            
            new_max = max(new_k_s, new_k_p)
            
            # We want to minimize new_max
            # Tie-break: minimize total new items (delta_s + delta_p)
            if new_max < best_score or (new_max == best_score and (delta_s + delta_p) < best_delta_sum + best_delta_prod):
                best_score = new_max
                best_idx = idx
                best_delta_sum = delta_s
                best_delta_prod = delta_p
        
        # Select the best edge
        if best_idx != -1:
            i, j, s, p = all_edges[best_idx]
            chosen_edges.append([i, j])
            chosen_sums.add(s)
            chosen_prods.add(p)
            remaining.remove(best_idx)
            
    # Calculate final score
    return chosen_edges, max(len(chosen_sums), len(chosen_prods))

def construct():
    start_time = time.time()
    deadline = start_time + 80.0
    
    n = P['n']
    
    # Generate initial value sets
    initial_sets = [
        list(range(1, n+1)),
        [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192],
        [1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41],
        sorted(random.sample(range(1, 100), n)),
        sorted(random.sample(range(1, 500), n)),
        # Small composites
        [2, 4, 6, 8, 9, 10, 12, 14, 15, 16, 18, 20, 21, 22],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 15, 16]
    ]
    
    best_values = None
    best_edges = None
    best_score = float('inf')
    
    # Evaluate initial sets
    for vals in initial_sets:
        if time.time() > deadline: break
        edges, score = greedy_select_edges(vals)
        if score < best_score:
            best_score = score
            best_values = list(vals)
            best_edges = edges
            
    # Local Search on Values
    # We will swap pairs of values and re-run greedy selection
    # To speed up, we can limit the number of swaps or use a timeout
    
    current_values = list(best_values)
    current_score = best_score
    current_edges = best_edges
    
    rng = random.Random(42)
    no_improve_count = 0
    max_no_improve = 5000
    
    while time.time() < deadline and no_improve_count < max_no_improve:
        i, j = rng.sample(range(n), 2)
        
        # Swap
        current_values[i], current_values[j] = current_values[j], current_values[i]
        
        # Validate constraints (distinct, 1..10000)
        # Swapping preserves distinctness. Range is preserved if initial was valid.
        # We should ensure initial sets were valid.
        
        new_edges, new_score = greedy_select_edges(current_values)
        
        if new_score < current_score:
            current_score = new_score
            current_edges = new_edges
            if new_score < best_score:
                best_score = new_score
                best_values = list(current_values)
                best_edges = new_edges
            no_improve_count = 0
        else:
            # Revert
            current_values[i], current_values[j] = current_values[j], current_values[i]
            no_improve_count += 1
            
    # Final validation
    if best_values is None:
        best_values = list(range(1, n+1))
        best_edges, best_score = greedy_select_edges(best_values)
        
    return {'values': [int(v) for v in best_values], 'edges': [[int(a), int(b)] for a, b in best_edges]}
