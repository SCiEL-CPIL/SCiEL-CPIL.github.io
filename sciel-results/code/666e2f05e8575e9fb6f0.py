import time
import random
from itertools import combinations

P = {'n': 14, 'm': 35, 'maximum': 10000}

def get_score(values, edges):
    sums = set()
    prods = set()
    for i, j in edges:
        sums.add(values[i] + values[j])
        prods.add(values[i] * values[j])
    return max(len(sums), len(prods))

def greedy_select_edges(values):
    n = len(values)
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
            
            if new_max < best_score or (new_max == best_score and (delta_s + delta_p) < best_delta_sum + best_delta_prod):
                best_score = new_max
                best_idx = idx
                best_delta_sum = delta_s
                best_delta_prod = delta_p
        
        if best_idx != -1:
            i, j, s, p = all_edges[best_idx]
            chosen_edges.append([i, j])
            chosen_sums.add(s)
            chosen_prods.add(p)
            remaining.remove(best_idx)
            
    return chosen_edges, max(len(chosen_sums), len(chosen_prods))

def local_search_edges(values, initial_edges):
    n = len(values)
    # Precompute sums and products for all possible edges
    edge_data = {}
    for i in range(n):
        for j in range(i+1, n):
            edge_data[(i,j)] = (values[i] + values[j], values[i] * values[j])
    
    current_edges = set(tuple(e) for e in initial_edges)
    
    # Initialize counts
    sum_counts = {}
    prod_counts = {}
    for e in current_edges:
        s, p = edge_data[e]
        sum_counts[s] = sum_counts.get(s, 0) + 1
        prod_counts[p] = prod_counts.get(p, 0) + 1
        
    distinct_sums = set(sum_counts.keys())
    distinct_prods = set(prod_counts.keys())
    current_score = max(len(distinct_sums), len(distinct_prods))
    
    all_possible_edges = [tuple(e) for e in combinations(range(n), 2)]
    
    improved = True
    while improved:
        improved = False
        best_move = None
        best_score = current_score
        
        # Try all moves: remove one edge, add one non-edge
        for e_out in list(current_edges):
            s_out, p_out = edge_data[e_out]
            
            # Temporarily remove
            sum_counts[s_out] -= 1
            if sum_counts[s_out] == 0:
                del sum_counts[s_out]
                distinct_sums.discard(s_out)
            else:
                # No change in distinct set, but count changed
                pass
                
            prod_counts[p_out] -= 1
            if prod_counts[p_out] == 0:
                del prod_counts[p_out]
                distinct_prods.discard(p_out)
            else:
                pass
            
            # Try adding e_in
            for e_in in all_possible_edges:
                if e_in in current_edges:
                    continue
                
                s_in, p_in = edge_data[e_in]
                
                # Check if adding this would improve
                new_distinct_sums = len(distinct_sums) + (0 if s_in in sum_counts else 1)
                new_distinct_prods = len(distinct_prods) + (0 if p_in in prod_counts else 1)
                
                new_score = max(new_distinct_sums, new_distinct_prods)
                
                if new_score < best_score:
                    best_score = new_score
                    best_move = (e_out, e_in)
            
            # Restore
            sum_counts[s_out] = sum_counts.get(s_out, 0) + 1
            if s_out not in distinct_sums:
                distinct_sums.add(s_out)
                
            prod_counts[p_out] = prod_counts.get(p_out, 0) + 1
            if p_out not in distinct_prods:
                distinct_prods.add(p_out)
        
        if best_move:
            e_out, e_in = best_move
            s_out, p_out = edge_data[e_out]
            s_in, p_in = edge_data[e_in]
            
            # Apply removal
            sum_counts[s_out] -= 1
            if sum_counts[s_out] == 0:
                del sum_counts[s_out]
                distinct_sums.discard(s_out)
                
            prod_counts[p_out] -= 1
            if prod_counts[p_out] == 0:
                del prod_counts[p_out]
                distinct_prods.discard(p_out)
                
            current_edges.remove(e_out)
            
            # Apply addition
            sum_counts[s_in] = sum_counts.get(s_in, 0) + 1
            if s_in not in distinct_sums:
                distinct_sums.add(s_in)
                
            prod_counts[p_in] = prod_counts.get(p_in, 0) + 1
            if p_in not in distinct_prods:
                distinct_prods.add(p_in)
                
            current_edges.add(e_in)
            
            current_score = best_score
            improved = True
            
    return [list(e) for e in current_edges], current_score

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    n = P['n']
    
    initial_sets = [
        list(range(1, n+1)),
        [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192],
        [1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41],
        sorted(random.sample(range(1, 100), n)),
        sorted(random.sample(range(1, 500), n)),
        [2, 4, 6, 8, 9, 10, 12, 14, 15, 16, 18, 20, 21, 22],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 15, 16],
        [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140],
        [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7][:n] # Distinct check needed
    ]
    
    # Ensure distinctness and positive integers
    valid_initial_sets = []
    for s in initial_sets:
        if len(s) == n and len(set(s)) == n and all(x > 0 for x in s):
            valid_initial_sets.append(s)
            
    if not valid_initial_sets:
        valid_initial_sets = [list(range(1, n+1))]

    best_values = None
    best_edges = None
    best_score = float('inf')
    
    # Evaluate initial sets
    for vals in valid_initial_sets:
        if time.time() > deadline: break
        
        # Greedy first to get a good starting point for local search
        edges, score = greedy_select_edges(vals)
        
        # Local search on edges
        edges, score = local_search_edges(vals, edges)
        
        if score < best_score:
            best_score = score
            best_values = list(vals)
            best_edges = edges
            
    # Local Search on Values
    if best_values is not None:
        current_values = list(best_values)
        current_score = best_score
        current_edges = best_edges
        
        rng = random.Random(42)
        no_improve_count = 0
        max_no_improve = 200 # Reduced to allow more initial sets or faster convergence
        
        while time.time() < deadline and no_improve_count < max_no_improve:
            # Random move
            move_type = rng.choice(['swap', 'replace', 'perturb'])
            
            if move_type == 'swap':
                i, j = rng.sample(range(n), 2)
                current_values[i], current_values[j] = current_values[j], current_values[i]
                
                # Check distinctness
                if len(set(current_values)) < n:
                    current_values[i], current_values[j] = current_values[j], current_values[i]
                    no_improve_count += 1
                    continue
                    
                new_edges, new_score = greedy_select_edges(current_values)
                new_edges, new_score = local_search_edges(current_values, new_edges)
                
                if new_score < current_score:
                    current_score = new_score
                    current_edges = new_edges
                    if new_score < best_score:
                        best_score = new_score
                        best_values = list(current_values)
                        best_edges = new_edges
                    no_improve_count = 0
                else:
                    current_values[i], current_values[j] = current_values[j], current_values[i]
                    no_improve_count += 1
                    
            elif move_type == 'replace':
                i = rng.randint(0, n-1)
                old_val = current_values[i]
                new_val = rng.randint(1, 10000)
                
                if new_val in current_values:
                    no_improve_count += 1
                    continue
                    
                current_values[i] = new_val
                
                new_edges, new_score = greedy_select_edges(current_values)
                new_edges, new_score = local_search_edges(current_values, new_edges)
                
                if new_score < current_score:
                    current_score = new_score
                    current_edges = new_edges
                    if new_score < best_score:
                        best_score = new_score
                        best_values = list(current_values)
                        best_edges = new_edges
                    no_improve_count = 0
                else:
                    current_values[i] = old_val
                    no_improve_count += 1
                    
            else: # perturb
                i = rng.randint(0, n-1)
                old_val = current_values[i]
                step = rng.choice([-1, 1, -2, 2, -5, 5, -10, 10])
                new_val = old_val + step
                
                if new_val <= 0 or new_val > 10000 or new_val in current_values:
                    no_improve_count += 1
                    continue
                    
                current_values[i] = new_val
                
                new_edges, new_score = greedy_select_edges(current_values)
                new_edges, new_score = local_search_edges(current_values, new_edges)
                
                if new_score < current_score:
                    current_score = new_score
                    current_edges = new_edges
                    if new_score < best_score:
                        best_score = new_score
                        best_values = list(current_values)
                        best_edges = new_edges
                    no_improve_count = 0
                else:
                    current_values[i] = old_val
                    no_improve_count += 1
            
    # Final validation
    if best_values is None:
        best_values = list(range(1, n+1))
        best_edges, best_score = greedy_select_edges(best_values)
        best_edges, best_score = local_search_edges(best_values, best_edges)
        
    return {'values': [int(v) for v in best_values], 'edges': [[int(a), int(b)] for a, b in best_edges]}
