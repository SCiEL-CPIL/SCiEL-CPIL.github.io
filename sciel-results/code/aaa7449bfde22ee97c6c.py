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

def collision_aware_edge_selection(values):
    """Select edges that maximize collisions with already-selected sum/product values."""
    n = len(values)
    all_edges = []
    for i in range(n):
        for j in range(i+1, n):
            s = values[i] + values[j]
            p = values[i] * values[j]
            all_edges.append((i, j, s, p))
    
    # Sort edges by product (smaller products have more collision potential)
    # and by sum (smaller sums have more collision potential)
    all_edges.sort(key=lambda x: (x[3], x[2]))
    
    chosen_edges = []
    chosen_sums = set()
    chosen_prods = set()
    
    # Phase 1: Greedily pick edges that create the most collisions
    remaining = list(range(len(all_edges)))
    
    for _ in range(P['m']):
        best_idx = -1
        best_collision_count = -1
        best_new_sum = False
        best_new_prod = False
        best_total_new = float('inf')
        
        for idx in remaining:
            i, j, s, p = all_edges[idx]
            
            # Count how many existing edges share this sum or product
            sum_collision = 1 if s in chosen_sums else 0
            prod_collision = 1 if p in chosen_prods else 0
            total_collision = sum_collision + prod_collision
            
            new_sum = s not in chosen_sums
            new_prod = p not in chosen_prods
            total_new = (1 if new_sum else 0) + (1 if new_prod else 0)
            
            # Prefer edges that collide with existing values
            if (total_collision > best_collision_count or
                (total_collision == best_collision_count and total_new < best_total_new)):
                best_collision_count = total_collision
                best_idx = idx
                best_total_new = total_new
                best_new_sum = new_sum
                best_new_prod = new_prod
        
        if best_idx != -1:
            i, j, s, p = all_edges[best_idx]
            chosen_edges.append([i, j])
            chosen_sums.add(s)
            chosen_prods.add(p)
            remaining.remove(best_idx)
            
    return chosen_edges, max(len(chosen_sums), len(chosen_prods))

def local_search_edges(values, initial_edges):
    """Local search to improve edge selection by swapping edges."""
    n = len(values)
    edge_data = {}
    for i in range(n):
        for j in range(i+1, n):
            edge_data[(i,j)] = (values[i] + values[j], values[i] * values[j])
    
    current_edges = set(tuple(e) for e in initial_edges)
    
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
        
        for e_out in list(current_edges):
            s_out, p_out = edge_data[e_out]
            
            sum_counts[s_out] -= 1
            if sum_counts[s_out] == 0:
                del sum_counts[s_out]
                distinct_sums.discard(s_out)
            prod_counts[p_out] -= 1
            if prod_counts[p_out] == 0:
                del prod_counts[p_out]
                distinct_prods.discard(p_out)
            
            for e_in in all_possible_edges:
                if e_in in current_edges:
                    continue
                
                s_in, p_in = edge_data[e_in]
                new_distinct_sums = len(distinct_sums) + (0 if s_in in sum_counts else 1)
                new_distinct_prods = len(distinct_prods) + (0 if p_in in prod_counts else 1)
                new_score = max(new_distinct_sums, new_distinct_prods)
                
                if new_score < best_score:
                    best_score = new_score
                    best_move = (e_out, e_in)
            
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
            
            sum_counts[s_out] -= 1
            if sum_counts[s_out] == 0:
                del sum_counts[s_out]
                distinct_sums.discard(s_out)
            prod_counts[p_out] -= 1
            if prod_counts[p_out] == 0:
                del prod_counts[p_out]
                distinct_prods.discard(p_out)
            current_edges.remove(e_out)
            
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
    
    # Generate value sets that favor multiplicative structure for product collisions
    # Strategy: use values with many shared factors to create product collisions
    initial_sets = [
        # Powers of 2 - many product collisions
        [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192],
        # Small integers with common factors
        [1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 27, 32, 36],
        # Mixed: some powers, some multiples
        [1, 2, 4, 8, 3, 6, 12, 24, 5, 10, 20, 40, 7, 14],
        # Sequential small numbers
        list(range(1, n+1)),
        # Numbers with many divisors
        [6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84],
        # Geometric progression with ratio 2
        [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 10000],
        # Random samples from structured ranges
        sorted(random.sample(range(1, 50), n)),
        sorted(random.sample(range(1, 200), n)),
        # Multiples of small numbers
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 18, 20],
        # Fibonacci-like
        [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377][:n],
    ]
    
    # Ensure distinctness and positive integers
    valid_initial_sets = []
    for s in initial_sets:
        if len(s) == n and len(set(s)) == n and all(isinstance(x, int) and x > 0 for x in s):
            valid_initial_sets.append(list(s))
            
    if not valid_initial_sets:
        valid_initial_sets = [list(range(1, n+1))]

    best_values = None
    best_edges = None
    best_score = float('inf')
    
    # Evaluate initial sets with collision-aware edge selection
    for vals in valid_initial_sets:
        if time.time() > deadline: break
        
        edges, score = collision_aware_edge_selection(vals)
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
        max_no_improve = 150
        
        while time.time() < deadline and no_improve_count < max_no_improve:
            move_type = rng.choice(['swap', 'replace', 'perturb'])
            
            if move_type == 'swap':
                i, j = rng.sample(range(n), 2)
                current_values[i], current_values[j] = current_values[j], current_values[i]
                
                if len(set(current_values)) < n:
                    current_values[i], current_values[j] = current_values[j], current_values[i]
                    no_improve_count += 1
                    continue
                    
                new_edges, new_score = collision_aware_edge_selection(current_values)
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
                
                new_edges, new_score = collision_aware_edge_selection(current_values)
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
                    
            else:  # perturb
                i = rng.randint(0, n-1)
                old_val = current_values[i]
                step = rng.choice([-1, 1, -2, 2, -5, 5, -10, 10])
                new_val = old_val + step
                
                if new_val <= 0 or new_val > 10000 or new_val in current_values:
                    no_improve_count += 1
                    continue
                    
                current_values[i] = new_val
                
                new_edges, new_score = collision_aware_edge_selection(current_values)
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
        best_edges, best_score = collision_aware_edge_selection(best_values)
        best_edges, best_score = local_search_edges(best_values, best_edges)
        
    return {'values': [int(v) for v in best_values], 'edges': [[int(a), int(b)] for a, b in best_edges]}
