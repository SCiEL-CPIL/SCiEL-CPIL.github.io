"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import random
import math

def construct():
    n = 14
    m = 35
    max_val = 10000
    
    best_score = float('inf')
    best_result = None
    
    # Candidate value sets
    candidate_values = []
    
    # 1. Standard AP 1..14
    candidate_values.append(list(range(1, 15)))
    
    # 2. Standard AP 2..15
    candidate_values.append(list(range(2, 16)))
    
    # 3. Powers of 2 (bad for distinct count, but good for structure)
    candidate_values.append([2**i for i in range(14)])
    
    # 4. Shifted AP
    candidate_values.append(list(range(5, 19)))
    
    # 5. Values with many small factors (divisor rich)
    candidate_values.append([1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20])
    
    # 6. Random small integers (distinct, 1-30)
    random.seed(42)
    for _ in range(20):
        vals = random.sample(range(1, 31), 14)
        candidate_values.append(vals)
        
    # 7. Random small integers (distinct, 1-50)
    for _ in range(20):
        vals = random.sample(range(1, 51), 14)
        candidate_values.append(vals)
        
    # 8. Dense small sets
    for start in range(1, 10):
        candidate_values.append(list(range(start, start + 14)))

    all_edges = list(itertools.combinations(range(n), 2))
    edge_indices = list(range(len(all_edges)))
    
    for values in candidate_values:
        # Precompute sums and products for all edges
        edge_sums = [0] * len(all_edges)
        edge_prods = [0] * len(all_edges)
        for idx, (i, j) in enumerate(all_edges):
            edge_sums[idx] = values[i] + values[j]
            edge_prods[idx] = values[i] * values[j]
        
        # Local Search / Simulated Annealing to select 35 edges
        best_local_score = float('inf')
        best_local_edges = None
        
        # Multiple restarts
        for restart in range(30):
            random.seed(restart * 100 + len(values) + 1)
            
            # Initial selection strategy:
            # 1. Random
            # 2. Smallest sums
            # 3. Largest sums
            # 4. Median sums
            
            if restart % 4 == 0:
                selected_indices = set(random.sample(edge_indices, m))
            elif restart % 4 == 1:
                # Sort by sum, take first 35
                sorted_by_sum = sorted(edge_indices, key=lambda x: edge_sums[x])
                selected_indices = set(sorted_by_sum[:m])
            elif restart % 4 == 2:
                # Sort by sum, take last 35
                sorted_by_sum = sorted(edge_indices, key=lambda x: edge_sums[x])
                selected_indices = set(sorted_by_sum[-m:])
            else:
                # Sort by sum, take middle 35
                sorted_by_sum = sorted(edge_indices, key=lambda x: edge_sums[x])
                mid = len(sorted_by_sum) // 2
                selected_indices = set(sorted_by_sum[mid - m//2 : mid + m//2 + 1])
                
            def get_score(selected_set):
                sums = set()
                prods = set()
                for idx in selected_set:
                    sums.add(edge_sums[idx])
                    prods.add(edge_prods[idx])
                return max(len(sums), len(prods))
            
            current_score = get_score(selected_indices)
            if current_score < best_local_score:
                best_local_score = current_score
                best_local_edges = set(selected_indices)
                
            # Simulated Annealing
            T = 10.0
            cooling = 0.95
            steps = 1000
            
            for step in range(steps):
                # Pick a random edge to remove
                remove_idx = random.choice(list(selected_indices))
                # Pick a random edge to add (not currently selected)
                add_idx = random.choice(edge_indices)
                while add_idx in selected_indices:
                    add_idx = random.choice(edge_indices)
                
                new_selected = set(selected_indices)
                new_selected.remove(remove_idx)
                new_selected.add(add_idx)
                
                new_score = get_score(new_selected)
                
                if new_score < current_score:
                    selected_indices = new_selected
                    current_score = new_score
                    if current_score < best_local_score:
                        best_local_score = current_score
                        best_local_edges = set(selected_indices)
                else:
                    # Accept with probability
                    delta = new_score - current_score
                    if delta > 0:
                        p = math.exp(-delta / T)
                        if random.random() < p:
                            selected_indices = new_selected
                            current_score = new_score
                    else:
                        # If new_score == current_score, delta=0, p=1
                        selected_indices = new_selected
                        current_score = new_score
                        
                T *= cooling
                
                # Early stop if very good
                if best_local_score <= 5:
                    break

        if best_local_edges is not None:
            # Convert back to vertex pairs
            edges = [list(all_edges[idx]) for idx in best_local_edges]
            # Ensure types are correct
            result = {
                'values': list(values),
                'edges': edges
            }
            
            # Calculate final score for safety
            final_sums = set()
            final_prods = set()
            for i, j in edges:
                final_sums.add(values[i] + values[j])
                final_prods.add(values[i] * values[j])
            final_score = max(len(final_sums), len(final_prods))
            
            if final_score < best_score:
                best_score = final_score
                best_result = result

    if best_result is None:
        # Fallback
        values = list(range(1, 15))
        edges = [list(e) for e in all_edges[:m]]
        return {'values': values, 'edges': edges}
    
    return best_result
# EVOLVE-BLOCK-END
