import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin for 90s limit
    
    n = 14
    m = 35
    values = list(range(1, n + 1))
    
    # Generate all possible edges
    all_edges = list(combinations(range(n), 2))
    num_edges = len(all_edges) # 91
    
    # Precompute sums and products for each edge
    edge_sums = []
    edge_prods = []
    for u, v in all_edges:
        edge_sums.append(values[u] + values[v])
        edge_prods.append(values[u] * values[v])
        
    # Objective function to evaluate a set of edge indices
    def evaluate(edge_indices):
        sum_counts = {}
        prod_counts = {}
        for idx in edge_indices:
            s = edge_sums[idx]
            p = edge_prods[idx]
            sum_counts[s] = sum_counts.get(s, 0) + 1
            prod_counts[p] = prod_counts.get(p, 0) + 1
        
        active_sums = len(sum_counts)
        active_prods = len(prod_counts)
        max_distinct = max(active_sums, active_prods)
        
        # Secondary: minimize number of sums/products that appear only once
        lonely = 0
        for count in sum_counts.values():
            if count == 1:
                lonely += 1
        for count in prod_counts.values():
            if count == 1:
                lonely += 1
                
        return max_distinct, lonely

    # Greedy Initialization
    def greedy_init():
        selected = []
        selected_set = set()
        sum_counts = {}
        prod_counts = {}
        
        for _ in range(m):
            best_idx = -1
            best_score = (float('inf'), float('inf'))
            
            for idx in range(num_edges):
                if idx in selected_set:
                    continue
                
                s = edge_sums[idx]
                p = edge_prods[idx]
                
                # Calculate new distinct counts
                new_active_sums = len(sum_counts) + (1 if sum_counts.get(s, 0) == 0 else 0)
                new_active_prods = len(prod_counts) + (1 if prod_counts.get(p, 0) == 0 else 0)
                new_max = max(new_active_sums, new_active_prods)
                
                # Calculate new lonely counts
                # If s was 0, it becomes 1 (lonely). If s was 1, it becomes 2 (not lonely). If s >= 2, stays not lonely.
                old_s_count = sum_counts.get(s, 0)
                new_s_count = old_s_count + 1
                old_p_count = prod_counts.get(p, 0)
                new_p_count = old_p_count + 1
                
                # Change in lonely sums
                delta_lonely_s = 0
                if old_s_count == 0:
                    delta_lonely_s += 1
                elif old_s_count == 1:
                    delta_lonely_s -= 1
                
                delta_lonely_p = 0
                if old_p_count == 0:
                    delta_lonely_p += 1
                elif old_p_count == 1:
                    delta_lonely_p -= 1
                
                # Current lonely count needs to be tracked. 
                # For simplicity in greedy, we just use the new max distinct. 
                # Tie-breaking: prefer edges that don't increase distinct count.
                # If same max, prefer one that results in lower total distinct? 
                # Let's just use (new_max, new_active_sums, new_active_prods) as proxy.
                
                score = (new_max, new_active_sums + new_active_prods)
                
                if score < best_score:
                    best_score = score
                    best_idx = idx
            
            if best_idx != -1:
                selected.append(best_idx)
                selected_set.add(best_idx)
                s = edge_sums[best_idx]
                p = edge_prods[best_idx]
                sum_counts[s] = sum_counts.get(s, 0) + 1
                prod_counts[p] = prod_counts.get(p, 0) + 1
                
        return selected

    # Local Search
    def local_search(initial_edges):
        selected = set(initial_edges)
        sum_counts = {}
        prod_counts = {}
        for idx in selected:
            s = edge_sums[idx]
            p = edge_prods[idx]
            sum_counts[s] = sum_counts.get(s, 0) + 1
            prod_counts[p] = prod_counts.get(p, 0) + 1
            
        current_max, current_lonely = evaluate(list(selected))
        best_max, best_lonely = current_max, current_lonely
        best_selected = set(selected)
        
        # Precompute list of non-selected edges for faster access? 
        # Since num_edges is 91, iterating all is fine.
        
        while time.time() < deadline:
            improved = False
            # Try all possible swaps: remove i, add j
            # i in selected, j not in selected
            selected_list = list(selected)
            non_selected_list = [i for i in range(num_edges) if i not in selected]
            
            best_swap = None
            best_swap_score = (best_max, best_lonely)
            
            for i in selected_list:
                s_out = edge_sums[i]
                p_out = edge_prods[i]
                
                # Tentatively remove i
                sum_counts[s_out] -= 1
                prod_counts[p_out] -= 1
                if sum_counts[s_out] == 0:
                    del sum_counts[s_out]
                if prod_counts[p_out] == 0:
                    del prod_counts[p_out]
                
                # Current state without i
                active_sums_no_i = len(sum_counts)
                active_prods_no_i = len(prod_counts)
                max_no_i = max(active_sums_no_i, active_prods_no_i)
                
                # Calculate lonely without i
                lonely_no_i = sum(1 for c in sum_counts.values() if c == 1) + \
                              sum(1 for c in prod_counts.values() if c == 1)
                
                for j in non_selected_list:
                    s_in = edge_sums[j]
                    p_in = edge_prods[j]
                    
                    # Tentatively add j
                    new_active_sums = active_sums_no_i + (1 if s_in not in sum_counts else 0)
                    new_active_prods = active_prods_no_i + (1 if p_in not in prod_counts else 0)
                    new_max = max(new_active_sums, new_active_prods)
                    
                    # Calculate new lonely
                    # We need to know if s_in was lonely in sum_counts (count 1) or not.
                    # If s_in in sum_counts:
                    #   if count was 1, it becomes 2 (lonely decreases by 1)
                    #   if count > 1, it becomes count+1 (lonely unchanged)
                    # If s_in not in sum_counts:
                    #   it becomes 1 (lonely increases by 1)
                    
                    old_s_count = sum_counts.get(s_in, 0)
                    old_p_count = prod_counts.get(p_in, 0)
                    
                    delta_lonely_s = 0
                    if old_s_count == 0:
                        delta_lonely_s = 1
                    elif old_s_count == 1:
                        delta_lonely_s = -1
                    
                    delta_lonely_p = 0
                    if old_p_count == 0:
                        delta_lonely_p = 1
                    elif old_p_count == 1:
                        delta_lonely_p = -1
                        
                    new_lonely = lonely_no_i + delta_lonely_s + delta_lonely_p
                    
                    score = (new_max, new_lonely)
                    
                    if score < best_swap_score:
                        best_swap_score = score
                        best_swap = (i, j)
                        
                # Restore i
                sum_counts[s_out] = sum_counts.get(s_out, 0) + 1
                prod_counts[p_out] = prod_counts.get(p_out, 0) + 1
            
            if best_swap:
                i, j = best_swap
                # Apply swap
                s_out = edge_sums[i]
                p_out = edge_prods[i]
                s_in = edge_sums[j]
                p_in = edge_prods[j]
                
                sum_counts[s_out] -= 1
                if sum_counts[s_out] == 0:
                    del sum_counts[s_out]
                prod_counts[p_out] -= 1
                if prod_counts[p_out] == 0:
                    del prod_counts[p_out]
                
                sum_counts[s_in] = sum_counts.get(s_in, 0) + 1
                prod_counts[p_in] = prod_counts.get(p_in, 0) + 1
                
                selected.remove(i)
                selected.add(j)
                
                best_max, best_lonely = best_swap_score
                best_selected = set(selected)
                improved = True
            else:
                # No improvement found in this full scan
                # Try random restarts or perturbations?
                # Given the small size, if no swap improves, we are likely at a local optimum.
                # We can try a few random swaps that don't worsen M to escape plateaus.
                if random.random() < 0.1:
                     # Random sideways move
                     i = random.choice(selected_list)
                     j = random.choice(non_selected_list)
                     s_out = edge_sums[i]
                     p_out = edge_prods[i]
                     s_in = edge_sums[j]
                     p_in = edge_prods[j]
                     
                     # Check if M stays same or improves
                     # Quick check:
                     # This is complex to do inline without full recalc.
                     # Let's just accept the current state and break to save time.
                     # Or, we can just stop. The greedy+local search usually finds a good solution quickly.
                     pass
                break

        return list(best_selected)

    # Run optimization
    initial_edges = greedy_init()
    optimized_edges = local_search(initial_edges)
    
    # Ensure types are correct
    result_edges = [[u, v] for u, v in (all_edges[i] for i in optimized_edges)]
    
    return {
        'values': values,
        'edges': result_edges
    }
