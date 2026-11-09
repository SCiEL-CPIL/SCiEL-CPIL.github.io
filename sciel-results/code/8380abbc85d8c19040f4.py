def _optimize_edges(values):
    n = len(values)
    all_edges = list(combinations(range(n), 2))
    pair_data = {}
    for i, j in all_edges:
        pair_data[(i, j)] = (values[i] + values[j], values[i] * values[j])
        
    # Greedy Initialization
    selected_edges = []
    sum_counts = {}
    prod_counts = {}
    current_sums = set()
    current_prods = set()
    
    remaining_edges = list(all_edges)
    random.shuffle(remaining_edges)
    
    for _ in range(35):
        best_edge = None
        best_score = float('inf')
        best_tie = float('inf')
        
        for e in remaining_edges:
            s, p = pair_data[e]
            
            # Calculate new counts
            new_sums_count = len(current_sums) + (0 if s in current_sums else 1)
            new_prods_count = len(current_prods) + (0 if p in current_prods else 1)
            
            score = max(new_sums_count, new_prods_count)
            tie = new_sums_count + new_prods_count
            
            if score < best_score or (score == best_score and tie < best_tie):
                best_score = score
                best_edge = e
                best_tie = tie
                
        if best_edge is None:
            break
            
        selected_edges.append(best_edge)
        remaining_edges.remove(best_edge)
        
        s, p = pair_data[best_edge]
        current_sums.add(s)
        current_prods.add(p)
        sum_counts[s] = sum_counts.get(s, 0) + 1
        prod_counts[p] = prod_counts.get(p, 0) + 1
        
    # Local Search on Edges
    selected_set = set(selected_edges)
    non_selected = [e for e in all_edges if e not in selected_set]
    
    improved = True
    max_iters = 50
    iters = 0
    
    while improved and iters < max_iters:
        improved = False
        iters += 1
        
        best_improvement = None
        best_new_score = max(len(current_sums), len(current_prods))
        
        # Try swaps
        for i, e_out in enumerate(selected_edges):
            for e_in in non_selected:
                # Simulate swap
                s_out, p_out = pair_data[e_out]
                s_in, p_in = pair_data[e_in]
                
                # Check if swap is valid (e_out != e_in)
                if e_out == e_in:
                    continue
                
                # Update counts temporarily
                # Remove e_out
                sum_counts[s_out] -= 1
                prod_counts[p_out] -= 1
                if sum_counts[s_out] == 0:
                    current_sums.discard(s_out)
                    del sum_counts[s_out]
                if prod_counts[p_out] == 0:
                    current_prods.discard(p_out)
                    del prod_counts[p_out]
                
                # Add e_in
                new_sums_count = len(current_sums) + (0 if s_in in current_sums else 1)
                new_prods_count = len(current_prods) + (0 if p_in in current_prods else 1)
                
                score = max(new_sums_count, new_prods_count)
                
                if score < best_new_score:
                    best_new_score = score
                    best_improvement = (i, e_in)
                    
                # Rollback
                if s_out in sum_counts:
                    sum_counts[s_out] += 1
                    current_sums.add(s_out)
                else:
                    sum_counts[s_out] = 1
                    current_sums.add(s_out)
                    
                if p_out in prod_counts:
                    prod_counts[p_out] += 1
                    current_prods.add(p_out)
                else:
                    prod_counts[p_out] = 1
                    current_prods.add(p_out)
                    
                # Undo add of e_in (it wasn't actually added, just checked)
                # But wait, if e_in was already in the set, we didn't change counts for it in the "add" step 
                # because we only checked "if s_in in current_sums".
                # We did NOT update sum_counts for s_in in the simulation.
                # So rollback for e_in is not needed for counts.
                
        if best_improvement:
            improved = True
            idx, new_edge = best_improvement
            old_edge = selected_edges[idx]
            
            # Apply swap
            s_out, p_out = pair_data[old_edge]
            s_in, p_in = pair_data[new_edge]
            
            # Remove old
            sum_counts[s_out] -= 1
            if sum_counts[s_out] == 0:
                current_sums.discard(s_out)
                del sum_counts[s_out]
            prod_counts[p_out] -= 1
            if prod_counts[p_out] == 0:
                current_prods.discard(p_out)
                del prod_counts[p_out]
                
            # Add new
            if s_in not in sum_counts:
                sum_counts[s_in] = 1
                current_sums.add(s_in)
            else:
                sum_counts[s_in] += 1
                
            if p_in not in prod_counts:
                prod_counts[p_in] = 1
                current_prods.add(p_in)
            else:
                prod_counts[p_in] += 1
                
            selected_edges[idx] = new_edge
            # Update non_selected
            selected_set.remove(old_edge)
            selected_set.add(new_edge)
            non_selected.remove(new_edge)
            non_selected.append(old_edge)

    return selected_edges, max(len(current_sums), len(current_prods))
