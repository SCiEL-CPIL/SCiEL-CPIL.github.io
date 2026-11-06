    # Map old indices to new indices 0..39
    # best_edges is a list of [u, v] where u, v are in 0..56
    # We need to know which 40 indices were chosen.
    # We stored best_edges, but we also need the mapping.
    
    # Let's store the subset_indices that gave the best result.
    best_subset = None
    
    # ... inside loop ...
    if count > best_count:
        best_count = count
        best_edges = edges_in_subset
        best_subset = subset_indices
        
    # After loop
    if best_subset is None:
        return []
        
    # Create mapping
    old_to_new = {old: new for new, old in enumerate(best_subset)}
    
    final_edges = []
    for u, v in best_edges:
        final_edges.append([old_to_new[u], old_to_new[v]])
        
    return final_edges
