<<<<<<< SEARCH
    # Step 2: NNI refinement (simplified for now - just return NJ)
    # TODO: Implement proper NNI with JC ML
    
    return nj_newick
# EVOLVE-BLOCK-END
=======
    # Step 2: NNI refinement with JC ML
    parsed = _parse_newick(nj_newick)
    adj, node_names, leaf_names = _build_adjacency(parsed)
    
    # Compute initial likelihood
    current_ll = _jc_log_likelihood(adj, leaf_names, sequences, labels)
    best_ll = current_ll
    best_adj = {k: set(v) for k, v in adj.items()}
    
    # NNI refinement
    max_iterations = 100
    for iteration in range(max_iterations):
        if time.time() > deadline:
            break
        
        moves = _nni_moves(adj, leaf_names)
        if not moves:
            break
        
        improved = False
        for (u, v, a, b, c, d) in moves:
            if time.time() > deadline:
                break
            
            # Save current state
            saved_adj = {k: set(v) for k, v in adj.items()}
            
            # Apply NNI
            _apply_nni(adj, u, v, a, b, c, d)
            
            # Compute new likelihood
            new_ll = _jc_log_likelihood(adj, leaf_names, sequences, labels)
            
            if new_ll > current_ll + 1e-10:
                current_ll = new_ll
                improved = True
                if current_ll > best_ll:
                    best_ll = current_ll
                    best_adj = {k: set(v) for k, v in adj.items()}
            else:
                # Restore
                adj = saved_adj
        
        if not improved:
            break
    
    # Convert best tree to Newick
    if best_adj:
        best_newick = _to_newick_from_adj(best_adj, leaf_names)
        return best_newick
    
    return nj_newick
# EVOLVE-BLOCK-END
>>>>>>> REPLACE
