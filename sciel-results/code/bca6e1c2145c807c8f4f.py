<<<<<<< SEARCH
    # Final validation
    if not validate(best_edges):
        # Fallback to known good solution
        best_edges = generate_two_k33()
=======
    # Final validation with exact cycle count
    if not validate(best_edges):
        # Fallback to known good solution
        best_edges = generate_two_k33()
    
    # Verify final answer with exact method
    adj_final = get_adjacency(best_edges)
    exact_lengths = find_distinct_cycle_lengths(adj_final)
    if len(exact_lengths) < best_score:
        best_score = len(exact_lengths)
>>>>>>> REPLACE
