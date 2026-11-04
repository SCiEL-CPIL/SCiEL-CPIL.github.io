    # Build result
    r_final = _max_radius_for_positions(best_pos)
    
    # Ensure strict feasibility by projecting to feasible radii
    # This handles floating point edge cases where r_i + r_j might slightly exceed d_ij
    _, r_final = _project_feasible(best_pos)
    
    result = np.column_stack([best_pos, r_final])
    return result
