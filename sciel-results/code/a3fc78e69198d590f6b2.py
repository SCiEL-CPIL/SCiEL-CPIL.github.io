    S = M.T @ s  # Shape (num_aps,)
    # ...
    i = chosen_index
    S -= 2.0 * old_val * M[i]  # M[i] is shape (num_aps,)
