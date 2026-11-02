    # Strassen blocks for k pairs
    for j_start in [0, 2]:
        for k_start in [0, 2, 4]:
            add_strassen_block(0, j_start, k_start)
    
    # Global singleton for k=6
    add_global_singleton_k_block(0, 6)
