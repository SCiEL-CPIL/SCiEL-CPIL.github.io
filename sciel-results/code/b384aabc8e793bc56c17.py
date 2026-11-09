    # Seed solution
    n_seed = 9
    seed_colors = [min(j - i, n_seed - j + i) - 1 for i in range(n_seed) for j in range(i + 1, n_seed)]
    
    max_n_reached = n_seed
    max_colors_snapshot = seed_colors
