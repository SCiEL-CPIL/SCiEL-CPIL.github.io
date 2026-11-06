<<<<<<< SEARCH
    # Initial coloring: (i + j) % 17
    init_colors = [(i + j) % 17 for i, j in edges]
    current_colors = init_colors[:]
=======
    # Initial coloring: (i * j) % 19 for multiplicative structure
    # This differs from additive and may give better color diversity in K4s
    init_colors = [(i * j) % 19 for i, j in edges]
    current_colors = init_colors[:]
    
    # Verify initial coloring is valid; if not, fall back to additive
    def _count_distinct(k_idx):
        return len(k4_color_counts[k_idx])
    
    # Quick check: count distinct colors in each K4
    all_valid = True
    for k_idx in range(num_k4s):
        if len(k4_color_counts[k_idx]) < 5:
            all_valid = False
            break
    
    if not all_valid:
        # Fall back to additive coloring
        current_colors = [(i + j) % 17 for i, j in edges]
        for k_idx, edge_idxs in enumerate(k4_edge_indices):
            counts = {}
            for e_idx in edge_idxs:
                c = current_colors[e_idx]
                counts[c] = counts.get(c, 0) + 1
            k4_color_counts[k_idx] = counts
>>>>>>> REPLACE

