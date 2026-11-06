def compute_alpha_fast(adj_mask, n, limit=11):
    """
    Returns True if alpha < limit.
    Uses greedy coloring of the complement graph.
    """
    # Build complement adj masks
    comp_mask = [0] * n
    for u in range(n):
        # Complement of adj_mask[u] within 0..n-1, excluding u
        all_mask = (1 << n) - 1
        comp_mask[u] = all_mask & ~(adj_mask[u] | (1 << u))

    def greedy_color(order):
        colors = [-1] * n
        color_used = 0
        for u in order:
            # Find available colors
            used_colors = set()
            m = comp_mask[u]
            while m:
                lsb = m & (-m)
                v = lsb.bit_length() - 1
                m ^= lsb
                if colors[v] != -1:
                    used_colors.add(colors[v])
            
            c = 0
            while c in used_colors:
                c += 1
            
            colors[u] = c
            if c > color_used:
                color_used = c
            if color_used >= limit - 1:
                return color_used + 1 # Exceeded limit
        return color_used + 1

    # Try several random orderings
    # Sort by degree in complement (descending) is a good heuristic
    degs = [comp_mask[i].bit_count() for i in range(n)]
    sorted_idx = sorted(range(n), key=lambda i: degs[i], reverse=True)
    
    # 1. Deterministic order
    if greedy_color(sorted_idx) < limit:
        return True

    # 2. Random orders
    import random
    for _ in range(20):
        random.shuffle(sorted_idx)
        if greedy_color(sorted_idx) < limit:
            return True
            
    return False
