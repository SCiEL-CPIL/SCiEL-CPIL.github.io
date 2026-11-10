    # Greedy initialization
    greedy_set = []
    greedy_mask = initial_mask
    greedy_sum = 0.0
    greedy_rem = initial_rem_weight
    for v in sorted_vertices:
        if (greedy_mask >> v) & 1:
            greedy_set.append(v)
            greedy_sum += weights[v]
            greedy_mask &= ~(1 << v) & ~neighbors_mask[v]
            # Update greedy_rem is not strictly needed for initialization, just for the bound in B&B
            # But we can just use greedy_sum as initial best_sum
    best_sum = greedy_sum
    # Reconstruct best_mask from greedy_set
    best_mask = 0
    for v in greedy_set:
        best_mask |= (1 << v)
