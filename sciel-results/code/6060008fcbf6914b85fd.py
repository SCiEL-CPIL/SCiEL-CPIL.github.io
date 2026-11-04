    g_mask, g_count = greedy_mis(candidates)
    if current_count + g_count > best_size:
        best_size = current_count + g_count
        best_mask = current_mask | g_mask
    