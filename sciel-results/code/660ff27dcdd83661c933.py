<<<<<<< SEARCH
def _local_search_delta(c, time_limit, ap_indices, ap_pos_idx):
    """Improve coloring via single-flip local search using vectorized delta."""
    start = time.time()
    current = _count_mono4(c)
    improved = True
    while improved:
        improved = False
        for i in range(N):
            if time.time() - start > time_limit:
                break
            delta = _delta_flip_vec(c, i, ap_indices, ap_pos_idx)
            if delta < 0:
                c[i] = 1 - c[i]
                current += delta
                improved = True
        if time.time() - start > time_limit:
            break
    return c, current
=======
def _local_search_delta(c, time_limit, ap_indices, ap_pos_idx):
    """Improve coloring via single-flip local search using vectorized delta."""
    start = time.time()
    current = _count_mono4_fast(c, ap_indices)
    improved = True
    while improved:
        improved = False
        deltas = _all_deltas_fast(c, ap_indices, ap_pos_idx)
        # Flip all positions with negative delta simultaneously (greedy batch)
        flip_mask = deltas < 0
        if np.any(flip_mask):
            c[flip_mask] = 1 - c[flip_mask]
            current += int(np.sum(deltas[flip_mask]))
            improved = True
        if time.time() - start > time_limit:
            break
    # Final polish: one more pass
    if time.time() - start < time_limit:
        deltas = _all_deltas_fast(c, ap_indices, ap_pos_idx)
        flip_mask = deltas < 0
        if np.any(flip_mask):
            c[flip_mask] = 1 - c[flip_mask]
            current += int(np.sum(deltas[flip_mask]))
    return c, current
>>>>>>> REPLACE

