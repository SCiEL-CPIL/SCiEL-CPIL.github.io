def _try_merge_blocks(U, V, W, n, m, p, rank):
    # ...
    best_rank = rank
    best_U, best_V, best_W = [row[:] for row in U], [row[:] for row in V], [row[:] for row in W]
    
    # The previous logic had a bug and did not actually reduce rank.
    # We simply return the original decomposition.
    return best_rank, best_U, best_V, best_W
