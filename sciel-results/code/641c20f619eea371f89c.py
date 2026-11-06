def _score_fast(x):
    c = x.reshape(N, 2)
    if not np.all(np.isfinite(c)):
        return -1.0
    if np.any(np.linalg.norm(c, axis=1) > 1.0):
        return -1.0
    
    # Pairwise distances
    # c[:, np.newaxis, :] - c[np.newaxis, :, :]
    # This creates a 50x50x2 array.
    # Squared distances
    sq_dists = np.sum((c[:, np.newaxis, :] - c[np.newaxis, :, :]) ** 2, axis=-1)
    # We only need i < j
    iu = np.triu_indices(N, k=1)
    min_pair_dist = np.sqrt(np.min(sq_dists[iu]))
    
    # Boundary
    norms = np.linalg.norm(c, axis=1)
    min_boundary_dist = np.min(1.0 - norms)
    
    return np.min(min_pair_dist, min_boundary_dist) / 2.0
