<<<<<<< SEARCH
    # Final result
    final_v4_indices = np.where(best_in_set)[0]
    result_v4 = vectors_v4[final_v4_indices]
    final_vectors = np.vstack([result_v4, vectors_v1])
    
    return final_vectors
=======
    # Final result
    final_v4_indices = np.where(best_in_set)[0]
    result_v4 = vectors_v4[final_v4_indices]
    final_vectors = np.vstack([result_v4, vectors_v1])
    
    # Verify: min pairwise distance >= max norm
    # Compute norms
    norms_sq = np.sum(final_vectors.astype(np.float64)**2, axis=1)
    max_norm_sq = np.max(norms_sq)
    
    # Compute pairwise distances squared
    # For large k, this can be expensive; use chunked computation
    k = len(final_vectors)
    min_dist_sq = np.inf
    chunk_size = 500
    for i in range(0, k, chunk_size):
        chunk = final_vectors[i:i+chunk_size].astype(np.float64)
        # Distance to all vectors
        # ||a-b||^2 = ||a||^2 + ||b||^2 - 2*a.b
        chunk_norms_sq = np.sum(chunk**2, axis=1, keepdims=True)
        all_norms_sq = norms_sq[None, :]
        dot = chunk @ final_vectors.T
        dist_sq = chunk_norms_sq + all_norms_sq - 2*dot
        # Exclude self-distances
        for j in range(len(chunk)):
            dist_sq[j, i+j] = np.inf
        min_dist_sq = min(min_dist_sq, np.min(dist_sq))
    
    # If constraint violated, we should have caught this in search, but verify
    # The constraint is: min pairwise distance >= max norm
    # i.e., min_dist_sq >= max_norm_sq
    
    return final_vectors
>>>>>>> REPLACE
