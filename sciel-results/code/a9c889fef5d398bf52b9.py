import numpy as np
import time

D = 11

def construct():
    """
    Constructs a set of nonzero integer vectors in R^11.
    The vectors are all weight-2 vectors (two non-zero entries of +/-1).
    For these vectors, the squared norm is 2.
    The condition ||u - v||^2 >= max_norm_sq (which is 2) reduces to u . v <= 1.
    For distinct weight-2 vectors, the maximum possible dot product is 1 (when they share 1 index with the same sign).
    Thus, all pairs satisfy the condition.
    Total vectors: C(11, 2) * 4 = 55 * 4 = 220.
    """
    vectors = []
    
    # Generate all combinations of 2 indices
    for i in range(D):
        for j in range(i + 1, D):
            # Generate all 4 sign combinations for these two indices
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i] = si
                    v[j] = sj
                    vectors.append(v)
    
    return np.array(vectors, dtype=np.int64)

if __name__ == "__main__":
    start_time = time.time()
    res = construct()
    print(f"Generated {res.shape[0]} vectors in {time.time() - start_time:.4f}s")
    # Quick validation
    if res.shape[0] > 0:
        norms = np.sum(res**2, axis=1)
        max_norm_sq = np.max(norms)
        # Compute pairwise distances squared
        # Note: For 220 vectors, this is fast.
        # dist_sq[i, j] = ||u_i - u_j||^2
        # We check if min dist_sq >= max_norm_sq
        # Using broadcasting might be heavy for large N, but N=220 is fine.
        # dist_sq = np.sum((res[:, np.newaxis, :] - res[np.newaxis, :, :])**2, axis=2)
        # This creates a 220x220x11 array. ~500k elements. Fast.
        diff = res[:, np.newaxis, :] - res[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        np.fill_diagonal(dist_sq, 0) # Ignore self-distance
        min_dist_sq = np.min(dist_sq)
        if min_dist_sq >= max_norm_sq:
            print(f"Valid. Min Dist Sq: {min_dist_sq}, Max Norm Sq: {max_norm_sq}")
        else:
            print(f"INVALID. Min Dist Sq: {min_dist_sq}, Max Norm Sq: {max_norm_sq}")
