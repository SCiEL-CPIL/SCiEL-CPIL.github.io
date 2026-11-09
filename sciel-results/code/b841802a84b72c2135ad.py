import numpy as np

def kissing_number11() -> np.ndarray:
    """
    Returns a set of 220 integer points in R^11 satisfying the kissing number condition:
    min_{i!=j} ||x_i - x_j||^2 >= max_i ||x_i||^2
    
    The construction uses all vectors with exactly two non-zero coordinates, which are +/-1.
    Total points: C(11, 2) * 4 = 55 * 4 = 220.
    Norm squared of each point is 2.
    Min squared distance between distinct points is 2 (when they share one non-zero coordinate with same sign).
    Thus 2 >= 2 holds.
    """
    d = 11
    
    # Generate all pairs of indices (i, j) with i < j
    indices = []
    for i in range(d):
        for j in range(i + 1, d):
            indices.append((i, j))
    
    # Convert to numpy array for vectorized operations
    # indices is a list of tuples, shape (55, 2)
    idx_arr = np.array(indices, dtype=np.int64)
    
    # Signs: -1, 1
    signs = np.array([-1, 1], dtype=np.int64)
    
    # We need to generate 4 variants for each pair:
    # (-1, -1), (-1, 1), (1, -1), (1, 1)
    # This is the cartesian product of signs with itself
    sign_pairs = np.array([(s1, s2) for s1 in signs for s2 in signs], dtype=np.int64)
    # sign_pairs shape: (4, 2)
    
    # We want to create an array of shape (220, 11)
    # Total points = 55 * 4 = 220
    
    # Repeat indices 4 times each
    # Each pair of indices corresponds to 4 sign combinations
    # idx_rep shape: (220, 2)
    idx_rep = np.repeat(idx_arr, 4, axis=0)
    
    # sign_rep shape: (220, 2)
    # sign_pairs is repeated 55 times
    sign_rep = np.tile(sign_pairs, (55, 1))
    
    # Initialize points array with zeros
    points = np.zeros((220, d), dtype=np.int64)
    
    # Assign the values
    # idx_rep[:, 0] are the row indices for the first non-zero coord
    # idx_rep[:, 1] are the row indices for the second non-zero coord
    # sign_rep[:, 0] are the values for the first coord
    # sign_rep[:, 1] are the values for the second coord
    
    points[np.arange(220), idx_rep[:, 0]] = sign_rep[:, 0]
    points[np.arange(220), idx_rep[:, 1]] = sign_rep[:, 1]
    
    # Verification
    # Check shape
    if points.shape != (220, 11):
        raise ValueError("Wrong shape")
    
    # Check no zero vectors
    norms_sq = np.sum(points * points, axis=1)
    if np.any(norms_sq == 0):
        raise ValueError("Zero vector found")
    
    # Check min dist >= max norm
    # Max norm sq is 2
    # Min dist sq should be 2
    
    # Vectorized distance check is O(N^2 * D), 220^2 * 11 ~ 500k ops, very fast
    # dist_sq[i, j] = ||points[i] - points[j]||^2
    # We only need to check i != j
    
    # For efficiency in verification, we can just check a few or use the known property.
    # But to be safe against bugs in generation:
    
    # Calculate pairwise distances
    # diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    # dists_sq = np.sum(diff * diff, axis=2)
    
    # This creates a 220x220 matrix.
    # dists_sq[i, i] = 0
    # We want min(dists_sq[i, j] for i != j) >= max(norms_sq)
    
    # Mask diagonal
    mask = ~np.eye(220, dtype=bool)
    min_dist_sq = np.min(dists_sq[mask]) if 'dists_sq' in locals() else 0
    
    # Actually, let's just compute it to be sure.
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    
    actual_min_dist = np.min(dists_sq)
    actual_max_norm = np.max(norms_sq)
    
    if actual_min_dist < actual_max_norm:
        raise ValueError(f"Constraint violated: min_dist {actual_min_dist} < max_norm {actual_max_norm}")
        
    return points
