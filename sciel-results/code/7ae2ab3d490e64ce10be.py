"""Kissing number in dimension 11: construct integer vectors with min pairwise distance >= max norm."""
from scipy.spatial.distance import pdist
import numpy as np
import time

D = 11
DEADLINE = 150.0
START_TIME = time.time()

def construct():
    """
    Return integer array of shape (k, d) where k is the number of vectors.
    The vectors must satisfy: for all i != j, ||v_i - v_j|| >= max_i ||v_i||.
    Score is k.
    """
    # Generate all vectors with exactly two non-zero coordinates (±1)
    # These have norm squared = 2, so max norm squared = 2.
    # Pairwise distance squared between any two such vectors:
    # - If they share no coordinates: distance squared = 4 >= 2 ✓
    # - If they share one coordinate: distance squared = 2 (if same sign at shared) or 4 (if opposite sign) 
    #   Wait: if they share coordinate i, v1[i]=a, v2[i]=b where a,b ∈ {±1}
    #   Then distance squared at coord i is (a-b)^2 = 0 if a=b, 4 if a≠b
    #   Plus the other non-zero coords contribute.
    #   Let's think more carefully.
    #   v1 has non-zeros at {i,j}, v2 has non-zeros at {k,l}
    #   Case 1: {i,j} ∩ {k,l} = ∅: dist^2 = 1+1+1+1 = 4 >= 2 ✓
    #   Case 2: |{i,j} ∩ {k,l}| = 1, say i=k:
    #     v1 = (a, b, 0, ...) at (i,j), v2 = (c, d, 0, ...) at (i,l) with l≠j
    #     dist^2 = (a-c)^2 + b^2 + d^2 = (a-c)^2 + 1 + 1
    #     If a=c: dist^2 = 2 >= 2 ✓
    #     If a≠c: dist^2 = 4 + 2 = 6 >= 2 ✓
    #   So all pairs are valid!
    
    vectors = []
    for i in range(D):
        for j in range(i + 1, D):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i] = si
                    v[j] = sj
                    vectors.append(v)
    
    arr = np.array(vectors, dtype=np.int64)
    k = arr.shape[0]
    
    # Check time and try to add more vectors if possible
    elapsed = time.time() - START_TIME
    if elapsed > DEADLINE - 5.0:
        return arr
    
    # Try to add vectors with 3 non-zero coordinates (±1)
    # These have norm squared = 3, which would increase max norm to sqrt(3)
    # Then we'd need min pairwise distance >= sqrt(3), i.e., dist^2 >= 3
    # This is more restrictive. Let's check if we can do better by mixing.
    # Actually, if we add a norm-3 vector, max norm becomes sqrt(3) ≈ 1.732
    # Then ALL pairs must have dist^2 >= 3
    # Two norm-2 vectors can have dist^2 = 2, which would violate this.
    # So we'd have to remove many norm-2 vectors. Not worth it.
    
    # Try adding vectors of form (±1, ±1, ..., ±1) with 11 non-zeros? Norm squared = 11.
    # Too large.
    
    # What about adding (±1, 0, 0, ...) vectors? Norm squared = 1.
    # These would be fine as long as min distance to others is >= max norm.
    # If max norm is still sqrt(2), then distance from (1,0,...) to (1,1,0,...) is 1 < sqrt(2). Not allowed.
    
    # The 220 vectors with exactly two ±1 entries is optimal for this family.
    # Known kissing number for d=11 is 264 or higher, but constructing that is complex.
    # Return the 220-vector solution.
    
    return arr

# EVOLVE-BLOCK-START
# (kept for compatibility with the original structure)
# EVOLVE-BLOCK-END

if __name__ == "__main__":
    result = construct()
    print(f"Number of vectors: {result.shape[0]}")
    print(f"Shape: {result.shape}")
    # Verify
    if result.shape[0] > 1:
        norms_sq = np.sum(result.astype(float)**2, axis=1)
        max_norm_sq = np.max(norms_sq)
        # Compute pairwise distances
        from scipy.spatial.distance import pdist, squareform
        dists = pdist(result.astype(float))
        min_dist = np.min(dists)
        min_dist_sq = min_dist**2
        print(f"Max norm squared: {max_norm_sq}")
        print(f"Min pairwise distance squared: {min_dist_sq}")
        print(f"Valid: {min_dist_sq >= max_norm_sq - 1e-9}")
