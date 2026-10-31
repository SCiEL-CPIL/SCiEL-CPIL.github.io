"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
from scipy.spatial.distance import cdist

def construct():
    """
    Construct nonzero integer vectors in R^11 using E8⊕A2⊕A1 decomposition.
    
    We generate candidate vectors from the product structure and use a vectorized
    greedy algorithm on the conflict graph with randomized tie-breaking.
    """
    np.random.seed(42)
    
    # Generate E8 minimal vectors (norm^2 = 2) in 8D
    e8_minimal = []
    # Type 1: two ±1's, rest 0
    for i, j in itertools.combinations(range(8), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * 8
                v[i] = si
                v[j] = sj
                e8_minimal.append(v)
    # Type 2: half-integers with even number of minus signs (scale by 2 to get integers)
    # Actually for integer vectors, we need to be careful. Let's use only the ±1,±1 type for now
    # and add more candidates from other sources.
    
    e8_minimal = np.array(e8_minimal, dtype=np.int64)  # 112 vectors
    
    # Generate A2 minimal vectors (norm^2 = 2) in 2D
    # A2 lattice: (a, b) where a+b ≡ 0 mod 2, or use the hexagonal lattice
    # Minimal vectors: (±1, ±1), (±1, ∓1) but that's just ±(1,1), ±(1,-1)
    # Actually A2 minimal vectors with norm^2=2: (±1, ±1) giving 4 vectors, but we need norm^2=2
    # (1,1) has norm^2=2, (-1,1) has norm^2=2, etc.
    a2_minimal = np.array([
        [1, 1], [-1, -1],
        [1, -1], [-1, 1],
        [2, 0], [-2, 0],
        [0, 2], [0, -2]
    ], dtype=np.int64)
    # Filter to only norm^2=2
    a2_norms_sq = np.sum(a2_minimal**2, axis=1)
    a2_minimal = a2_minimal[a2_norms_sq == 2]  # 4 vectors
    
    # Generate A1 minimal vectors (norm^2 = 2) in 1D
    # A1 in 1D: just ±1, but norm^2=1. For norm^2=2, we need ±√2 which isn't integer.
    # Let's use norm^2=1 for A1 and adjust.
    a1_minimal = np.array([[1], [-1]], dtype=np.int64)  # norm^2=1
    
    # Build candidate vectors in R^11 from E8⊕A2⊕A1
    candidates = []
    
    # Case 1: E8 part has norm^2=2, A2 part zero, A1 part zero
    for e8 in e8_minimal:
        v = np.zeros(11, dtype=np.int64)
        v[:8] = e8
        candidates.append(v)
    
    # Case 2: E8 part zero, A2 part has norm^2=2, A1 part zero
    for a2 in a2_minimal:
        v = np.zeros(11, dtype=np.int64)
        v[8:10] = a2
        candidates.append(v)
    
    # Case 3: E8 part zero, A2 part zero, A1 part has norm^2=1 (scale consideration)
    for a1 in a1_minimal:
        v = np.zeros(11, dtype=np.int64)
        v[10] = a1[0]
        candidates.append(v)
    
    # Case 4: E8 part has norm^2=2, A2 part has norm^2=2, A1 part zero
    for e8 in e8_minimal:
        for a2 in a2_minimal:
            v = np.zeros(11, dtype=np.int64)
            v[:8] = e8
            v[8:10] = a2
            candidates.append(v)
    
    # Case 5: E8 part has norm^2=2, A2 part zero, A1 part has norm^2=1
    for e8 in e8_minimal:
        for a1 in a1_minimal:
            v = np.zeros(11, dtype=np.int64)
            v[:8] = e8
            v[10] = a1[0]
            candidates.append(v)
    
    # Case 6: E8 part zero, A2 part has norm^2=2, A1 part has norm^2=1
    for a2 in a2_minimal:
        for a1 in a1_minimal:
            v = np.zeros(11, dtype=np.int64)
            v[8:10] = a2
            v[10] = a1[0]
            candidates.append(v)
    
    # Also add pure weight-2 vectors from all 11 coordinates for completeness
    for i, j in itertools.combinations(range(11), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = np.zeros(11, dtype=np.int64)
                v[i] = si
                v[j] = sj
                candidates.append(v)
    
    # Add weight-4 vectors (four ±1's) for more candidates
    for positions in itertools.combinations(range(11), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(11, dtype=np.int64)
            for pos, sign in zip(positions, signs):
                v[pos] = sign
            candidates.append(v)
    
    candidates = np.array(candidates, dtype=np.int64)
    
    # Remove duplicates
    candidates = np.unique(candidates, axis=0)
    
    # Compute squared norms
    norms_sq = np.sum(candidates**2, axis=1)
    max_norm_sq = int(np.max(norms_sq))
    
    # We need: for all pairs i, j, ||c_i - c_j||^2 >= max_norm_sq
    # ||c_i - c_j||^2 = ||c_i||^2 + ||c_j||^2 - 2 <c_i, c_j> >= max_norm_sq
    
    # Vectorized greedy with randomized tie-breaking
    n = len(candidates)
    selected_mask = np.zeros(n, dtype=bool)
    selected_indices = []
    
    # Precompute all pairwise dot products for efficiency
    # This is n x n, might be large but let's try
    dots = candidates @ candidates.T
    
    # Create a priority queue based on norm (prefer higher norm vectors)
    order = np.argsort(-norms_sq)
    
    for idx in order:
        if selected_mask[idx]:
            continue
        
        # Check compatibility with all selected vectors
        if len(selected_indices) == 0:
            selected_mask[idx] = True
            selected_indices.append(idx)
            continue
        
        sel_idx_array = np.array(selected_indices)
        sel_dots = dots[idx, sel_idx_array]
        sel_norms_sq = norms_sq[sel_idx_array]
        
        # dist_sq = norms_sq[idx] + sel_norms_sq - 2 * sel_dots
        dist_sq = norms_sq[idx] + sel_norms_sq - 2 * sel_dots
        
        # All distances must be >= max_norm_sq
        if np.all(dist_sq >= max_norm_sq):
            selected_mask[idx] = True
            selected_indices.append(idx)
    
    result = candidates[selected_indices]
    return result
# EVOLVE-BLOCK-END
