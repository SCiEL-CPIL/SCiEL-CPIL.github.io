import numpy as np
from itertools import combinations, product
import time

D = 11

def _generate_weight_k_vectors(k):
    """Generate all integer vectors in R^D with exactly k non-zero entries, each +/- 1."""
    vectors = []
    for pos in combinations(range(D), k):
        for signs in range(2**k):
            v = np.zeros(D, dtype=np.int64)
            for i, p in enumerate(pos):
                v[p] = 1 if (signs >> i) & 1 else -1
            vectors.append(v)
    return np.array(vectors, dtype=np.int64)

def _greedy_pack(vectors, max_dot, n_trials=20):
    """
    Greedily select a subset of vectors such that pairwise dot product <= max_dot.
    """
    if len(vectors) == 0:
        return np.array([], dtype=np.int64).reshape(0, D)
    
    N = len(vectors)
    # Compute all pairwise dot products
    dots = vectors @ vectors.T
    
    best_set = []
    best_indices = []
    rng = np.random.default_rng(12345)
    
    for trial in range(n_trials):
        perm = rng.permutation(N)
        current_indices = []
        
        for idx in perm:
            v = vectors[idx]
            compatible = True
            # Check against current set
            if len(current_indices) > 0:
                # Vectorized check: dots[idx, current_indices] <= max_dot
                if np.any(dots[idx, current_indices] > max_dot):
                    compatible = False
            if compatible:
                current_indices.append(idx)
                
        if len(current_indices) > len(best_indices):
            best_indices = current_indices
            
    return vectors[best_indices].copy()

def _refine_greedy(vectors, max_dot, n_restarts=5):
    """
    A slightly smarter greedy that tries to add vectors back in a second pass,
    or uses a priority queue based on 'slack' or degree.
    For now, let's stick to robust random shuffles but more of them.
    """
    if len(vectors) == 0:
        return np.array([], dtype=np.int64).reshape(0, D)
    
    N = len(vectors)
    dots = vectors @ vectors.T
    
    best_indices = []
    rng = np.random.default_rng(999)
    
    # Strategy 1: Random shuffles
    for _ in range(50):
        perm = rng.permutation(N)
        current = []
        for idx in perm:
            if len(current) > 0 and np.any(dots[idx, current] > max_dot):
                continue
            current.append(idx)
        if len(current) > len(best_indices):
            best_indices = current
            
    # Strategy 2: Greedy by "in-degree" (fewest conflicts)
    # This is harder to implement efficiently without precomputation, 
    # but we can approximate by picking vectors that have lower average dot product?
    # Let's try sorting by norm (all same) or just random.
    
    return vectors[best_indices].copy()

def construct():
    start_time = time.time()
    timeout = 115  # Leave 5s margin
    
    # 1. Weight 2 vectors
    # All 220 weight-2 vectors are mutually compatible with dot <= 1?
    # u = e_i +/- e_j, v = e_k +/- e_l
    # If disjoint: dot = 0
    # If share 1 index: dot = +/-1
    # If same indices: dot = +/-2 (but u != v, so if u=e1+e2, v=e1+e2 is not allowed as distinct? 
    # Wait, "nonzero integer vectors". Usually distinct vectors.
    # If u = e1+e2 and v = e1+e2, they are the same vector. The problem implies a set of vectors.
    # So we consider distinct vectors.
    # For distinct weight 2 vectors:
    # Case 1: Disjoint supports. Dot = 0.
    # Case 2: Share 1 index. e.g., e1+e2 and e1+e3. Dot = 1.
    # Case 3: Share 1 index, opposite sign. e1+e2 and -e1+e3. Dot = -1.
    # Case 4: Same support, different signs. e1+e2 and e1-e2. Dot = 0.
    # Case 5: Same support, opposite signs. e1+e2 and -e1-e2. Dot = -2.
    # Max dot is 1.
    # So ALL weight 2 vectors can be included.
    v2 = _generate_weight_k_vectors(2)
    best_vectors = v2
    best_k = len(v2)
    
    if time.time() - start_time > timeout:
        return best_vectors

    # 2. Weight 3 vectors
    # Norm^2 = 3.
    # Condition: dist^2 >= 3 => 6 - 2 dot >= 3 => 2 dot <= 3 => dot <= 1.5 => dot <= 1.
    # So we need a subset of weight 3 vectors with pairwise dot <= 1.
    # Total weight 3 vectors: C(11,3) * 2^3 = 165 * 8 = 1320.
    v3 = _generate_weight_k_vectors(3)
    # Greedy pack
    packed_v3 = _refine_greedy(v3, max_dot=1, n_restarts=10)
    if len(packed_v3) > best_k:
        best_vectors = packed_v3
        best_k = len(packed_v3)
        
    if time.time() - start_time > timeout:
        return best_vectors

    # 3. Weight 4 vectors
    # Norm^2 = 4.
    # Condition: dist^2 >= 4 => 8 - 2 dot >= 4 => 2 dot <= 4 => dot <= 2.
    # Total weight 4 vectors: C(11,4) * 2^4 = 330 * 16 = 5280.
    v4 = _generate_weight_k_vectors(4)
    # This is the bottleneck. 5280 vectors.
    # Dots matrix is 5280x5280 ~ 27M entries. Fits in memory (216MB).
    # Greedy is O(N^2) per trial. 50 trials * 27M ops ~ 1.3B ops. Might be slow in Python/NumPy if not careful.
    # But _refine_greedy uses vectorized checks inside the loop, which is O(k) per candidate.
    # Average k might be large.
    # Let's try a few trials.
    packed_v4 = _refine_greedy(v4, max_dot=2, n_restarts=10)
    if len(packed_v4) > best_k:
        best_vectors = packed_v4
        best_k = len(packed_v4)
        
    if time.time() - start_time > timeout:
        return best_vectors

    # 4. Weight 5 vectors
    # Norm^2 = 5.
    # Condition: dist^2 >= 5 => 10 - 2 dot >= 5 => 2 dot <= 5 => dot <= 2.5 => dot <= 2.
    # Total weight 5 vectors: C(11,5) * 2^5 = 462 * 32 = 14784.
    # This is getting large. 14784^2 ~ 218M entries. 1.7GB memory. Might be too much or slow.
    # Let's skip weight 5 if time is tight or memory is a concern.
    # However, the kissing number in 11D is known to be 506 (for the Leech lattice construction? No, Leech is 24D).
    # The best known kissing number for 11D is 506.
    # Weight 2 gives 220.
    # Weight 3 might give more?
    # Weight 4 might give more?
    
    # Let's check if we can improve on weight 4 with more trials.
    if time.time() - start_time < timeout - 10:
        # Try more trials for weight 4
        packed_v4_more = _refine_greedy(v4, max_dot=2, n_restarts=30)
        if len(packed_v4_more) > best_k:
            best_vectors = packed_v4_more
            best_k = len(packed_v4_more)

    return best_vectors
