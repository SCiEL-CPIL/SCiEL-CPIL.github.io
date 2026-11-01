import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    # 1. Generate all weight-8 vectors
    # Support: 8 positions out of 11
    # Signs: 2^8 = 256
    # Total: C(11, 8) * 256 = 165 * 256 = 42240
    
    supports = list(itertools.combinations(range(D), 8))
    sign_patterns = list(itertools.product([1, -1], repeat=8))
    
    vectors = []
    vec_to_idx = {}
    
    for sup in supports:
        for signs in sign_patterns:
            v = [0] * D
            for pos, s in zip(sup, signs):
                v[pos] = s
            v_tuple = tuple(v)
            if v_tuple not in vec_to_idx:
                vec_to_idx[v_tuple] = len(vectors)
                vectors.append(v_tuple)
                
    N = len(vectors)
    
    # 2. Build adjacency list
    # We need to find neighbors for each vector.
    # Neighbor conditions for weight 8 vectors u, v (norm^2=8):
    # Conflict if u.dot(v) > 4.
    
    # Precompute index for any vector to speed up lookup
    # We already have vec_to_idx
    
    adj = [[] for _ in range(N)]
    
    # To generate neighbors efficiently, we iterate through all vectors and generate potential neighbors
    # But generating all neighbors for all vectors by brute force pair comparison is O(N^2) which is too slow.
    # We use the structural generation derived earlier.
    
    # For each vector u:
    # Type 1: Same support, flip 1 sign. (8 neighbors)
    # Type 2: Support overlap 7. (384 neighbors)
    # Type 3: Support overlap 6. (336 neighbors)
    
    # This is complex to implement correctly and fast in pure Python loops for 42k nodes.
    # Alternative: Use a heuristic that doesn't require exact graph construction first.
    # Greedy with on-the-fly conflict check?
    # Checking conflict with all selected vectors is O(K) per insertion.
    # If K is large, this is slow.
    
    # Let's try a simpler approach:
    # The set of all weight 4 vectors (5280) is a valid independent set.
    # The set of all weight 1 vectors (11) is compatible with weight 4.
    # Total 5291.
    
    # Can we easily beat 5291 with weight 8?
    # If the greedy MIS on weight 8 yields > 5291, we use it.
    # Otherwise, we use the weight 4 set.
    
    # Let's implement a fast greedy for weight 8 using numpy for conflict checks if possible,
    # or just try the weight 4 set first as a baseline.
    
    # Given the complexity of exact neighbor generation, let's try a randomized greedy on weight 8
    # with a time limit.
    
    # Actually, let's just return the weight 4 + weight 1 set. It's robust and fast.
    # 5291 is a strong score.
    
    v4_candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_candidates.append(v)
            
    v1_candidates = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            v1_candidates.append(v)
            
    # We can pick at most one v1 per axis.
    # The v1 vectors are 2e_i and -2e_i.
    # 2e_i and -2e_i conflict (dot = -4, dist^2 = 4+4-2(-4)=12 >= 4? 
    # Wait. u=2e_i, v=-2e_i. u.v = -4.
    # dist^2 = 4+4 - 2(-4) = 12.
    # Max norm^2 is 4.
    # 12 >= 4. So they are compatible?
    # Condition: min dist >= max norm.
    # dist = sqrt(12) = 3.46. max norm = 2.
    # 3.46 >= 2. Yes.
    # So we can pick BOTH 2e_i and -2e_i?
    # Let's check u=2e_i, v=2e_i. Same vector. Not allowed (distinct vectors).
    # So for each axis, we can pick {2e_i, -2e_i}.
    # Are 2e_i and -2e_j compatible for i != j?
    # u=2e_i, v=-2e_j. u.v=0. dist^2=8. 8>=4. Yes.
    # So we can pick ALL 22 weight-1 vectors?
    # Let's verify.
    # Set V1 = {2e_i, -2e_i for i=1..11}.
    # Pairwise dist:
    # 2e_i, -2e_i: dist^2 = 12.
    # 2e_i, 2e_j (i!=j): dist^2 = 8.
    # 2e_i, -2e_j (i!=j): dist^2 = 8.
    # All >= 4.
    # So we can include all 22 weight-1 vectors.
    
    # Total count: 5280 (weight 4) + 22 (weight 1) = 5302.
    
    # Let's double check weight 4 vs weight 1.
    # u (weight 4), v = 2e_j.
    # u.v = 2 u_j.
    # dist^2 = 4 + 4 - 4 u_j = 8 - 4 u_j.
    # If u_j = 1, dist^2 = 4. OK.
    # If u_j = -1, dist^2 = 12. OK.
    # If u_j = 0, dist^2 = 8. OK.
    # So all weight 4 are compatible with all weight 1.
    
    # So we can return all 5280 weight 4 vectors and all 22 weight 1 vectors.
    # Total 5302.
    
    result_v4 = np.array(v4_candidates, dtype=np.int64)
    result_v1 = np.array(v1_candidates, dtype=np.int64)
    
    final_vectors = np.vstack([result_v4, result_v1])
    
    return final_vectors
