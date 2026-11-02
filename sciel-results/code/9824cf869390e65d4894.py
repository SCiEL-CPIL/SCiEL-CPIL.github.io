"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations
from collections import defaultdict


def construct():
    """
    Construct nonzero integer vectors in R^11 whose minimum pairwise distance
    is at least the maximum vector norm.
    
    Strategy: Use vectors of norm 2 (i.e., with two non-zero entries of magnitude 1,
    or one entry of magnitude sqrt(2) but we need integers, so two entries of 1).
    Actually for integer vectors, norm 2 means vectors like e_i + e_j or e_i - e_j.
    
    The key insight: For vectors of norm 2, the distance between two such vectors
    v and w is ||v - w||. We need ||v - w|| >= max_norm = 2 for all pairs.
    So ||v - w||^2 >= 4.
    
    ||v - w||^2 = ||v||^2 + ||w||^2 - 2 v·w = 4 + 4 - 2 v·w = 8 - 2 v·w
    We need 8 - 2 v·w >= 4, i.e., v·w <= 2.
    
    For norm-2 integer vectors (two ±1 entries), the dot product v·w can be:
    - 2: if v and w share both positions with same signs (e.g., e1+e2 and e1+e2) - but these are the same vector
    - 1: if they share one position with same sign, or two positions but one sign differs... let me think more carefully
    - 0: if they share no positions, or share positions with opposite signs that cancel
    - -1: similar
    - -2: if they share both positions with opposite signs
    
    Actually, let's enumerate: v has nonzeros at positions i,j with values si, sj.
    w has nonzeros at positions k,l with values sk, sl.
    
    v·w = sum over positions of v[pos]*w[pos].
    
    If {i,j} and {k,l} are disjoint: v·w = 0
    If they share exactly one position, say i=k: v·w = si*sk (which is ±1)
    If {i,j} = {k,l}: v·w = si*sk + sj*sl where k=i, l=j (or k=j, l=i)
      If k=i, l=j: v·w = si*si + sj*sj = 1 + 1 = 2 (same vector)
      If k=j, l=i: v·w = si*sj + sj*si = 2*si*sj = ±2
    
    So for distinct vectors, the dot product can be:
    - 0 (disjoint supports)
    - ±1 (share one coordinate)
    - ±2 (same support, but different sign patterns)
    
    We need v·w <= 2 for all pairs. Since the max possible is 2 (for identical vectors),
    and for distinct vectors with same support, v·w = ±2.
    
    If v·w = 2, that means v = w (same vector), which we don't have duplicates.
    If v·w = -2, that's fine since -2 <= 2.
    
    Wait, I need to re-examine. The condition is min pairwise distance >= max norm.
    If all vectors have norm 2, then max norm = 2.
    Distance = ||v - w||. We need ||v - w|| >= 2 for all v != w.
    ||v - w||^2 = 8 - 2(v·w) >= 4 => v·w <= 2.
    
    For distinct vectors, v·w can be at most... let's see:
    - Same support, e.g., v = e1+e2, w = e1+e2: same vector, not distinct
    - v = e1+e2, w = e1-e2: v·w = 1 - 1 = 0
    - v = e1+e2, w = e1+e3: v·w = 1
    - v = e1+e2, w = -e1-e2: v·w = -2
    
    So for distinct vectors, v·w <= 1 always? Let me check:
    Can two distinct norm-2 integer vectors have dot product 2?
    v·w = 2 means they agree on both nonzero positions with same signs. That means v = w.
    So for distinct vectors, v·w <= 1 < 2. 
    
    Therefore, ALL distinct norm-2 integer vectors satisfy the condition!
    
    How many are there?
    Choose 2 positions out of 11: C(11,2) = 55
    For each pair, 4 sign combinations: (1,1), (1,-1), (-1,1), (-1,-1)
    Total: 55 * 4 = 220 vectors.
    
    But wait, we could also use vectors of different norms. The max norm would then be larger.
    Let's consider using vectors of norm 1 (unit basis vectors e_i).
    There are 11 such vectors. Their mutual distances: ||e_i - e_j|| = sqrt(2) for i != j.
    Max norm = 1. We need distance >= 1, and sqrt(2) >= 1, so this works.
    
    Can we mix norm-1 and norm-2 vectors?
    If we include norm-2 vectors, max norm = 2.
    For a norm-1 vector e_i and a norm-2 vector v with nonzeros at j,k:
    Distance squared = ||e_i - v||^2 = 1 + 4 - 2(e_i·v) = 5 - 2*(v_i)
    If i is not in {j,k}: v_i = 0, distance^2 = 5, distance = sqrt(5) >= 2. OK.
    If i is in {j,k}, say i=j: v_i = ±1, distance^2 = 5 - 2*(±1) = 3 or 7.
    distance = sqrt(3) ≈ 1.73 < 2. NOT OK.
    
    So we can't mix norm-1 and norm-2 vectors if the norm-1 vector shares a coordinate with the norm-2 vector.
    
    What about only norm-2 vectors? We get 220.
    
    Can we do better with higher norm vectors?
    Consider vectors of norm 3: e.g., e1+e2+e3.
    Number of such vectors: C(11,3) * 8 = 165 * 8 = 1320.
    But we need to check pairwise distances.
    
    Actually, the best approach might be to use a greedy algorithm to build up a set of vectors.
    
    Let me try a greedy approach: generate candidate vectors and greedily add them if they satisfy the constraint.
    """
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill
    
    # Generate all norm-2 vectors (two ±1 entries)
    candidates = []
    for i, j in combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i], v[j] = si, sj
                candidates.append(tuple(v))
    
    # Also try norm-1 vectors (single ±1 entry)
    for i in range(D):
        for s in (1, -1):
            v = [0] * D
            v[i] = s
            candidates.append(tuple(v))
    
    # Greedy selection: sort by some heuristic and add if valid
    # For norm-2 vectors, all pairs are valid as shown above
    # For norm-1 vectors, they conflict with norm-2 vectors sharing a coordinate
    
    # Let's just use all norm-2 vectors
    result = []
    max_norm_sq = 0
    
    for v_tuple in candidates:
        v = np.array(v_tuple, dtype=np.int64)
        norm_sq = np.dot(v, v)
        
        # Check against all existing vectors
        valid = True
        for u in result:
            dist_sq = np.dot(v - u, v - u)
            # We need dist >= max(norm_v, norm_u, current_max_norm)
            # Actually the condition is: min pairwise distance >= max norm of all vectors
            # This is a global constraint, so we need to track the max norm
            pass
        
        # Simpler: just collect all norm-2 vectors since they all work together
        if norm_sq == 2:
            result.append(v)
    
    if len(result) > 0:
        return np.array(result, dtype=np.int64)
    
    # Fallback: return empty or basic set
    return np.array([], dtype=np.int64).reshape(0, D)

# EVOLVE-BLOCK-END
