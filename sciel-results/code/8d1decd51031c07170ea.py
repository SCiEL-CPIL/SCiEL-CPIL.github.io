"""Improved construction for kissing number in dimension 11 using weight-2 vectors."""
import time
import numpy as np
from itertools import combinations

D = 11

# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    deadline = start + 110  # safety margin from 150s limit

    # D11 lattice: all integer vectors with exactly two non-zero entries,
    # each being +1 or -1, at any two distinct coordinates.
    # Total: C(11, 2) * 2^2 = 55 * 4 = 220 vectors.
    
    # Each vector has norm squared = 2, so norm = sqrt(2).
    # Pairwise dot products are in {-2, -1, 0, 1, 2}.
    # We need min pairwise distance >= max norm.
    # Distance squared between v_i and v_j = 2 + 2 - 2*(v_i . v_j) = 4 - 2*(v_i . v_j).
    # Max norm squared = 2.
    # We need 4 - 2*(v_i . v_j) >= 2, i.e., v_i . v_j <= 1.
    
    # For D11 vectors (weight 2), the dot product v_i . v_j can be:
    # - 2 if same vector (diagonal, exclude)
    # - 1 if they share one coordinate with same sign
    # - 0 if they share one coordinate with opposite signs, or share no coordinates
    # - -1 if they share one coordinate with opposite signs... wait
    # Let me think more carefully.
    
    # v_i has non-zero at positions p, q with signs s_p, s_q.
    # v_j has non-zero at positions r, t with signs s_r, s_t.
    # Dot product = sum over common positions of (sign_i * sign_j).
    
    # If {p,q} = {r,t}: dot = s_p*s_r + s_q*s_t (which is 2 if same vector, or could be -2 if negated)
    # Actually if p=r and q=t: dot = s_p*s_r + s_q*s_t
    #   - If same signs: 1 + 1 = 2 (same vector)
    #   - If one pair same, one different: 1 - 1 = 0
    #   - If both different: -1 - 1 = -2 (negated vector)
    # If p=r, q!=r, q!=t (one shared): dot = s_p*s_r
    # If no shared: dot = 0
    
    # So dot products are in {-2, -1, 0, 1, 2}.
    # We need dot <= 1 for all distinct pairs.
    
    # Problem: if we include both v and -v, their dot is -2 which is fine.
    # But wait: v . (-v) = -||v||^2 = -2. That's <= 1. OK.
    # However, v . v = 2 (diagonal, we exclude self).
    
    # The issue: two different weight-2 vectors can have dot = 1 (share one coordinate, same sign).
    # That's allowed since 1 <= 1.
    
    # Can two weight-2 vectors have dot = 2? Only if they are the same vector.
    # Can they have dot > 1 without being identical? No, max is 2 for identical.
    
    # So all 220 weight-2 vectors are mutually compatible (dot <= 1 for distinct pairs).
    
    vectors = []
    for i in range(D):
        for j in range(i + 1, D):
            for si in [1, -1]:
                for sj in [1, -1]:
                    v = [0] * D
                    v[i] = si
                    v[j] = sj
                    vectors.append(v)
    
    result = np.array(vectors, dtype=np.int64)
    
    if time.time() > deadline:
        return result
    
    return result
# EVOLVE-BLOCK-END
