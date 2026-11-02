"""Program for kissing number in dimension 11 using greedy integer vector search."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools

def construct():
    # Approach: Use all integer vectors with squared norm 2.
    # These vectors have exactly two non-zero entries, each being +/-1.
    # For any two distinct such vectors u, v:
    # ||u||^2 = 2, ||v||^2 = 2.
    # ||u-v||^2 = 2 + 2 - 2 u.dot(v) = 4 - 2 u.dot(v).
    # We require ||u-v|| >= max(||u||, ||v||) = sqrt(2).
    # ||u-v||^2 >= 2.
    # 4 - 2 u.dot(v) >= 2 => 2 u.dot(v) <= 2 => u.dot(v) <= 1.
    # 
    # Max dot product for distinct vectors of this type:
    # If they share 1 index, dot is +/-1.
    # If they share 2 indices, they are either identical (dot 2, excluded) or differ in signs.
    # If they differ in signs at both shared indices, dot is -2 or 0.
    # Specifically, if supports are same:
    # u = (1, 1), v = (-1, 1) -> dot 0.
    # u = (1, 1), v = (1, -1) -> dot 0.
    # u = (1, 1), v = (-1, -1) -> dot -2.
    # So max dot for distinct vectors is 1.
    # Thus, all 220 vectors are mutually compatible.
    
    vectors = []
    for i in range(D):
        for j in range(i + 1, D):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i] = si
                    v[j] = sj
                    vectors.append(v)
    
    return np.array(vectors, dtype=np.int64)
# EVOLVE-BLOCK-END
