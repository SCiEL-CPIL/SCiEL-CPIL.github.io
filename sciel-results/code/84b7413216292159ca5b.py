"""Program for kissing number in dimension 11 using greedy augmentation of D_11 lattice."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools


def construct():
    # Start with D_11 minimal vectors (norm sqrt(2))
    base_vectors = []
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i] = si
                v[j] = sj
                base_vectors.append(v)
    
    vecs = np.array(base_vectors, dtype=np.int64)
    max_norm_sq = 2  # norm squared of base vectors
    
    # Try to add vectors with norm squared 3 (three 1s/-1s)
    # But need to check distance constraint
    # For now, return the base set which is proven valid
    return vecs
# EVOLVE-BLOCK-END
