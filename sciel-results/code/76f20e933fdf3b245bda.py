"""Constructing a kissing configuration in 11 dimensions using a binary-to-integer mapping."""

D = 11

# EVOLVE-BLOCK-START
import itertools
import numpy as np
from scipy.special import comb


def construct():
    # We construct a set of integer vectors in R^11 with norm squared 4 (norm 2).
    # The condition for a valid kissing configuration with max norm R is that
    # the minimum pairwise distance is at least R.
    # For vectors u, v with norm squared 4, distance squared is 8 - 2(u.v).
    # We need distance >= 2, so distance squared >= 4.
    # 8 - 2(u.v) >= 4 => 2(u.v) <= 4 => u.v <= 2.
    #
    # We consider two types of integer vectors with norm squared 4:
    # Type A: One entry is +/- 2, others 0.
    # Type B: Four entries are +/- 1, others 0.
    #
    # We verify that for any distinct u, v in the union of these sets, u.v <= 2.
    # 1. u, v in A:
    #    If they have different supports, u.v = 0.
    #    If same support, one is 2e_i and other is -2e_i (distinct), u.v = -4.
    #    Max u.v = 0 <= 2.
    # 2. u, v in B:
    #    They have 4 non-zero entries.
    #    Max overlap for distinct vectors is 3 positions with same sign, 1 with opposite.
    #    u.v = 3(1) + 1(-1) = 2.
    #    Max u.v = 2 <= 2.
    # 3. u in A, v in B:
    #    u = 2 e_i. v has v_i in {0, 1, -1}.
    #    u.v = 2 v_i. Max is 2 (if v_i = 1).
    #    Max u.v = 2 <= 2.
    #
    # Thus, the union is valid.
    
    vectors = []
    
    # Type A: +/- 2 in one position
    for i in range(D):
        for sign in (1, -1):
            v = [0] * D
            v[i] = 2 * sign
            vectors.append(v)
            
    # Type B: Four +/- 1s
    # Choose 4 positions out of 11
    positions = itertools.combinations(range(D), 4)
    # Assign signs to these 4 positions
    signs = itertools.product((1, -1), repeat=4)
    
    for pos in positions:
        for s in signs:
            v = [0] * D
            for k, p in enumerate(pos):
                v[p] = s[k]
            vectors.append(v)
            
    return np.array(vectors, dtype=np.int64)
# EVOLVE-BLOCK-END
