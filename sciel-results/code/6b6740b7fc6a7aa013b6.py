"""Construct a kissing configuration in dimension 11 using a known lattice structure."""
import numpy as np
import time

D = 11

# EVOLVE-BLOCK-START
import itertools
import numpy as np

def construct():
    """
    Construct nonzero integer vectors in R^11 whose minimum pairwise distance
    is at least the maximum vector norm.
    
    We use the D11 lattice roots: all vectors of the form (+/-1, ..., +/-1, 0, ...)
    or (0, ..., +/-1, ..., +/-1, 0, ...) with exactly two non-zero entries of magnitude 1.
    
    The D11 lattice roots are vectors with two coordinates being +1/-1 and the rest 0.
    There are C(11,2) * 4 = 55 * 4 = 220 such vectors.
    
    For any two distinct such vectors u and v:
    - ||u||^2 = 2, ||v||^2 = 2
    - (u-v)^2 = ||u||^2 + ||v||^2 - 2*u.v = 4 - 2*u.v
    
    We need ||u-v|| >= max(||u||, ||v||) = sqrt(2)
    i.e., (u-v)^2 >= 2
    i.e., 4 - 2*u.v >= 2
    i.e., u.v <= 1
    
    The dot product u.v can be:
    - 0: if supports are disjoint or orthogonal
    - 2: if u and v have the same support and same signs (but they'd be the same vector, excluded)
    - 0: if same support but differ in one sign position
    - -2: if same support but differ in both sign positions
    - 1: if supports overlap in exactly one coordinate with matching sign
    - -1: if supports overlap in exactly one coordinate with opposite sign
    - 0: if supports overlap in exactly one coordinate... wait, let me reconsider.
    
    Actually for two vectors with two non-zero entries each:
    If they share 0 coordinates: dot = 0
    If they share 1 coordinate: dot = s1*s2 where s1, s2 are +/-1, so dot = +/-1
    If they share 2 coordinates: dot = s1*s1' + s2*s2' where signs can vary, so dot in {-2, 0, 2}
    
    The maximum dot product between distinct vectors is 1 (when they share one coordinate with matching sign).
    So (u-v)^2 >= 4 - 2*1 = 2. This satisfies the constraint.
    
    Therefore, all 220 D11 lattice roots form a valid kissing configuration with k=220.
    """
    vectors = []
    
    # Generate all vectors with exactly two non-zero entries, each being +1 or -1
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i] = si
                v[j] = sj
                vectors.append(v)
    
    return np.array(vectors, dtype=np.int64)
# EVOLVE-BLOCK-END
