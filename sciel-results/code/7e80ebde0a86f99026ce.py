import time
import math

_START = time.time()
_DEADLINE = _START + 280.0

_cache = {}

def _algebraic_weight(vector, n):
    """
    Compute the algebraic weight of a vector in the group algebra of Z_3^n.
    
    The weight is the sum of squared magnitudes of the DFT coefficients at non-zero frequencies.
    For a single vector v, its indicator function is 1 at v and 0 elsewhere.
    The DFT at frequency chi is sum_x f(x) * chi(x-v) where chi is a character.
    
    For Z_3^n, characters are chi_a(x) = omega^{a·x} where omega = e^{2*pi*i/3}.
    
    The DFT coefficient at frequency a is:
    F(a) = omega^{a·v}
    
    So |F(a)|^2 = 1 for all a.
    
    But we want to measure how "spread out" the vector is in frequency space.
    The sum of |F(a)|^2 over all a is 3^n (Parseval).
    
    A better measure: consider the vector's contribution to additive energy.
    A vector v forms an AP with u and w if v+u+w = 0 (mod 3) with distinct points.
    
    The key insight: vectors that are "central" (balanced across coordinates) 
    tend to form fewer APs because they don't align with low-frequency structures.
    
    We compute a smooth surrogate: the L2 norm of the vector's "frequency profile".
    For a single vector, this is related to how many characters it activates equally.
    
    A practical measure: compute the sum over all non-zero frequencies a of |F(a)|^2,
    but weight by the "spread" of the vector's coordinates.
    
    Actually, for a single vector, the DFT is just a phase, so all magnitudes are 1.
    The distinguishing feature must come from the vector's position in the group.
    
    Better approach: use the vector's coordinates to compute a measure of how
    "aligned" it is with low-dimensional subspaces (which form APs).
    
    For each coordinate, the value 0, 1, or 2 determines the character evaluation.
    A vector with all coordinates equal (like (1,1,...,1)) lies in a 1D subspace
    and is prone to forming APs.
    
    We compute: sum over coordinates of (value - n/3)^2, which measures deviation
    from the "central" balanced state.
    """
    # Measure how far the vector is from being "balanced" in each coordinate
    # For a single vector, this is just the sum of squares of coordinate values
    # after centering.
    
    # The algebraic weight should be high for vectors that are "generic" 
    # (not lying in low-dimensional subspaces).
    
    # Compute the sum of squared coordinate values
    sum_sq = sum(x * x for x in vector)
    
    # The ideal balanced vector has each coordinate = 1 (since 1 is the "center" of {0,1,2}
    # in the sense that it's equidistant from 0 and 2)
    # Actually, in Z_3, 0, 1, 2 are all symmetric. The "center" is not well-defined.
    
    # Better: compute the variance of the coordinates
    # A vector with all coordinates the same has variance 0 and is prone to APs.
    # A vector with mixed coordinates has higher variance and is more "generic".
    
    mean = sum(vector) / n
    variance = sum((x - mean) ** 2 for x in vector)
    
    # Also compute the L2 norm as a secondary measure
    l2_norm = math.sqrt(sum(x * x for x in vector))
    
    # The algebraic weight: higher variance is better (more generic)
    # Lower L2 norm is better (closer to origin, which is symmetric)
    
    return variance * 100.0 - l2_norm * 10.0

def _precompute(n):
    if time.time() - _START > _DEADLINE - 10.0:
        return None  # Timeout risk

    # For each vector, compute its algebraic weight
    score_map = {}
    for v in _generate_vectors(n):
        score_map[v] = _algebraic_weight(v, n)

    return score_map

def _generate_vectors(n):
    """Generate all vectors in {0,1,2}^n."""
    from itertools import product
    return [tuple(v) for v in product([0, 1, 2], repeat=n)]

def priority(vector, n):
    """
    Return a float priority for a vector in {0,1,2}^n.
    """
    # Deadline check
    if time.time() - _START > _DEADLINE:
        return 0.0

    if n not in _cache:
        # Pre-compute if not already done
        # Check time before starting heavy pre-computation
        if time.time() - _START > _DEADLINE - 60.0:
            # Not enough time for pre-computation, use fallback
            _cache[n] = None  # Mark as fallback
        else:
            _cache[n] = _precompute(n)

    if _cache[n] is None:
        # Fallback to simple heuristic
        return _algebraic_weight(vector, n)

    return _cache[n][vector]
