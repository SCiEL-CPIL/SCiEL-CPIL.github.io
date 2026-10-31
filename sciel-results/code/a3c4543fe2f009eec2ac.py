"""Cap set greedy builder with algebraic-independence-aware priority."""
import time
import math
import numpy as np

# EVOLVE-BLOCK-START
_start_time = time.time()
_DEADLINE = 315  # seconds, safety margin before 330s kill

# Global state
_accepted_vectors = set()
_accepted_list = []
_complement_map = {}  # vector -> (-vector) mod 3

# For the algebraic independence heuristic, we maintain a set of "basis" vectors
# derived from accepted vectors. We use a lightweight proxy: track the span of
# coordinate-wise products. Since full linear algebra over F3 with many monomials
# is expensive, we use a hash-based independence test on a subset of monomials.

# We'll track a set of "signature" hashes for low-degree monomial evaluations
# to approximate algebraic independence.
_monomial_hashes = set()

def _compute_monomial_signature(v, n):
    """Compute a signature based on low-degree monomials evaluated at v.
    Monomials: x_i (degree 1), x_i * x_j (degree 2) for i < j.
    We map these to a hash to test for linear independence over the integers
    (as a proxy for F3 independence).
    
    Actually, for a better heuristic: we want vectors that are "far" from the
    span of previously accepted vectors in the monomial feature space.
    We approximate this by checking if the monomial evaluation vector is
    linearly independent from previous ones. For speed, we use a randomized
    projection (Johnson-Lindenstrauss style) into a fixed dimension.
    """
    # Degree-1 features: x_i for i in range(n)
    # Degree-2 features: x_i * x_j for i < j
    # Total features: n + n*(n-1)/2
    
    # For very large n, this is expensive. We'll use a subsampled approach.
    # Use a fixed random projection to reduce dimensionality.
    # But we need deterministic behavior, so use a hash-based projection.
    
    # Simpler approach: just use the coordinate values and pairwise products
    # as a tuple, and check if it's "new" compared to previous signatures.
    # This is not quite linear independence, but it captures algebraic structure.
    
    # Even simpler and faster: use the sum of squares and sum of cubes as
    # invariants that change with algebraic structure.
    
    # Let's use a different approach: track the "algebraic norm" which is
    # the sum of x_i^2 and sum of x_i*x_j. Vectors with distinct norms
    # are more likely to be algebraically independent.
    
    sum_sq = sum(x * x for x in v)
    sum_prod = 0
    for i in range(n):
        for j in range(i + 1, n):
            sum_prod += v[i] * v[j]
    
    # Create a signature from these invariants
    return (sum_sq, sum_prod % (2**32))

def _count_conflicts(v):
    """Count how many existing vectors a have (-a - v) mod 3 also in the set."""
    n = len(v)
    count = 0
    for a in _accepted_list:
        comp_a = _complement_map[a]
        c = tuple((comp_a[i] - v[i]) % 3 for i in range(n))
        if c in _accepted_vectors:
            count += 1
    return count

def _count_near_conflicts(v, max_check=30):
    """Count vectors a where (-a-v) mod 3 is at Hamming distance <=1 from an accepted vector."""
    n = len(v)
    count = 0
    limit = min(len(_accepted_list), max_check)
    for a in _accepted_list[:limit]:
        comp_a = _complement_map[a]
        c = tuple((comp_a[i] - v[i]) % 3 for i in range(n))
        # Check if c is at Hamming distance 1 from any accepted vector
        for coord_idx in range(n):
            for delta in [1, 2]:
                c_alt = list(c)
                c_alt[coord_idx] = (c_alt[coord_idx] + delta) % 3
                if tuple(c_alt) in _accepted_vectors:
                    count += 1
                    break
    return count

def priority(vector, n):
    """Priority for greedy cap set construction with algebraic independence heuristic."""
    # Deadline guard
    if time.time() - _start_time > _DEADLINE:
        return -1e18
    
    v = tuple(vector)
    n = len(v)
    
    # Core: conflict count
    if _accepted_list:
        conflicts = _count_conflicts(v)
    else:
        conflicts = 0
    
    # Coordinate distribution analysis
    counts = [0, 0, 0]
    for x in v:
        counts[x] += 1
    
    entropy = 0.0
    for c in counts:
        if c > 0:
            p = c / n
            entropy -= p * math.log(p + 1e-15)
    
    distinct = sum(1 for c in counts if c > 0)
    balance_score = entropy + 0.3 * distinct
    
    # Center deviation: vectors farther from all-1s are safer
    center_deviation = sum(abs(x - 1) for x in v) / n
    geo_score = center_deviation
    
    # Hamming weight
    hw = sum(1 for x in v if x != 0)
    hw_center = n / 3.0
    hw_penalty = abs(hw - hw_center) / n
    
    # Algebraic independence heuristic:
    # Compute monomial signature and check novelty
    alg_score = 0.0
    if _accepted_list:
        sig = _compute_monomial_signature(v, n)
        # If this signature is new, it's algebraically "independent"
        if sig not in _monomial_hashes:
            alg_score = 0.5  # Bonus for new algebraic structure
        else:
            alg_score = -0.2  # Penalty for redundant algebraic structure
    
    # Near-conflict penalty (smooth surrogate for hard conflicts)
    smooth_penalty = 0.0
    if _accepted_list and conflicts == 0 and len(_accepted_vectors) <= 300:
        near = _count_near_conflicts(v, max_check=20)
        smooth_penalty = 0.15 * near
    
    # Deterministic tie-breaker
    h = 0
    for i, x in enumerate(v):
        h = (h * 31 + x * (i + 1)) & 0xFFFFFFFF
    tie_break = (h / 0xFFFFFFFF) * 0.003
    
    # Combine scores
    conflict_penalty = 2.0 * conflicts
    
    total = (balance_score 
             + 0.4 * geo_score 
             + alg_score 
             - hw_penalty 
             - conflict_penalty 
             - smooth_penalty 
             + tie_break)
    
    return total

def _accept_vector(v):
    """Record that vector v has been accepted into the cap set."""
    _accepted_vectors.add(v)
    _accepted_list.append(v)
    _complement_map[v] = tuple((-x) % 3 for x in v)
    # Update monomial signature set
    sig = _compute_monomial_signature(v, len(v))
    _monomial_hashes.add(sig)

# EVOLVE-BLOCK-END
