"""Cap set greedy builder with geometric-aware priority."""
import time
import itertools
import numpy as np

# EVOLVE-BLOCK-START
import math
_start_time = time.time()
_DEADLINE = 325  # seconds, safety margin before 330s kill

# Global state for tracking accepted vectors and computing conflict scores
_accepted_vectors = set()
_accepted_list = []
# Precomputed: for each accepted vector a, store the "complement" (-a) mod 3
# to speed up conflict checking
_complement_map = {}  # maps vector -> its complement (-v) mod 3

def _count_conflicts(v):
    """Count how many existing vectors a have (-a - v) mod 3 also in the set.
    This measures how many 'witness' lines the vector would participate in.
    Uses precomputed complements for speed.
    """
    n = len(v)
    count = 0
    for a in _accepted_list:
        # c = (-a - v) mod 3 = complement(a) - v mod 3
        comp_a = _complement_map[a]
        c = tuple((comp_a[i] - v[i]) % 3 for i in range(n))
        if c in _accepted_vectors:
            count += 1
    return count

def _hamming_weight(v):
    """Count of non-zero coordinates (or distance from all-zeros)."""
    return sum(1 for x in v if x != 0)

def priority(vector, n):
    """Priority for greedy cap set construction.
    
    Uses a global saturation metric: counts how many affine lines the vector
    would form with pairs of already-accepted vectors. Vectors with fewer
    conflicts are prioritized to maximize long-term cap set capacity.
    Also incorporates geometric balance heuristics and Hamming-weight
    symmetry breaking.
    """
    # Deadline guard
    if time.time() - _start_time > _DEADLINE:
        return -1e18
    
    v = tuple(vector)
    n = len(v)
    
    # Core: conflict count - number of existing vectors a where (-a-v) mod 3 is also accepted
    # This directly measures geometric conflict potential
    if _accepted_list:
        conflicts = _count_conflicts(v)
    else:
        conflicts = 0
    
    # Count of each coordinate value
    counts = [0, 0, 0]
    for x in v:
        counts[x] += 1
    
    # Entropy of coordinate distribution
    entropy = 0.0
    for c in counts:
        if c > 0:
            p = c / n
            entropy -= p * math.log(p + 1e-15)
    
    # Number of distinct coordinate values used
    distinct = sum(1 for c in counts if c > 0)
    
    # Balance: prefer vectors with all three values present and well-distributed
    balance_score = entropy + 0.3 * distinct
    
    # Geometric heuristic: vectors with coordinates closer to 1 (middle)
    # are more likely to form lines. Higher center_deviation = safer.
    center_deviation = sum(abs(x - 1) for x in v) / n
    geo_score = center_deviation
    
    # Hamming weight: symmetry-breaking tie-breaker
    hw = _hamming_weight(v)
    # Prefer intermediate Hamming weights (neither too sparse nor too dense)
    hw_center = n / 3.0  # expected Hamming weight if uniformly random
    hw_penalty = abs(hw - hw_center) / n
    
    # Sum of coordinates mod 3
    coord_sum = sum(v) % 3
    sum_penalty = 0.1 if coord_sum == 0 else 0.0
    
    # Deterministic pseudo-random tie-breaker based on vector
    h = 0
    for i, x in enumerate(v):
        h = (h * 31 + x * (i + 1)) & 0xFFFFFFFF
    tie_break = (h / 0xFFFFFFFF) * 0.005
    
    # The conflict count is the dominant signal: minimize it
    # Use a slightly reduced weight to allow geometric heuristics more influence
    conflict_penalty = 1.5 * conflicts
    
    # Smooth surrogate: use soft penalty for potential conflicts
    # This captures vectors that are "close" to forming lines
    # by considering how many vectors a have (-a - v) close to being in the set
    # (approximated by checking a few neighbors)
    smooth_penalty = 0.0
    if _accepted_list and conflicts == 0:
        # If no hard conflicts, check for near-conflicts
        # Count vectors a where (-a - v) is at Hamming distance 1 from an accepted vector
        near_count = 0
        for a in _accepted_list[:50]:  # Limit to 50 for speed
            comp_a = _complement_map[a]
            c = tuple((comp_a[i] - v[i]) % 3 for i in range(n))
            # Check if c is at Hamming distance 1 from any accepted vector
            # This is expensive, so only do it for small sets
            if len(_accepted_vectors) <= 200:
                for acc in _accepted_vectors:
                    if sum(1 for i in range(n) if c[i] != acc[i]) == 1:
                        near_count += 1
                        break
        smooth_penalty = 0.1 * near_count
    
    total = balance_score + 0.5 * geo_score - sum_penalty - conflict_penalty - hw_penalty - smooth_penalty + tie_break
    return total

def _accept_vector(v):
    """Record that vector v has been accepted into the cap set."""
    _accepted_vectors.add(v)
    _accepted_list.append(v)
    # Precompute complement
    _complement_map[v] = tuple((-x) % 3 for x in v)

# EVOLVE-BLOCK-END
