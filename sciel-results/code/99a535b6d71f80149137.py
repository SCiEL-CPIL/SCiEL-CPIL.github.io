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

def _is_cap_set_safe(vec, current_set):
    """Check if adding vec to current_set keeps it a cap set.
    A cap set has no three distinct points a,b,c with a+b+c ≡ 0 mod 3.
    """
    for a in current_set:
        for b in current_set:
            if a == b or b == vec or a == vec:
                continue
            # Check if a, b, vec form an affine line: a + b + vec ≡ 0 mod 3
            # i.e., vec ≡ -a-b mod 3
            diff = [(-a[i] - b[i] - vec[i]) % 3 for i in range(len(vec))]
            if all(d == 0 for d in diff):
                return False
    return True

def _count_conflicts(v):
    """Count how many existing vectors a have (-a - v) mod 3 also in the set.
    This measures how many 'witness' lines the vector would participate in.
    """
    n = len(v)
    count = 0
    for a in _accepted_list:
        # Compute c = (-a - v) mod 3
        c = [(-a[i] - v[i]) % 3 for i in range(n)]
        if tuple(c) in _accepted_vectors:
            count += 1
    return count

def priority(vector, n):
    """Priority for greedy cap set construction.
    
    Uses a global saturation metric: counts how many affine lines the vector
    would form with pairs of already-accepted vectors. Vectors with fewer
    conflicts are prioritized to maximize long-term cap set capacity.
    Also incorporates geometric balance heuristics.
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
    
    # Sum of coordinates mod 3
    coord_sum = sum(v) % 3
    sum_penalty = 0.1 if coord_sum == 0 else 0.0
    
    # Deterministic pseudo-random tie-breaker based on vector
    h = 0
    for i, x in enumerate(v):
        h = (h * 31 + x * (i + 1)) & 0xFFFFFFFF
    tie_break = (h / 0xFFFFFFFF) * 0.005
    
    # The conflict count is the dominant signal: minimize it
    # Scale it down since it can grow with set size
    conflict_penalty = 2.0 * conflicts
    
    total = balance_score + 0.5 * geo_score - sum_penalty - conflict_penalty + tie_break
    return total

def _accept_vector(v):
    """Record that vector v has been accepted into the cap set."""
    _accepted_vectors.add(v)
    _accepted_list.append(v)

# EVOLVE-BLOCK-END
