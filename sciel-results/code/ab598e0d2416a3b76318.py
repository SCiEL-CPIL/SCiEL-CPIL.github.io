"""Cap set greedy builder with geometric-aware priority."""
import time
import itertools
import numpy as np

# EVOLVE-BLOCK-START
import math
import zlib
_start_time = time.time()
_DEADLINE = 325  # seconds, safety margin before 330s kill

# Global state for tracking accepted vectors and computing conflict scores
_accepted_vectors = set()
_accepted_list = []
# Precomputed: for each accepted vector a, store the "complement" (-a) mod 3
# to speed up conflict checking
_complement_map = {}  # maps vector -> its complement (-v) mod 3

# Smooth surrogate: per-coordinate conflict counts
# For each coordinate index i and value val (0,1,2), count how many
# accepted vectors a have a[i] = val. This lets us cheaply estimate
# how many "near-conflict" lines v would participate in.
# near_conflict_pressure[i][val] = count of accepted a with a[i] == val
_coord_pressure = []  # list of [count0, count1, count2] per coordinate index
_current_n = 0

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

def _smooth_conflict_score(v):
    """Smooth surrogate for conflict pressure using per-coordinate counts.
    
    For each coordinate i, the number of accepted vectors a that would
    create a line through v at position i is proportional to how many
    accepted vectors have the "opposite" value at position i.
    Specifically, a line through v at position i requires a[i] + v[i] ≡ 0 (mod 3)
    for the third point to be determined, but the actual conflict is when
    (-a - v) mod 3 is in the set. A cheaper proxy: count how many accepted
    vectors a satisfy a[i] == (-v[i]) mod 3 for each i, sum over i.
    This captures the "density" of the set in directions that conflict with v.
    """
    n = len(v)
    if n != _current_n or not _coord_pressure:
        return 0.0
    total = 0.0
    for i in range(n):
        # The "conflicting" value at position i for vector v is (-v[i]) mod 3
        conflict_val = (-v[i]) % 3
        total += _coord_pressure[i][conflict_val]
    return total

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
    
    # CRC32-based deterministic pseudo-random tie-breaker (faster than manual hash)
    try:
        h = zlib.crc32(v) & 0xFFFFFFFF
    except Exception:
        h = 0
    tie_break = (h / 0xFFFFFFFF) * 0.005
    
    # The conflict count is the dominant signal: minimize it
    conflict_penalty = 1.5 * conflicts
    
    # Smooth surrogate: O(n) per-coordinate conflict pressure
    # This is a cheap proxy for how "crowded" the set is in directions
    # that would create lines with v. Computed in O(n) instead of O(n*|S|).
    smooth_penalty = 0.05 * _smooth_conflict_score(v)
    
    total = balance_score + 0.5 * geo_score - sum_penalty - conflict_penalty - hw_penalty - smooth_penalty + tie_break
    return total

def _accept_vector(v):
    """Record that vector v has been accepted into the cap set."""
    global _coord_pressure, _current_n
    _accepted_vectors.add(v)
    _accepted_list.append(v)
    # Precompute complement
    _complement_map[v] = tuple((-x) % 3 for x in v)
    # Update per-coordinate pressure
    n = len(v)
    if n != _current_n:
        # Reset for new dimension
        _coord_pressure = [[0, 0, 0] for _ in range(n)]
        _current_n = n
        # Rebuild from all accepted vectors
        for a in _accepted_list:
            for i in range(n):
                _coord_pressure[i][a[i]] += 1
    else:
        for i in range(n):
            _coord_pressure[i][v[i]] += 1

# EVOLVE-BLOCK-END
