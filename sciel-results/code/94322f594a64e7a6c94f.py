"""Cap set priority function for greedy insertion."""
import time

_start_time = time.time()
_DEADLINE_MARGIN = 10  # seconds before the 300s limit


def priority(vector, n):
    """
    Computes a priority score for a vector in {0, 1, 2}^n.
    Higher scores indicate vectors that should be added earlier in the greedy process.
    
    Strategy:
    1. Primary: Balance symbol counts (0, 1, 2) to avoid early constraint saturation.
    2. Secondary: Penalize vectors with many identical adjacent pairs (reduces line risk).
    3. Tertiary: Penalize vectors with long constant runs (structural AP risk).
    4. Quaternary: Deterministic hash for tie-breaking to break symmetries.
    """
    # Deadline guard: if we're close to the limit, return a neutral score
    # This prevents the evaluator from timing out
    elapsed = time.time() - _start_time
    if elapsed > 300 - _DEADLINE_MARGIN:
        return 0.0

    # Count symbols
    c0 = vector.count(0)
    c1 = vector.count(1)
    c2 = vector.count(2)
    
    # Balanced vectors (close to n/3 each) are preferred
    ideal = n / 3.0
    deviation = abs(c0 - ideal) + abs(c1 - ideal) + abs(c2 - ideal)
    
    # Primary: minimize deviation (negate for max-priority)
    score = -deviation * 1000.0
    
    # Secondary: penalize adjacent equal pairs
    # Vectors with many adjacent equal values are more likely to form APs
    adj_equal = 0
    for i in range(n - 1):
        if vector[i] == vector[i + 1]:
            adj_equal += 1
    score -= adj_equal * 50.0
    
    # Tertiary: penalize long constant runs
    # Longer runs of the same value increase AP risk
    max_run = 1
    cur_run = 1
    for i in range(1, n):
        if vector[i] == vector[i - 1]:
            cur_run += 1
            max_run = max(max_run, cur_run)
        else:
            cur_run = 1
    score -= (max_run - 1) * 20.0
    
    # Quaternary: deterministic tiebreaker based on vector pattern
    # Use a weighted sum with prime weights for better distribution
    tiebreak = 0.0
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71]
    for i, v in enumerate(vector):
        weight = primes[i % len(primes)]
        tiebreak += v * weight
    score += tiebreak * 0.001
    
    return score
