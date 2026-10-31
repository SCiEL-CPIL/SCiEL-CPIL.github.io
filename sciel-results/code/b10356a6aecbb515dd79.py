"""Cap set construction via geometric-center priority with entropy refinement."""
import time
import math

# EVOLVE-BLOCK-START

# Global state for tracking accepted vectors
_accepted_vectors = set()
_accepted_list = []
_start_time = time.time()
_DEADLINE = 320  # safety margin before 330s kill

def _fast_hash(vector, n, seed=0):
    """Fast deterministic pseudo-random priority via linear prime-based hash."""
    # Use a simple but effective hash that's fast for large n
    h = 0
    # Process in chunks for speed
    for i, v in enumerate(vector):
        h = (h * 31 + v * (i + 17) + seed) & 0xFFFFFFFFFFFFFFFF
    # Mix
    h = (h ^ (h >> 32)) * 0xff51afd7ed558ccd
    h = (h ^ (h >> 33)) & 0xFFFFFFFFFFFFFFFF
    return h / (1 << 64)

def _geometric_center_score(count_0, count_1, count_2, n):
    """
    Product of distances to all-zeros and all-twos vectors.
    Vectors with many 1s and balanced 0s/2s are prioritized because they're
    geometrically central and less likely to form 3-term APs with boundary vectors.
    """
    dist_to_zero = count_1 + 2 * count_2
    dist_to_two = 2 * count_0 + count_1
    center_score = dist_to_zero * dist_to_two
    
    # Diversity bonus: reward having all three values present
    diversity_bonus = 0.0
    if count_0 > 0 and count_1 > 0 and count_2 > 0:
        # Scale bonus by how balanced the distribution is
        min_count = min(count_0, count_1, count_2)
        diversity_bonus = min_count * 2.0
    elif count_1 > 0 and (count_0 > 0 or count_2 > 0):
        # Partial bonus for having 1s and at least one other type
        diversity_bonus = 0.5 * min(count_0 + count_2, count_1)
    
    return center_score + diversity_bonus

def _entropy_score(count_0, count_1, count_2, n):
    """
    Shannon entropy of the coordinate distribution.
    Higher entropy means more uniform distribution, which correlates with
    better cap set properties.
    """
    entropy = 0.0
    for c in [count_0, count_1, count_2]:
        if c > 0:
            p = c / n
            entropy -= p * math.log(p, 3)  # log base 3 to normalize to [0,1]
    return entropy

def priority(vector, n):
    """
    Priority function for greedy cap set construction.
    Combines geometric centering, entropy balance, and hash-based diversity.
    """
    # Deadline guard
    if time.time() - _start_time > _DEADLINE:
        return -1e18
    
    v = tuple(vector)
    
    # Count coordinate values
    count_0 = 0
    count_1 = 0
    count_2 = 0
    for x in vector:
        if x == 0:
            count_0 += 1
        elif x == 1:
            count_1 += 1
        else:
            count_2 += 1
    
    # Core: geometric center score - dominant signal
    geo_score = _geometric_center_score(count_0, count_1, count_2, n)
    
    # Entropy: measure of distribution balance
    entropy = _entropy_score(count_0, count_1, count_2, n)
    
    # Balance: prefer n1 and n2 to be close to each other
    balance = abs(count_1 - count_2) / max(1, n)
    balance_score = (1.0 - balance) * 0.3
    
    # Multiple hash layers for robustness (fast)
    h1 = _fast_hash(vector, n, seed=1)
    h2 = _fast_hash(vector, n, seed=2)
    
    # Hash component provides diversity and tie-breaking
    hash_score = (h1 * 0.6 + h2 * 0.4) * 0.1
    
    # Combine: geometric center dominates, entropy and balance provide refinement
    # Scale geometric score to be the primary signal
    total = geo_score * 0.1 + entropy * 2.0 + balance_score + hash_score
    
    return total

# EVOLVE-BLOCK-END
