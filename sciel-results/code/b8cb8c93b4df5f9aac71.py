"""Cap set greedy builder with geometric line-avoidance priority."""
import time
from functools import lru_cache

# EVOLVE-BLOCK-START
def priority(vector, n):
    """
    Priority for greedy cap set construction.
    
    The evaluator calls this to order vectors. We use a multi-factor heuristic:
    1. Coordinate balance: vectors with more distinct coordinates spread out
    2. Sum balance: vectors whose coordinate sum is near n (mod 3) are less likely
       to form lines with balanced pairs
    3. A static geometric score based on pairwise line-formation potential
    
    Since priority() is called per-vector without knowledge of the current set state,
    we compute a static score that correlates with low line-formation probability.
    """
    # Convert to tuple for hashing
    v = tuple(vector)
    
    # Factor 1: Number of distinct coordinates (more spread = better)
    distinct = len(set(v))
    distinct_score = distinct / n if n > 0 else 0.0
    
    # Factor 2: Coordinate sum modulo 3 balance
    # In a cap set, we want to avoid configurations where three vectors sum to 0 mod 3
    # Vectors with sum ≡ 1 or 2 (mod 3) are less likely to complete lines with
    # vectors of complementary sums
    coord_sum = sum(v)
    sum_mod = coord_sum % 3
    # Prefer sum_mod == 1 (slightly), penalize sum_mod == 0 (most dangerous)
    sum_score = {0: 0.0, 1: 1.0, 2: 0.8}[sum_mod]
    
    # Factor 3: Pairwise coordinate agreement pattern
    # Count how many coordinates are 0, 1, 2
    counts = [0, 0, 0]
    for x in v:
        counts[x] += 1
    # Balanced vectors (close to n/3 each) are more "spread out"
    balance = 1.0 - (abs(counts[0] - counts[1]) + abs(counts[1] - counts[2]) + abs(counts[0] - counts[2])) / (3 * n) if n > 0 else 0.0
    
    # Factor 4: Hash-based tie-breaking for determinism
    hash_score = (hash(v) % 1000) / 1000.0 * 0.1  # Small tiebreaker
    
    # Combined priority: higher is better (added first)
    # Weight the factors to emphasize geometric spread
    total = 0.35 * distinct_score + 0.30 * sum_score + 0.25 * balance + hash_score
    
    return total
# EVOLVE-BLOCK-END
