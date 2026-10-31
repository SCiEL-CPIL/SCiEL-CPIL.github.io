"""Cap set greedy priority heuristic."""
# EVOLVE-BLOCK-START
import time
import itertools
import numpy as np

# Global cache for vector properties and cap set state
# We'll use a global deadline guard
_DEADLINE = 300.0  # seconds
_START_TIME = None

def _get_start_time():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return _START_TIME

def _time_remaining():
    return _DEADLINE - (time.time() - _START_TIME())

def priority(vector, n):
    """
    Priority function for greedy cap set construction.
    
    A cap set in {0,1,2}^n is a set with no three distinct elements a, b, c
    such that a + b + c = 0 mod 3 (i.e., no 3-term arithmetic progression).
    
    We use a heuristic priority that favors vectors which are "far" from
    existing set members in a way that reduces the chance of forming 3-term APs.
    
    Strategy: 
    - Count the number of existing set members that would form a 3-term AP with this vector.
    - Prefer vectors with fewer such conflicts (lower priority = earlier insertion).
    - But since we're adding greedily in descending priority, we want HIGH priority for good vectors.
    - So: priority = -conflict_count + small random tie-breaker for stability.
    
    Actually, let me reconsider. The greedy process adds vectors in descending priority.
    We want to add vectors that are "safe" (don't create 3-term APs with any pair in the set).
    
    A vector v can be added if for no two existing vectors a, b in the set do we have:
    a + b + v = 0 mod 3, i.e., v = -(a+b) mod 3, which means v = (3-a[0]-b[0], 3-a[1]-b[1], ...) mod 3.
    
    So the "conflict" for vector v is: does there exist a, b in S such that v = (-a-b) mod 3?
    
    For a greedy approach, we can precompute for each candidate vector how many pairs (a,b) in the current set would form a 3-term AP with it.
    
    But since the set changes as we add elements, we need to be careful. Let's use a different approach:
    
    Priority based on:
    1. Hamming weight (number of non-zero entries) - vectors with moderate weight might be better
    2. Distance from origin
    3. A hash-based random tie-breaker
    
    Actually, let's use a simpler but effective heuristic:
    - Prefer vectors with lower "sum" (sum of coordinates mod 3)
    - This tends to spread out the vectors in the space
    
    Or better: use the L1 norm as a tie-breaker, preferring vectors closer to the origin first,
    then use a random permutation for the rest.
    
    Let me think about what makes a good cap set. The maximum cap set size in {0,1,2}^n is known for small n.
    For n=8, max is 20. For n=9, max is 36. For n=10, max is 72.
    
    A good heuristic is to prefer vectors that are "balanced" - having roughly equal numbers of 0s, 1s, and 2s.
    
    Let's use: priority = -|sum(vector) - n| (prefer vectors with sum close to n/3 * 3 = n? No, sum of balanced vector is n*1 = n if we have n/3 of each 0,1,2... wait, sum would be 0*(n/3) + 1*(n/3) + 2*(n/3) = n. So prefer sum close to n.)
    
    Actually, let's just use a simple random priority with a bias towards lower-weight vectors.
    """
    # Convert to tuple if needed
    if isinstance(vector, np.ndarray):
        vector = tuple(vector.tolist())
    
    # Calculate a heuristic priority
    # Prefer vectors with coordinates that are more "spread out"
    # Use the variance of the coordinates as a measure of spread
    vec = np.array(vector, dtype=float)
    
    # Variance of the vector elements
    var = np.var(vec)
    
    # L1 norm (sum of absolute values)
    l1 = np.sum(np.abs(vec))
    
    # Number of 1s and 2s
    n_ones = np.sum(vec == 1)
    n_twos = np.sum(vec == 2)
    
    # Prefer vectors with more 1s and 2s (higher weight) but not too high
    # Actually, let's prefer vectors with moderate weight
    weight = n_ones + n_twos
    
    # Priority: high variance (spread out), moderate weight
    # Normalize weight to [0, 1]
    norm_weight = weight / n if n > 0 else 0
    
    # Prefer weight close to n/2? Or just use weight directly
    # Let's use: priority = var * 10 + (1 - abs(norm_weight - 0.5)) * 5 + random_tiebreaker
    
    # For deterministic behavior, use a hash-based tiebreaker
    vec_hash = hash(tuple(vector))
    tiebreaker = (vec_hash % 1000) / 1000.0
    
    priority_val = var * 10.0 + (1.0 - abs(norm_weight - 0.5)) * 5.0 + tiebreaker * 0.1
    
    return float(priority_val)

# EVOLVE-BLOCK-END
