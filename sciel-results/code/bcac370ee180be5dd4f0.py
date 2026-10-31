"""Cap set construction via greedy selection with structured priorities."""
import time
import itertools

# EVOLVE-BLOCK-START

_START_TIME = time.time()
_DEADLINE = 300.0  # safety margin before 330s kill

def _elapsed():
    return time.time() - _START_TIME

def _remaining():
    return _DEADLINE - _elapsed()

def _fast_hash(vector):
    """Fast deterministic hash using linear combination with primes."""
    h = 0
    # Use different primes for each position to reduce collisions
    for i, v in enumerate(vector):
        h = (h * 131 + v * (17 + i * 31)) & 0xFFFFFFFF
    return h

def _center_score(vector, n):
    """
    Score based on distance from the center (1,1,...,1).
    Vectors closer to center have more balanced 0,1,2 distribution
    and form fewer 3-term APs, allowing denser packing.
    """
    # Distance from center: count of non-1 coordinates
    non_ones = sum(1 for v in vector if v != 1)
    # Also consider the balance of 0s and 2s
    zeros = sum(1 for v in vector if v == 0)
    twos = sum(1 for v in vector if v == 2)
    
    # Primary: fewer non-ones is better (closer to center)
    # Secondary: balance between 0s and 2s
    balance_penalty = abs(zeros - twos)
    
    # Return a score where higher = closer to center + more balanced
    # Scale so that center_score dominates the hash tiebreaker
    return -(non_ones * 10.0 + balance_penalty)

def _diversity_bonus(vector, n):
    """
    Bonus for vectors that have diverse patterns.
    Vectors with all same symbol (0,0,...,0 or 1,1,...,1 or 2,2,...,2)
    are at the boundary and form APs easily with other boundary vectors.
    """
    unique_symbols = len(set(vector))
    # More unique symbols = more diverse = slightly better
    return unique_symbols * 0.5

def _structure_score(vector, n):
    """
    Additional structural score based on coordinate patterns.
    Vectors with mixed patterns across coordinates are preferred.
    """
    # Count transitions in the vector (adjacent different values)
    transitions = 0
    for i in range(n - 1):
        if vector[i] != vector[i + 1]:
            transitions += 1
    # More transitions = more mixed = slightly better
    return transitions * 0.1

def priority(vector, n):
    """
    Priority function for greedy cap set construction.
    
    Strategy:
    1. Center-targeting: prefer vectors close to (1,1,...,1) to avoid boundary APs
    2. Balance: prefer vectors with balanced 0,1,2 distribution
    3. Diversity: slight bonus for diverse coordinate patterns
    4. Hash: deterministic tie-breaking
    
    The center-targeting is the key insight: vectors near the center of {0,1,2}^n
    form fewer 3-term arithmetic progressions, so the greedy algorithm can pack
    more vectors before hitting the cap set constraint.
    """
    # Check deadline - if close, return a safe value
    if _remaining() < 5.0:
        return 0.0
    
    # Primary: center score (dominates)
    cs = _center_score(vector, n)
    
    # Secondary: diversity bonus
    db = _diversity_bonus(vector, n)
    
    # Tertiary: structure score
    ss = _structure_score(vector, n)
    
    # Quaternary: hash for deterministic tie-breaking
    h = _fast_hash(vector) / 0xFFFFFFFF
    
    # Combine: center score dominates, others are smaller perturbations
    # Scale so center_score changes by ~10 per non-one, hash is < 1
    return cs + db + ss + h

# EVOLVE-BLOCK-END
