"""
Cap set construction via greedy insertion with heuristic priority.

The evaluator calls priority(vector, n) for each candidate vector in {0,1,2}^n,
and inserts it into the current set if doing so keeps the set a cap set (no 3-term AP).

Key structural insight: A 3-term AP in {0,1,2}^n is a triple (a, b, c) where a + c = 2b
coordinate-wise. In each coordinate:
- If b_i = 0: a_i = c_i = 0 (trivial, always satisfied)
- If b_i = 1: a_i = c_i = 1 OR {a_i, c_i} = {0, 2}
- If b_i = 2: a_i = c_i = 2 (trivial, always satisfied)

So non-trivial AP structure only comes from coordinates where b_i = 1 and the pair
(a_i, c_i) = (0, 2) or (2, 0).

Strategy: We want vectors that are "structurally safe" - they participate in few potential
3-term APs. The key heuristics:
1. Balance of 1-count: vectors with count_1 near n/2 are more "central" and less likely
   to be extreme points that form APs.
2. Spatial spread of 1-positions: 1s spread across distant coordinates reduce the chance
   that two other vectors in the set can "sandwich" this vector in an AP.
3. Zero-position spread: zeros distributed across distant coordinates mean the vector
   is "far" from the all-zeros vector in a structural sense, reducing symmetric triples.

We use a deterministic hash-based tie-breaker for stability.
"""

import time
import math

_START_TIME = time.time()
_DEADLINE = _START_TIME + 290

def _check_deadline():
    return time.time() < _DEADLINE

def _spatial_spread(positions, n):
    """Compute normalized spatial spread of positions in [0, n-1]."""
    k = len(positions)
    if k <= 1:
        return 0.0
    mean_pos = sum(positions) / k
    spread = sum((p - mean_pos) ** 2 for p in positions)
    # Maximum possible spread for k points in [0, n-1]
    # The max is achieved by putting k/2 at 0 and k/2 at n-1 (or similar extremes)
    # A simple upper bound: k * (n/2)^2
    max_spread = k * (n / 2.0) ** 2
    if max_spread < 1e-9:
        return 0.0
    return spread / max_spread

def priority(vector, n):
    """
    Static priority for cap set construction.
    
    Primary: balance of 1-count around n/2
    Secondary: spatial spread of 1-positions
    Tertiary: spatial spread of 0-positions
    Quaternary: small deterministic tie-breaker for stability
    """
    if not _check_deadline():
        return 0.0
    
    # Count 1s and 0s
    count_1 = 0
    count_0 = 0
    positions_1 = []
    positions_0 = []
    
    for i, x in enumerate(vector):
        if x == 1:
            count_1 += 1
            positions_1.append(i)
        elif x == 0:
            count_0 += 1
            positions_0.append(i)
    
    # Primary: balance score (higher when count_1 is close to n/2)
    target = n / 2.0
    balance_score = -abs(count_1 - target)
    
    # Secondary: spatial spread of 1-positions
    spread_1 = _spatial_spread(positions_1, n)
    
    # Tertiary: spatial spread of 0-positions
    spread_0 = _spatial_spread(positions_0, n)
    
    # Combine: balance is primary, spreads are secondary/tertiary
    # Weight so that balance dominates, then 1-spread, then 0-spread
    combined = balance_score + 0.15 * spread_1 + 0.10 * spread_0
    
    # Quaternary: small hash-based tie-breaker for determinism
    h = hash(vector) % 1000000
    combined += h * 1e-9
    
    return combined

# EVOLVE-BLOCK-START
# The above priority function is the main logic
# EVOLVE-BLOCK-END
