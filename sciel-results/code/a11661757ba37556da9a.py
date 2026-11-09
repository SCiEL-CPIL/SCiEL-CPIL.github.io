"""Cap set greedy construction with balanced-digit priority and deadline guard."""
import time
import itertools
from functools import lru_cache

# EVOLVE-BLOCK-START

_DEADLINE_SECONDS = 320.0  # safety margin before 330s kill
_start_time = None

def _elapsed():
    global _start_time
    if _start_time is None:
        _start_time = time.monotonic()
    return time.monotonic() - _start_time

def _remaining():
    return _DEADLINE_SECONDS - _elapsed()

def priority(vector, n):
    """
    Structural change: Weighted zero-position bias.
    
    Key insight: 0 is the additive identity in F_3. For a 3-term AP (a, b, c),
    we need a + c = 2b mod 3. At any position where a[i]=c[i]=0, we must have b[i]=0.
    This constrains the AP structure. Vectors with more zeros, especially spread
    across distant coordinates, are structurally less likely to form APs.
    
    Strategy: Assign higher priority to vectors with:
    1. More total zeros (each zero reduces AP participation)
    2. Zeros spread across diverse positions (reduces correlated constraints)
    3. A small balanced-digit component as secondary heuristic
    """
    n = len(vector)
    
    # Count zeros and track their positions
    zero_count = 0
    zero_positions = []
    c1 = 0
    c2 = 0
    for i, v in enumerate(vector):
        if v == 0:
            zero_count += 1
            zero_positions.append(i)
        elif v == 1:
            c1 += 1
        else:
            c2 += 1
    
    # Component 1: Reward more zeros (primary structural bias)
    # Each zero position constrains APs; more zeros = fewer APs possible
    zero_bias = zero_count * 1.0
    
    # Component 2: Reward spread of zeros across positions
    # If zeros are clustered, they constrain similar AP structures.
    # If spread out, they constrain diverse AP structures more effectively.
    # Use pairwise distance sum as a measure of spread.
    if len(zero_positions) >= 2:
        # Sum of absolute differences between all pairs of zero positions
        # This rewards zeros that are far apart
        spread = 0.0
        zp = zero_positions
        m = len(zp)
        for i in range(m):
            for j in range(i + 1, m):
                spread += abs(zp[i] - zp[j])
        # Normalize by max possible spread
        max_spread = m * (m - 1) * (n - 1) / 2.0
        if max_spread > 0:
            spread_component = spread / max_spread
        else:
            spread_component = 0.0
    else:
        spread_component = 0.0
    
    # Component 3: Secondary balanced-digit heuristic
    c0 = zero_count
    expected = n / 3.0
    var = (c0 - expected) ** 2 + (c1 - expected) ** 2 + (c2 - expected) ** 2
    balance_component = -var * 0.1  # small weight
    
    # Component 4: Lex tie-breaking (very small)
    rank = 0
    for v in vector:
        rank = rank * 3 + v
    lex_component = -rank / (3 ** n) * 0.01
    
    # Combine: zero_bias dominates, then spread, then balance, then lex
    return zero_bias + spread_component * 0.5 + balance_component + lex_component

# EVOLVE-BLOCK-END

# The evaluator will call priority(vector, n) to get priorities, then sort all vectors
# in descending priority and greedily add them if they keep the set a cap set.
# The score is the size of the resulting cap set.
