"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import hashlib

def priority(vector, n):
    """
    Priority for cap set greedy insertion.
    Uses a combination of:
    1. Symbol balance (entropy-like measure) - rewards uniform distribution
    2. Sum mod 3 class - structural grouping
    3. A hash-based tiebreaker for deterministic diversity
    """
    # Count symbols
    c0 = c1 = c2 = 0
    s = 0
    for x in vector:
        if x == 0:
            c0 += 1
        elif x == 1:
            c1 += 1
        else:
            c2 += 1
        s += x
    
    # Balance measure: how close to uniform (n/3 each)
    # Use negative variance from uniform
    target = n / 3.0
    var = (c0 - target) ** 2 + (c1 - target) ** 2 + (c2 - target) ** 2
    
    # Sum mod 3 class: prefer 0, then 1, then 2
    smod = s % 3
    class_bonus = {0: 2.0, 1: 1.0, 2: 0.0}[smod]
    
    # Hash for tiebreaking
    h = int(hashlib.md5(str(vector).encode()).hexdigest(), 16) % (10**9)
    hash_component = h / 10**9
    
    # Combine: balance is primary, class is secondary, hash is tertiary
    # Normalize variance: max variance is n^2 (all in one symbol), min is ~n/3
    # Scale so that balance dominates
    balance_score = -var / (n * n + 1)  # in [-1, 0], closer to 0 is better
    class_score = class_bonus / 3.0      # in [0, 0.667]
    
    # Weight: balance 70%, class 20%, hash 10%
    total = 0.7 * balance_score + 0.2 * class_score + 0.1 * hash_component
    
    return total
# EVOLVE-BLOCK-END
