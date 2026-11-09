import time
import math

_T0 = time.time()
_DEADLINE = _T0 + 280.0

def priority(vector, n):
    """
    Priority for a vector in {0,1,2}^n for greedy cap-set construction.
    
    Strategy:
    1. Structural Balance: Vectors with symbol counts close to n/3 are preferred.
    2. 0/2 Spread: Vectors where 0s and 2s are balanced are preferred to avoid 
       clustering in subspaces defined by fixed 0s or 2s.
    3. High-Quality Deterministic Hash: A strong bit-mixing hash breaks symmetries 
       and prevents the greedy algorithm from clustering in specific structural regions, 
       allowing for a more diverse and larger cap set.
    """
    if time.time() > _DEADLINE:
        return 0.0

    c0 = 0
    c1 = 0
    c2 = 0
    for v in vector:
        if v == 0:
            c0 += 1
        elif v == 1:
            c1 += 1
        else:
            c2 += 1

    # 1. Structural Balance Score
    # Target is n/3 of each symbol.
    target = n / 3.0
    # Squared deviation from ideal balance
    penalty_balance = (c0 - target) ** 2 + (c1 - target) ** 2 + (c2 - target) ** 2
    
    # 2. 0/2 Imbalance Penalty
    # In F_3, 0 and 2 are symmetric. If a vector has many 0s and few 2s (or vice versa),
    # it might be "biased" towards a specific hyperplane structure.
    # We penalize the squared difference between count of 0s and 2s.
    # Scale by n to keep magnitude comparable to penalty_balance.
    diff_02 = c0 - c2
    penalty_02 = (diff_02 * diff_02) / max(n, 1)

    # Combined structural score
    # We want to MINIMIZE penalties, so we negate them.
    structural_score = -penalty_balance - 0.5 * penalty_02
    
    # 3. Deterministic Hash for Tie-breaking and Diversity
    # We create a unique integer representation of the vector using base-3 encoding.
    # This is fast and covers the entire space uniquely.
    val = 0
    for v in vector:
        val = val * 3 + v
        
    # Strong bit-mixing (splitmix64 finalizer style) to ensure good distribution
    # Even if val is small (for small n), the mixing ensures high entropy in the bits.
    val = (val ^ (val >> 30)) * 0xbf58476d1ce4e5b9
    val = (val ^ (val >> 27)) * 0x94d049bb133111eb
    val = val ^ (val >> 31)
    
    # Convert to a small float in [-1, 1]
    # We use modulo 2^64 to handle potential large integers, then normalize.
    # Python integers are arbitrary precision, but the mixing constants are 64-bit.
    # We mask to 64 bits to ensure consistent behavior and speed.
    val_64 = val & ((1 << 64) - 1)
    
    # Map to [-1, 1]
    # Dividing by 2^63 gives [0, 2). Subtracting 1 gives [-1, 2).
    # We want a small perturbation.
    # Using a small coefficient ensures this only breaks ties.
    # The structural score is typically in the range of -10 to 0 for n~10-20.
    # A tie-breaker of +/- 0.01 is safe.
    hash_float = (val_64 / (1 << 63)) - 1.0
    tie_break = hash_float * 0.01

    final_score = structural_score + tie_break

    return final_score
