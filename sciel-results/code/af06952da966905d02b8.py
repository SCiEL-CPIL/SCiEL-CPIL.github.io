import time
import math

_start_time = None
_DEADLINE = 330 * 0.95  # 313.5s safety margin

# Precomputed constants
_MAX_ENTROPY = math.log(3)

def priority(vector, n):
    """
    Priority function for greedy cap set construction in {0,1,2}^n.
    
    Strategy (combined design, smooth surrogates + bitwise parity):
    1. Primary: Smooth quadratic distinct-symbol score. Quadratic in the number
       of present symbols (0,1,4,9 * 10000) gives a smoother landscape.
    2. Secondary: Hamming weight relative to all-ones (count of 2s). Favors
       vectors "heavy" in the 2-coordinate to break symmetry.
    3. Tertiary: Quadratic balance penalty. L2 deviation from uniform distribution.
    4. Quaternary: Smooth variance-based danger proxy. High variance = higher line risk.
    5. Quinary: Bitwise parity-based chaotic interleaving. Treats vector as ternary
       integer, XORs with a fixed mask, and uses bit count for deterministic
       pseudo-random tie-breaking that breaks correlation with local symbol balance.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    elif time.time() - _start_time > _DEADLINE:
        # Past deadline: return a constant to signal "stop caring about ordering"
        return 0.0

    # Fast symbol counting in one pass (collects counts, sum, and sum-of-squares)
    c0 = 0
    c1 = 0
    c2 = 0
    s = 0
    s2 = 0
    # Also compute ternary integer representation for bitwise parity
    ternary_int = 0
    for i, v in enumerate(vector):
        if v == 0:
            c0 += 1
        elif v == 1:
            c1 += 1
            s += 1
            s2 += 1
        else:  # v == 2
            c2 += 1
            s += 2
            s2 += 4
        # Build ternary integer: digit i is at position i
        ternary_int |= v << (2 * i)  # Each ternary digit needs 2 bits (0,1,2)

    # Factor 1: Smooth quadratic distinct-symbol score.
    # Quadratic in presence count: 0 -> 0, 1 -> 10000, 2 -> 40000, 3 -> 90000.
    num_distinct = (1 if c0 > 0 else 0) + (1 if c1 > 0 else 0) + (1 if c2 > 0 else 0)
    distinct_score = 10000.0 * num_distinct * num_distinct

    # Factor 2: Hamming weight relative to all-ones (count of 2s).
    # Favors vectors with more 2s to break symmetry and explore the full space.
    weight_score = c2 * 100.0

    # Factor 3: Quadratic balance penalty (smooth surrogate for L1).
    # L2 deviation from the uniform (n/3, n/3, n/3) distribution.
    ideal = n / 3.0
    dev_sq = (c0 - ideal) ** 2 + (c1 - ideal) ** 2 + (c2 - ideal) ** 2
    balance_score = -dev_sq * 5.0

    # Factor 4: Smooth variance-based danger proxy.
    # High coordinate variance = more extreme coordinates = higher line risk.
    if n > 0:
        mean = s / n
        variance = s2 / n - mean * mean
    else:
        variance = 0.0
    danger_score = -variance * 5000.0

    # Factor 5: Bitwise parity-based chaotic interleaving.
    # XOR the ternary integer with a fixed mask, then count bits.
    # This creates a deterministic but pseudo-random ordering that breaks
    # statistical correlation between local symbol balance and global line formation.
    # The mask is chosen to have good bit-distribution properties.
    mask = 0x5A5A5A5A  # Alternating pattern for good bit distribution
    xor_val = ternary_int ^ mask
    # Count set bits (popcount)
    bit_count = bin(xor_val).count('1')
    # Normalize to [0, 1] range and scale
    max_bits = 2 * n  # Max bits in ternary representation
    parity_score = (bit_count / max_bits if max_bits > 0 else 0.0) * 100.0

    # Combined priority: higher is added first.
    total = distinct_score + weight_score + balance_score + danger_score + parity_score
    return float(total)
