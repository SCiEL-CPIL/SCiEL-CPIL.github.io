import math
import time
import zlib
import struct

# Global deadline tracking
_start_time = None
_DEADLINE_FRACTION = 0.95  # Use 95% of time budget

def priority(vector, n):
    """
    Priority function for greedy cap set construction in {0,1,2}^n.
    
    Strategy:
    1. Primary: Prioritize vectors with all 3 distinct symbols.
    2. Secondary: Maximize Shannon Entropy of the symbol distribution.
       Higher entropy implies more "random" distribution, which tends to
       spread out the vectors in the space, reducing the likelihood of
       forming affine lines with other similarly distributed vectors.
    3. Tertiary: Robust hash for deterministic tie-breaking.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Check deadline
    if time.time() - _start_time > 300 * _DEADLINE_FRACTION:
        # Return a very low priority to stop processing if we are close to timeout
        # This is a safeguard, though the evaluator usually handles this.
        return -1e10

    # Count symbols
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
    
    counts = [c0, c1, c2]
    distinct_count = sum(1 for c in counts if c > 0)
    
    # Smooth distinct score: use log-sum-exp style soft-max over indicator
    # This avoids hard tier jumps. For distinct_count d, score ~ d * 10000 but smoothed
    # Use: sum over i of (1 - exp(-counts[i])) which is 0 if count=0, ~1 if count>0
    smooth_distinct = 0.0
    for c in counts:
        if c > 0:
            smooth_distinct += 1.0 - math.exp(-c)
    # Scale to be comparable to distinct_count * 10000
    distinct_score = smooth_distinct * 10000.0
    
    # Shannon Entropy
    entropy = 0.0
    for c in counts:
        if c > 0:
            p = c / n
            entropy -= p * math.log(p)
    
    # Hamming weight: count of non-zero elements
    hamming_weight = c1 + c2
    weight_deviation = abs(hamming_weight - n / 2.0)
    weight_score = -weight_deviation * weight_deviation * 100.0
    
    # Balance score: -sum(p_i^2) maximized when uniform
    p0 = c0 / n
    p1 = c1 / n
    p2 = c2 / n
    balance_score = -(p0 * p0 + p1 * p1 + p2 * p2) * 5000.0
    
    # Multiplicative complexity: product of (count_i + 1) / (n/3 + 1)
    # Rewards vectors where all three symbols are present in reasonable proportions
    # This is a "full-rank" structural score
    prod_factor = 1.0
    for c in counts:
        prod_factor *= (c + 1.0) / (n / 3.0 + 1.0)
    # Normalize: max value when c0=c1=c2=n/3 is (n/3+1)^3/(n/3+1)^3 = 1
    # Min value when one symbol is 0: (1/(n/3+1)) * ... 
    # Scale to contribute meaningfully
    mult_score = prod_factor * 3000.0
    
    # Secondary score: Higher entropy is better
    entropy_score = entropy * 1000.0
    
    # Tertiary: CRC32 for fast deterministic tie-breaking
    # Pack vector into bytes for CRC32
    h = zlib.crc32(struct.pack('B' * n, *vector)) & 0xFFFFFFFF
    
    # Normalize hash to [0, 1]
    hash_score = h / 4294967296.0
    
    return float(distinct_score + entropy_score + balance_score + weight_score + mult_score + hash_score)
