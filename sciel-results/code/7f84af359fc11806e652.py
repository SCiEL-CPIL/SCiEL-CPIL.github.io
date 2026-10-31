import time

_START_TIME = time.time()
_DEADLINE = 300.0  # Safety margin before 330s kill

# Fixed mask for XOR-based chaotic interleaving
# Use a pattern that creates good bit distribution in base-3 representation
_MASK = 0x5A5A5A5A  # Alternating 0101 pattern

def _ternary_to_int(vec, n):
    """Convert ternary vector to integer representation."""
    val = 0
    for i in range(n):
        val = val * 3 + vec[i]
    return val

def _popcount(x):
    """Count set bits in integer."""
    count = 0
    while x:
        x &= x - 1
        count += 1
    return count

def priority(vector, n):
    # Check deadline
    if time.time() > _DEADLINE:
        return 0.0
    
    # Convert vector to ternary integer
    val = _ternary_to_int(vector, n)
    
    # XOR with fixed mask to create chaotic interleaving
    xored = val ^ _MASK
    
    # Priority based on popcount of XOR result - creates non-monotonic
    # relationship between vector components and priority
    pc = _popcount(xored)
    
    # Add secondary term based on digit sum for tie-breaking variety
    digit_sum = sum(vector)
    
    # Combine: primary chaotic signal + secondary balance signal
    # Scale to make popcount dominant
    return float(pc * 1000 + digit_sum)
