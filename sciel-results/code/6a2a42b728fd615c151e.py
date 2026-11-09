import time

_T0 = time.time()
_DEADLINE = _T0 + 280.0

def priority(vector, n):
    if time.time() > _DEADLINE:
        return 0.0

    count_0 = 0
    count_1 = 0
    count_2 = 0
    for v in vector:
        if v == 0:
            count_0 += 1
        elif v == 1:
            count_1 += 1
        else:
            count_2 += 1
            
    # Primary: Minimize "AP-prone" pairs (sum to 2 mod 3)
    # Pairs (0,2), (2,0), (1,1)
    # Count of (0,2) pairs: count_0 * count_2
    # Count of (1,1) pairs: count_1 * (count_1 - 1) // 2
    bad_pairs = count_0 * count_2 + (count_1 * (count_1 - 1)) // 2
    
    # Secondary: Balance count_1 around n/2
    primary_dev = abs(2 * count_1 - n)
    
    # Combine: Minimize bad_pairs first, then primary_dev
    # Max bad_pairs: n=10, count_0=5, count_2=5 -> 25. count_1=0 -> 0.
    # Max primary_dev: 10.
    # Weight bad_pairs more?
    # Let's say bad_pairs * 10 + primary_dev
    penalty = bad_pairs * 10 + primary_dev
    
    # Higher is better
    # Max penalty ~ 10 * (n^2/4) + n
    max_penalty = (n * n) // 4 * 10 + n
    layer_score = max_penalty - penalty
    
    # Hash for tie-breaking
    idx = 0
    for v in reversed(vector):
        idx = idx * 3 + v

    def mix64(h):
        h ^= h >> 33
        h = (h * 0xff51afd7ed558ccd) & 0xFFFFFFFFFFFFFFFF
        h ^= h >> 33
        h = (h * 0xc4ceb9fe1a85ec53) & 0xFFFFFFFFFFFFFFFF
        h ^= h >> 33
        return h

    seed_val = idx ^ (n * 0x9E3779B97F4A7C15)
    mixed = mix64(seed_val)
    
    hash_fraction = mixed / (2**64)
    
    return float(layer_score + hash_fraction)
