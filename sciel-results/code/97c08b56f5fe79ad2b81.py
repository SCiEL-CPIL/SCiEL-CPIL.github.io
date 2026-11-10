"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from fractions import Fraction
    import math
    import bisect
    
    # Denominators are 2..48, indices 0..46
    denoms = list(range(2, 49))
    n = len(denoms)  # 47
    
    # Compute LCM of all denominators 2..48
    def lcm(a, b):
        return a * b // math.gcd(a, b)
    
    L = 1
    for d in denoms:
        L = lcm(L, d)
    
    # Convert each 1/d to integer numerator over L
    # val[i] = L // denoms[i]
    vals = [L // d for d in denoms]
    
    # Meet-in-the-middle: split 47 items into two halves
    # half1: indices 0..22 (23 items), half2: indices 23..46 (24 items)
    half1_size = 23
    half2_size = 24
    
    half1_vals = vals[:half1_size]
    half2_vals = vals[half1_size:]
    
    # Generate all subset sums for half1 with index tracking
    # Each entry: (sum, frozenset of local indices)
    # To save memory, store as (sum, bitmask) where bitmask indicates which indices are included
    
    sums1 = [(0, 0)]  # (sum, bitmask)
    for i, v in enumerate(half1_vals):
        new_entries = []
        for s, mask in sums1:
            new_mask = mask | (1 << i)
            new_s = s + v
            new_entries.append((new_s, new_mask))
        sums1.extend(new_entries)
    
    # Generate all subset sums for half2 with index tracking
    sums2 = [(0, 0)]
    for i, v in enumerate(half2_vals):
        new_entries = []
        for s, mask in sums2:
            new_mask = mask | (1 << i)
            new_s = s + v
            new_entries.append((new_s, new_mask))
        sums2.extend(new_entries)
    
    # Sort sums2 by sum for binary search
    sums2.sort(key=lambda x: x[0])
    
    # Extract just the sums for binary search
    sums2_values = [s for s, _ in sums2]
    
    # Find best combination: maximize s1 + s2 subject to s1 + s2 < L
    best_total = 0
    best_sum1 = 0
    best_mask1 = 0
    best_sum2 = 0
    best_mask2 = 0
    
    for s1, mask1 in sums1:
        if s1 >= L:
            continue
        limit = L - s1
        # Find largest s2 in sums2_values with s2 < limit
        idx = bisect.bisect_left(sums2_values, limit)
        if idx > 0:
            s2, mask2 = sums2[idx - 1]
            total = s1 + s2
            if total > best_total:
                best_total = total
                best_sum1 = s1
                best_mask1 = mask1
                best_sum2 = s2
                best_mask2 = mask2
    
    # Reconstruct indices from bitmasks
    selected = []
    
    # Half 1 indices (local to half1, which are global indices 0..22)
    for i in range(half1_size):
        if best_mask1 & (1 << i):
            selected.append(i)
    
    # Half 2 indices (local to half2, which are global indices 23..46)
    for i in range(half2_size):
        if best_mask2 & (1 << i):
            selected.append(i + half1_size)
    
    # Verify
    total_frac = sum(Fraction(1, denoms[i]) for i in selected)
    assert total_frac < 1, f"Sum {total_frac} is not < 1"
    assert len(selected) > 0, "Empty selection"
    
    return selected
# EVOLVE-BLOCK-END
