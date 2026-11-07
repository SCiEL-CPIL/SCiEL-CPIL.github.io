"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools

def construct():
    """
    Construct a set of distinct integers in 1..256 such that:
    - Two subsets with different cardinalities have different sums.
    - Maximize the set size.
    
    Strategy:
    1. Use a bitwise DP to track achievable sums for each cardinality.
    2. Greedily add numbers from structured orders (geometric, powers of 2, etc.)
    3. The key constraint: for any k1 != k2, the set of sums achievable with cardinality k1
       must be disjoint from the set of sums achievable with cardinality k2.
    
    We use bitsets: sums_by_k[k] is a bitset where bit s is 1 if sum s is achievable with cardinality k.
    When adding a new number x, we check:
      For each existing k, the new sums at cardinality k+1 are (sums_by_k[k] << x).
      We need to ensure that for all k1 != k2, the new bitsets are disjoint.
    """
    
    max_val = 256
    max_size = 60
    
    def run_greedy(order):
        sums_by_k = [0] * (max_size + 1)
        sums_by_k[0] = 1  # bit 0 set
        
        result = []
        
        for x in order:
            if x < 1 or x > max_val:
                continue
            if x in result:
                continue
            
            # Check if we can add x
            new_sums_by_k = [0] * (max_size + 1)
            new_sums_by_k[0] = 1
            
            valid = True
            # Compute new sums and check validity
            for k in range(1, max_size + 1):
                new_sums_by_k[k] = sums_by_k[k] | (sums_by_k[k - 1] << x)
            
            # Check disjointness
            for k1 in range(max_size + 1):
                if new_sums_by_k[k1] == 0:
                    continue
                for k2 in range(k1 + 1, max_size + 1):
                    if new_sums_by_k[k2] == 0:
                        continue
                    if new_sums_by_k[k1] & new_sums_by_k[k2]:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                result.append(x)
                sums_by_k = new_sums_by_k
                if len(result) >= max_size:
                    break
        
        return result

    # Generate multiple structured orders to try
    orders = []
    
    # Order 1: Decreasing
    orders.append(list(range(max_val, 0, -1)))
    
    # Order 2: Increasing
    orders.append(list(range(1, max_val + 1)))
    
    # Order 3: Powers of 2 first, then remaining in decreasing order
    powers_of_2 = [2**i for i in range(0, 8) if 2**i <= max_val]
    remaining = [x for x in range(max_val, 0, -1) if x not in set(powers_of_2)]
    orders.append(powers_of_2 + remaining)
    
    # Order 4: Geometric progression with ratio ~1.5, then fill in
    geo = []
    val = 1
    while val <= max_val:
        geo.append(val)
        val = int(val * 1.5) + 1
    geo = [x for x in geo if x <= max_val]
    remaining = [x for x in range(max_val, 0, -1) if x not in set(geo)]
    orders.append(geo + remaining)
    
    # Order 5: Large values first (top half), then small values
    top_half = list(range(max_val, max_val // 2, -1))
    bottom_half = list(range(max_val // 2, 0, -1))
    orders.append(top_half + bottom_half)
    
    # Order 6: Alternate large and small
    large = list(range(max_val, max_val // 2, -1))
    small = list(range(max_val // 2, 0, -1))
    interleaved = []
    li, si = 0, 0
    while li < len(large) or si < len(small):
        if li < len(large):
            interleaved.append(large[li])
            li += 1
        if si < len(small):
            interleaved.append(small[si])
            si += 1
    orders.append(interleaved)
    
    # Try all orders and pick the best
    best_result = []
    for order in orders:
        result = run_greedy(order)
        if len(result) > len(best_result):
            best_result = result
        if len(best_result) >= max_size:
            break
    
    return best_result
# EVOLVE-BLOCK-END
