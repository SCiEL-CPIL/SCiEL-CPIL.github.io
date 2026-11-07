"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import sys

def construct():
    """
    Construct a set of distinct integers in 1..256 such that:
    - Two subsets with different cardinalities have different sums.
    - Maximize the set size.
    
    Strategy:
    1. Use bitwise DP for subset sums per cardinality.
    2. Try multiple structured orders (decreasing, increasing, alternating, etc.).
    3. Use optimized collision checking by tracking only newly added bits.
    """
    
    max_val = 256
    max_size = 60
    
    def run_greedy(order):
        sums_by_k = [0] * (max_size + 1)
        sums_by_k[0] = 1  # bit 0 set
        
        result = []
        
        for x in order:
            # Compute new sums and added bits
            new_sums_by_k = [0] * (max_size + 1)
            added_bits = [0] * (max_size + 1)
            
            new_sums_by_k[0] = 1
            # added_bits[0] is 0 since we don't add to k=0
            
            valid = True
            for k in range(1, max_size + 1):
                shifted = sums_by_k[k - 1] << x
                # Mask out sums that are too large? 
                # Max possible sum is 256 * 60 = 15360. 
                # Python ints are arbitrary precision, so no mask needed unless we want to save space.
                # But bitwise ops on huge ints are slower.
                # Let's mask to reasonable size. Max sum with k elements <= 256*k.
                # But we don't know k in advance for the whole set.
                # However, 15360 bits is ~2KB. Very small.
                
                new_sums_by_k[k] = sums_by_k[k] | shifted
                added_bits[k] = shifted & ~sums_by_k[k]
            
            # Check disjointness
            # We need to ensure no intersection between any new_sums_by_k[i] and new_sums_by_k[j] for i != j.
            # Since old was valid, we only need to check intersections involving added_bits.
            # Intersection types:
            # 1. added_bits[i] & new_sums_by_k[j] (i != j)
            # 2. new_sums_by_k[i] & added_bits[j] (i != j) -- this is symmetric to 1 if we swap i,j?
            #    added_bits[j] & new_sums_by_k[i]. Yes.
            # So we just need to check:
            # For all i, j with i != j: added_bits[i] & new_sums_by_k[j] == 0
            
            # Optimization:
            # Iterate i where added_bits[i] != 0
            # Check against all j != i where new_sums_by_k[j] != 0
            
            for i in range(max_size + 1):
                if added_bits[i] == 0:
                    continue
                for j in range(max_size + 1):
                    if i == j:
                        continue
                    if new_sums_by_k[j] == 0:
                        continue
                    if added_bits[i] & new_sums_by_k[j]:
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

    # Define structured orders
    orders = []
    
    # 1. Decreasing
    orders.append(list(range(max_val, 0, -1)))
    
    # 2. Increasing
    orders.append(list(range(1, max_val + 1)))
    
    # 3. Alternating High/Low
    # 256, 1, 255, 2, 254, 3 ...
    alt = []
    low = 1
    high = max_val
    while low <= high:
        alt.append(high)
        if low != high:
            alt.append(low)
        low += 1
        high -= 1
    orders.append(alt)
    
    # 4. Center Out
    # 128, 1, 255, 2, 254, 3 ...
    center = []
    center_val = max_val // 2
    low = 1
    high = max_val
    # Start from center, then go out
    # Actually, let's do: center, center-1, center+1, center-2, center+2 ...
    c_orders = []
    diff = 0
    while True:
        left = center_val - diff
        right = center_val + diff
        if left >= 1:
            c_orders.append(left)
        if right <= max_val and right != left:
            c_orders.append(right)
        if left < 1 and right > max_val:
            break
        diff += 1
    orders.append(c_orders)
    
    # 5. "Even then Odd" decreasing
    evens = list(range(max_val, 0, -2))
    odds = list(range(max_val - 1, 0, -2))
    orders.append(evens + odds)
    
    # 6. "Odd then Even" decreasing
    orders.append(odds + evens)

    best_result = []
    for order in orders:
        res = run_greedy(order)
        if len(res) > len(best_result):
            best_result = res
            if len(best_result) >= max_size:
                break

    return best_result
# EVOLVE-BLOCK-END
