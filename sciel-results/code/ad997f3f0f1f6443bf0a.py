import time
import random

_DEADLINE = 85.0

def _zero_sum_size_ge_2(zs):
    """
    Check if 0 can be formed as a sum of a subset of zs with size >= 2.
    zs are integers (can be negative, zero).
    Uses sorted-bounds pruning to skip most k values.
    """
    n = len(zs)
    if n < 2:
        return False
    
    # Separate positive and negative values
    pos = sorted([z for z in zs if z > 0])
    neg = sorted([-z for z in zs if z < 0])  # neg is sorted ascending (smaller abs first)
    zeros = sum(1 for z in zs if z == 0)
    
    # If we have at least 2 zeros, any 2 zeros sum to 0
    if zeros >= 2:
        return True
    
    # We need subset sum 0 with size >= 2
    # Total positive sum
    total_pos = sum(pos)
    total_neg = sum(neg)
    
    # Quick bound check: if total_pos != total_neg, we still might form 0
    # but let's do DP with bounds
    
    # Maximum possible sum range: -total_neg to total_pos
    # We want to know if sum 0 is achievable with size >= 2
    
    # Use DP with bitsets, tracking min/max achievable sum for each size
    # dp[k] = bitset where bit i means sum (i - offset) is achievable
    offset = total_neg  # shift so that -total_neg maps to 0
    max_bit = total_pos + total_neg  # max sum is total_pos, so bit index goes up to total_pos + offset
    
    # dp[k] is a Python int used as bitset
    dp = [0] * (n + 1)
    dp[0] = 1 << offset  # sum 0 with size 0
    
    # Track min and max achievable sum for each size for pruning
    min_sum = [0] * (n + 1)  # min_sum[k] = minimum sum achievable with size k
    max_sum = [0] * (n + 1)  # max_sum[k] = maximum sum achievable with size k
    min_sum[0] = 0
    max_sum[0] = 0
    reachable = [True] + [False] * n
    
    for z in zs:
        if z == 0:
            continue  # zeros handled separately
        for k in range(n, 0, -1):
            if not reachable[k-1]:
                continue
            if z > 0:
                shifted = dp[k-1] << z
                dp[k] |= shifted
                min_sum[k] = min(min_sum[k], min_sum[k-1] + z) if reachable[k] else min_sum[k-1] + z
                max_sum[k] = max(max_sum[k], max_sum[k-1] + z) if reachable[k] else max_sum[k-1] + z
            else:
                shifted = dp[k-1] >> (-z)
                dp[k] |= shifted
                min_sum[k] = min(min_sum[k], min_sum[k-1] + z) if reachable[k] else min_sum[k-1] + z
                max_sum[k] = max(max_sum[k], max_sum[k-1] + z) if reachable[k] else max_sum[k-1] + z
            reachable[k] = True
    
    # Check if any k >= 2 has bit offset set (sum 0)
    for k in range(2, n + 1):
        if reachable[k] and dp[k] & (1 << offset):
            return True
    return False

def _can_form_sum(target, vals):
    """
    Check if target can be formed as a sum of a subset of vals with size >= 1.
    vals are integers (can be negative).
    Returns True if yes.
    """
    if not vals:
        return False
        
    n = len(vals)
    # Sum of absolute values for bounds
    max_sum = sum(abs(v) for v in vals)
    if abs(target) > max_sum:
        return False
        
    offset = max_sum
    # We need subset size >= 1
    # dp[k] bitset
    dp = [0] * (n + 1)
    dp[0] = 1 << offset
    
    for v in vals:
        # Iterate k downwards
        for k in range(n, 0, -1):
            if dp[k-1] == 0:
                continue
            if v >= 0:
                shifted = dp[k-1] << v
            else:
                shifted = dp[k-1] >> (-v)
            dp[k] |= shifted
            
    # Check if any k >= 1 has bit (offset + target) set
    bit_idx = offset + target
    for k in range(1, n + 1):
        if dp[k] & (1 << bit_idx):
            return True
    return False

def _is_mean_of_subset(C, i):
    """
    Check if C[i] is the mean of any subset of at least 2 OTHER elements in C.
    """
    a = C[i]
    others = C[:i] + C[i+1:]
    n = len(others)
    if n < 2:
        return False
    
    zs = [y - a for y in others]
    
    # We need sum 0 with size >= 2
    return _zero_sum_size_ge_2(zs)

def _is_valid_incremental(selected, x):
    """
    Check if x can be added to selected without violating constraints.
    selected is a list of ints.
    x is an int not in selected.
    Returns True if valid.
    """
    # Check 1: x is mean of subset of selected (size >= 2)
    if len(selected) >= 2:
        if _zero_sum_size_ge_2([y - x for y in selected]):
            return False

    # Check 2: any y in selected is mean of subset of selected U {x} containing x
    # 2y = sum(Z) where x in Z, Z subset selected U {x}, Z != {y}
    # y - x + sum(Z' - y) = 0 where Z' subset selected \ {y, x}
    # sum(Z' - y) = y - x
    # We need subset sum (y-x) from {z - y : z in selected, z != y} with size >= 1
    # (Note: x != y since distinct, so z != x is automatic when z in selected)
    
    for y in selected:
        target = y - x
        if target == 0:
            continue
        vals = [z - y for z in selected if z != y]
        if not vals:
            continue
        if _can_form_sum(target, vals):
            return False
            
    return True

def _full_check(C):
    """
    Full verification of a candidate set C.
    Returns True if C is valid.
    """
    for i in range(len(C)):
        if _is_mean_of_subset(C, i):
            return False
    return True

def construct():
    start_time = time.time()
    best_set = []
    
    candidates = list(range(1, 513))
    
    # Define several orders
    orders = []
    
    # 1. Spread (High-Low) - prioritize extremes to avoid middle averages
    spread = []
    lo, hi = 1, 512
    while lo <= hi:
        spread.append(hi)
        if lo != hi:
            spread.append(lo)
        lo += 1
        hi -= 1
    orders.append(spread)
    
    # 2. Spread (Low-High)
    orders.append(spread[::-1])
    
    # 3. Boundary focused - start with very small and very large
    boundary = []
    for i in range(256):
        boundary.append(1 + i)
        boundary.append(512 - i)
    orders.append(boundary)
    
    # 4. Power-of-2 style: 1,2,4,8,... then fill
    powers = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    rest = [x for x in candidates if x not in set(powers)]
    orders.append(powers + rest)
    
    # 5. Randoms with seed
    rng = random.Random(42)
    for _ in range(30):
        r = candidates[:]
        rng.shuffle(r)
        orders.append(r)
    
    # 6. Reverse random
    rng2 = random.Random(999)
    for _ in range(30):
        r = candidates[:]
        rng2.shuffle(r)
        orders.append(r)
    
    for order in orders:
        if time.time() - start_time > _DEADLINE - 5:
            break
            
        selected = []
        sel_set = set()
        
        for x in order:
            if time.time() - start_time > _DEADLINE - 5:
                break
                
            if len(selected) >= 48:
                break
                
            if x in sel_set:
                continue
                
            # Fast pair check (optimization)
            # 1. x mean of pair in selected
            valid = True
            for i in range(len(selected)):
                for j in range(i+1, len(selected)):
                    if selected[i] + selected[j] == 2 * x:
                        valid = False
                        break
                if not valid:
                    break
            
            # 2. selected[i] mean of pair (x, selected[j])
            if valid:
                for a in selected:
                    y = 2 * a - x
                    if y in sel_set and y != a:
                        valid = False
                        break
                        
            if not valid:
                continue
                
            # Full incremental check
            if _is_valid_incremental(selected, x):
                selected.append(x)
                sel_set.add(x)
                
        if len(selected) > len(best_set):
            best_set = selected[:]
            
    # Final verification
    if best_set:
        # Double check best_set
        C = best_set[:]
        for i in range(len(C)):
            if _is_mean_of_subset(C, i):
                # Fallback
                return [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
        return best_set
    else:
        return [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
