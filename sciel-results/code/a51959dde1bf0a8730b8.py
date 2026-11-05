import time
import sys

_DEADLINE = 55.0  # seconds, safety margin before 60s limit

def _check_violation_bitset(a, others):
    """
    Check if 'a' is the arithmetic mean of any subset of 'others' with size >= 2.
    others: list of integers, does not contain 'a'.
    Returns True if violation exists.
    """
    n = len(others)
    if n < 2:
        return False
    
    # 1. Check k=2 (pairs)
    # a = (x + y) / 2 => x + y = 2a
    target2 = 2 * a
    seen = set()
    for x in others:
        if (target2 - x) in seen:
            return True
        seen.add(x)
    
    # 2. Check k >= 3 using Bitset DP
    # Transform: z_i = x_i - a. We need subset sum 0.
    zs = [x - a for x in others]
    
    # Max absolute sum is sum(|z_i|). 
    # To be safe, we can bound the bitset size.
    # Max sum magnitude: 512 * 48 = 24576.
    # We use an offset of 25000 to handle negatives.
    OFFSET = 25000
    MAX_BITS = 50000 # Range -25000 to +25000
    
    # dp[k] is an integer bitset. Bit i is set if sum (i - OFFSET) is achievable with k elements.
    # Initialize dp[0] with sum 0.
    dp = [0] * (n + 1)
    dp[0] = 1 << OFFSET
    
    current_max_k = 0
    for z in zs:
        # Update dp in reverse order to avoid using an element multiple times
        # We only need to go up to current_max_k + 1
        for k in range(current_max_k, 0, -1):
            if dp[k-1]:
                if z >= 0:
                    dp[k] |= (dp[k-1] << z)
                else:
                    dp[k] |= (dp[k-1] >> -z)
        
        # Check if 0 is in dp[k] for any k >= 3?
        # We can check after each element addition to potentially exit early?
        # No, the subset must be formed from the processed elements.
        # If we find 0 in dp[k] for k>=3, we can return True.
        # However, we must ensure the subset size is exactly k.
        
        # Optimization: Check only k=3 and k=4 frequently?
        # Actually, checking all k up to current_max_k+1 is cheap if we just check the bit at OFFSET.
        
        # Let's check for violation at the end of each z iteration?
        # It might be faster to just finish the DP and then check.
        # But early exit is better.
        
        # Check k=3
        if current_max_k >= 2: # After adding at least 3 elements? No, k is size.
            # We need to check if bit OFFSET is set in dp[k] for k>=3
            # We can just check after the loop for this z?
            pass
        
        current_max_k += 1
        
        # Early exit check:
        # Check if 0 is achievable with exactly k elements for any k in [3, current_max_k]
        # This is O(N) per element, so O(N^2) total.
        # Bit check is O(1).
        for k in range(3, current_max_k + 1):
            if dp[k] & (1 << OFFSET):
                return True

    return False

def _full_check(candidate):
    """Check all elements for violations."""
    for i in range(len(candidate)):
        a = candidate[i]
        others = candidate[:i] + candidate[i+1:]
        if _check_violation_bitset(a, others):
            return True
    return False

def construct():
    start_time = time.time()
    
    # Candidate pool
    # We want to maximize cardinality.
    # Strategy: Try multiple orders.
    
    # Order 1: Increasing
    order1 = list(range(1, 513))
    
    # Order 2: Decreasing
    order2 = list(range(512, 0, -1))
    
    # Order 3: "Sparse" first (Powers of 2, then others)
    powers_of_2 = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    rest = [x for x in range(1, 513) if x not in powers_of_2]
    order3 = powers_of_2 + rest
    
    # Order 4: Odds then Evens
    odds = list(range(1, 513, 2))
    evens = list(range(2, 513, 2))
    order4 = odds + evens
    
    # Order 5: Multiplicative structure (2^i * 3^j)
    mult_struct = set()
    for i in range(10):
        for j in range(10):
            val = (2 ** i) * (3 ** j)
            if 1 <= val <= 512:
                mult_struct.add(val)
    mult_list = sorted(mult_struct)
    rest_mult = [x for x in range(1, 513) if x not in mult_struct]
    order5 = mult_list + rest_mult
    
    orders = [order1, order2, order3, order4, order5]
    
    best_set = []
    
    for order in orders:
        if time.time() - start_time > _DEADLINE - 15:
            break
            
        selected = []
        for x in order:
            if time.time() - start_time > _DEADLINE - 15:
                break
            
            candidate = selected + [x]
            
            # Quick check: Is x the mean of any pair in selected?
            # This is O(N)
            target2 = 2 * x
            seen = set()
            pair_violation = False
            for y in selected:
                if (target2 - y) in seen:
                    pair_violation = True
                    break
                seen.add(y)
            
            if pair_violation:
                continue
            
            # Full check for the new element x against selected
            # x is the last element, index len(selected)
            if _check_violation_bitset(x, selected):
                continue
            
            # Check if adding x creates a violation for any EXISTING element
            # This is the expensive part.
            # We can spot check or do full check.
            # For N < 20, full check is fast.
            # For N >= 20, we might need to be selective.
            
            # Let's do a full check for all elements if N is small, 
            # or check a subset of existing elements if N is large.
            # However, to ensure correctness, we MUST verify the whole set.
            # But we can do it incrementally?
            # No, adding x can create a mean violation for an old element a 
            # only if x is part of the subset S that averages to a.
            # So we only need to check if a is the mean of a subset S such that x in S.
            # This is equivalent to checking if (sum(S\{x}) - (|S|-1)a) = x - a is achievable?
            # Basically, for each existing a, check if there is a subset of (selected \ {a}) 
            # that sums to k*a - x for some k?
            # This is still complex.
            
            # Given the deadline, let's just run _full_check on the candidate.
            # _full_check is O(N^2 * bitset_ops).
            # With N=48, it should be fast enough in Python if optimized.
            
            if _full_check(candidate):
                continue
            
            selected.append(x)
            
            if len(selected) >= 48:
                break
        
        if len(selected) > len(best_set):
            best_set = selected[:]
            
    # Final verification
    if best_set:
        if _full_check(best_set):
            # Fallback
            return [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
        else:
            return best_set
    else:
        return [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
