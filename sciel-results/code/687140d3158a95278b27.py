import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin for 60s limit
    
    n = 128
    N = n  # Alias for clarity
    
    # Bitmask representation: bit i represents whether element (i+1) is in the set
    # Elements are 1..128, so bit 0 = element 1, bit 1 = element 2, ..., bit 127 = element 128
    
    def bitmask_to_set(mask):
        return {i+1 for i in range(n) if (mask >> i) & 1}
    
    def set_to_bitmask(s):
        mask = 0
        for x in s:
            mask |= (1 << (x-1))
        return mask
    
    # Precompute for each x, which pairs (a,b) with a < b, a+b = x
    # And which triples (a,b,c) would be violated if x is added
    
    # For fast validity checking, precompute conflict information
    # When adding x, conflicts arise from:
    # Case 1: x is a member. Need a,b in S such that (a+b in S or a+b==x) and a+x in S and b+x in S
    # Case 2: x is a sum. Need a,b,c in S distinct, with one of the sums equal to x
    
    # Precompute for each x: list of (a,b) pairs where a+b == x, a < b
    sum_pairs = {x: [] for x in range(2, 2*n+1)}
    for a in range(1, n+1):
        for b in range(a+1, n+1):
            s = a + b
            if s <= 2*n:
                sum_pairs[s].append((a, b))
    
    # Precompute for each x: list of (a,b) pairs where a+x <= n (for case 1)
    # Actually, we need a+x in S, so a+x must be in 1..n
    # For given x, valid a values are those where 1 <= a <= n and 1 <= a+x <= n
    # i.e., max(1, 1-x) <= a <= min(n, n-x)
    
    def is_valid_addition_bitmask(mask, x):
        """Check if adding x (1-indexed) to set represented by mask is valid."""
        # x is 1-indexed, bit position is x-1
        x_bit = x - 1
        
        # Case 1: x is a member (c = x)
        # Need a, b in S (distinct, a != b, a != x, b != x) such that:
        # - a+x in S (bit a+x-1 is set)
        # - b+x in S (bit b+x-1 is set)  
        # - a+b in S or a+b == x
        
        # Find all a in S where a+x is also in S
        # a ranges from 1 to n, a+x ranges from x+1 to x+n
        # For a+x to be in 1..n, we need a <= n-x
        # Also a != x (but x not in S yet, so a != x is automatic if a in S)
        
        # Collect candidates for case 1
        case1_candidates = []
        for a in range(1, min(n, n-x) + 1):  # a+x <= n means a <= n-x
            a_bit = a - 1
            if (mask >> a_bit) & 1:  # a in S
                ax_bit = (a + x) - 1
                if (mask >> ax_bit) & 1:  # a+x in S
                    case1_candidates.append(a)
        
        # Check pairs in case1_candidates
        if len(case1_candidates) >= 2:
            for i in range(len(case1_candidates)):
                a = case1_candidates[i]
                for j in range(i+1, len(case1_candidates)):
                    b = case1_candidates[j]
                    # a != b guaranteed by i < j and distinct values
                    # Check a+b in S or a+b == x
                    ab = a + b
                    if ab == x:
                        return False
                    ab_bit = ab - 1
                    if 1 <= ab <= n and (mask >> ab_bit) & 1:
                        return False
        
        # Case 2: x is a sum
        # Subcase 2a: x = a+b, need c in S, a+c in S, b+c in S, all distinct
        # For each pair (a,b) with a+b == x, check if there exists c in S distinct from a,b
        # such that a+c in S and b+c in S
        
        if x in sum_pairs:
            for (a, b) in sum_pairs[x]:
                # a and b are distinct (a < b from construction)
                # Need c in S, c != a, c != b, a+c in S, b+c in S
                # c ranges such that a+c <= n and b+c <= n, so c <= n - max(a,b) = n - b
                # Also c >= 1, c != a, c != b
                
                # For efficiency, iterate c
                for c in range(1, n - b + 1):  # b+c <= n means c <= n-b
                    if c == a or c == b:
                        continue
                    c_bit = c - 1
                    if not ((mask >> c_bit) & 1):  # c not in S
                        continue
                    ac_bit = (a + c) - 1
                    bc_bit = (b + c) - 1
                    if (mask >> ac_bit) & 1 and (mask >> bc_bit) & 1:
                        return False
        
        return True
    
    # Initialize with upper half
    initial_set = set(range(n//2 + 1, n + 1))
    current_mask = set_to_bitmask(initial_set)
    best_mask = current_mask
    best_size = len(initial_set)
    
    all_elements = set(range(1, n + 1))
    
    attempts = 0
    max_attempts = 500000
    
    while time.time() < deadline and attempts < max_attempts:
        attempts += 1
        
        # Try to add a random missing element
        current_set = bitmask_to_set(current_mask)
        missing = all_elements - current_set
        
        if missing:
            missing_list = list(missing)
            random.shuffle(missing_list)
            added = False
            for x in missing_list:
                if is_valid_addition_bitmask(current_mask, x):
                    current_mask |= (1 << (x-1))
                    if current_mask.bit_count() > best_size:
                        best_size = current_mask.bit_count()
                        best_mask = current_mask
                    added = True
                    break
            
            if added:
                continue
        
        # If no element could be added, remove a random element
        if current_mask:
            current_set = bitmask_to_set(current_mask)
            if current_set:
                to_remove = random.choice(list(current_set))
                current_mask &= ~(1 << (to_remove-1))
        
        # Check time periodically to avoid long loops
        if attempts % 1000 == 0 and time.time() > deadline:
            break
    
    # Convert best_mask back to list of ints
    result = [i+1 for i in range(n) if (best_mask >> i) & 1]
    
    # Ensure all are Python ints
    result = [int(x) for x in result]
    
    return result
