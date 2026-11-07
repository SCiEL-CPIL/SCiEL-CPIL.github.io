"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    deadline = time.time() + 55  # Safety margin for 60s limit
    n = 128
    
    # Bitmask representation for fast operations
    # bit i represents whether (i+1) is in the set
    # Using Python int as arbitrary precision bitmask
    
    def is_valid_addition(x, mask):
        """Check if adding x to the set represented by mask keeps it valid.
        x is 1-indexed. mask has bit (x-1) set if x is in the set.
        """
        # Case 1: x is a member (a=x)
        # Need b,c in S distinct, b!=x, c!=x
        # x+b in S, x+c in S, b+c in S
        # Find all b in S (b != x) such that x+b is in S
        A_x = []
        for b in range(1, n + 1):
            if b == x:
                continue
            if (mask >> (b - 1)) & 1 and (mask >> (x + b - 1)) & 1:
                A_x.append(b)
        
        if len(A_x) >= 2:
            for i in range(len(A_x)):
                for j in range(i + 1, len(A_x)):
                    if (mask >> (A_x[i] + A_x[j] - 1)) & 1:
                        return False
        
        # Case 2: x is a sum (a+b=x)
        # Need a,b,c in S distinct
        # a+b=x, a+c in S, b+c in S
        for a in range(1, x):
            b = x - a
            if b <= a:
                continue
            if ((mask >> (a - 1)) & 1) and ((mask >> (b - 1)) & 1):
                # Found pair (a,b) with a+b=x
                for c in range(1, n + 1):
                    if c == a or c == b:
                        continue
                    if ((mask >> (c - 1)) & 1) and ((mask >> (a + c - 1)) & 1) and ((mask >> (b + c - 1)) & 1):
                        return False
        
        return True

    # Initial Set: Upper Half [65..128]
    # Create mask for this set
    S_mask = 0
    for i in range(65, n + 1):
        S_mask |= (1 << (i - 1))
    
    best_size = n - 64  # 64 elements
    best_mask = S_mask
    
    # Greedy Pass 1: Deterministic (1 to 64)
    for x in range(1, 65):
        if time.time() > deadline:
            break
        if is_valid_addition(x, S_mask):
            S_mask |= (1 << (x - 1))
            best_size += 1
            best_mask = S_mask
    
    # Random restart strategy with randomized greedy order
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        # Randomize the greedy order for the lower half
        lower_half = list(range(1, 65))
        random.shuffle(lower_half)
        
        S_try = 0
        for i in range(65, n + 1):
            S_try |= (1 << (i - 1))
        
        current_size = n - 64
        
        for x in lower_half:
            if time.time() > deadline:
                break
            if is_valid_addition(x, S_try):
                S_try |= (1 << (x - 1))
                current_size += 1
        
        # Local Search: Try to add any remaining number
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for x in range(1, n + 1):
                if not ((S_try >> (x - 1)) & 1):
                    if is_valid_addition(x, S_try):
                        S_try |= (1 << (x - 1))
                        current_size += 1
                        improved = True
                        break
        
        if current_size > best_size:
            best_size = current_size
            best_mask = S_try
            
        # If we have time, try to swap to increase size?
        # Swap: remove y, add z. Size stays same.
        # To increase size, we need a sequence of swaps that ends with a net add.
        # This is complex. The "Add if possible" loop above should find most improvements.
        
        # Additional: Try removing one element and adding two?
        # This is a 1-2 exchange. Let's try it if we have time.
        if time.time() < deadline:
            # Try 1-2 exchanges: remove one element y, then try to add two elements
            # This can increase size by 1
            found_improvement = True
            while found_improvement and time.time() < deadline:
                found_improvement = False
                # Try removing each element y
                for y in range(1, n + 1):
                    if not ((S_try >> (y - 1)) & 1):
                        continue  # y not in set
                    # Try removing y
                    S_removed = S_try & ~(1 << (y - 1))
                    size_removed = current_size - 1
                    # Now try to add elements in random order
                    candidates = list(range(1, n + 1))
                    random.shuffle(candidates)
                    added = 0
                    S_new = S_removed
                    for x in candidates:
                        if time.time() > deadline:
                            break
                        if x == y:
                            continue
                        if not ((S_new >> (x - 1)) & 1):
                            if is_valid_addition(x, S_new):
                                S_new |= (1 << (x - 1))
                                added += 1
                    # If we added more than 1, we have a net gain
                    if added > 1:
                        S_try = S_new
                        current_size = size_removed + added
                        found_improvement = True
                        break
                    # Even if added == 1, we might be in a different local optimum
                    # But size is same, so no gain. Skip for now.
            
            # Update best if improved
            if current_size > best_size:
                best_size = current_size
                best_mask = S_try
    
    # Convert best_mask to list of integers
    result = []
    for i in range(1, n + 1):
        if (best_mask >> (i - 1)) & 1:
            result.append(i)
    
    return result

# EVOLVE-BLOCK-END
