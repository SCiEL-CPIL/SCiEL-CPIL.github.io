import time
import random
import numpy as np

def construct():
    n = 512
    start_time = time.time()
    time_limit = 55.0  # 5s buffer

    # 1. Precompute conflicts
    # conflicts[x] is a list of masks. Each mask has 3 bits set.
    # If (S & mask) == mask, then x cannot be added.
    conflicts = [[] for _ in range(n)]
    
    # Generate all 4-APs
    for d in range(1, n // 3 + 1):
        for start in range(1, n - 3 * d + 1):
            a = start
            b = start + d
            c = start + 2 * d
            e = start + 3 * d
            
            # Convert to 0-based indices
            a_idx = a - 1
            b_idx = b - 1
            c_idx = c - 1
            e_idx = e - 1
            
            # For each element, store the mask of the other three
            # a is in AP with b, c, e
            conflicts[a_idx].append((1 << b_idx) | (1 << c_idx) | (1 << e_idx))
            # b is in AP with a, c, e
            conflicts[b_idx].append((1 << a_idx) | (1 << c_idx) | (1 << e_idx))
            # c is in AP with a, b, e
            conflicts[c_idx].append((1 << a_idx) | (1 << b_idx) | (1 << e_idx))
            # e is in AP with a, b, c
            conflicts[e_idx].append((1 << a_idx) | (1 << b_idx) | (1 << c_idx))

    # Helper: Check if x can be added to S
    def can_add(S, x):
        for mask in conflicts[x]:
            if (S & mask) == mask:
                return False
        return True

    # Helper: Greedy fill
    def greedy_fill(S, order):
        S = S  # Bitmask is immutable, so we just update the local var
        for x in order:
            if not (S & (1 << x)):
                if can_add(S, x):
                    S |= (1 << x)
        return S

    # Helper: Popcount
    def popcount(S):
        return bin(S).count('1')

    # Initial Constructions
    best_S = 0
    best_size = 0
    
    # 1. Sorted
    S = greedy_fill(0, list(range(n)))
    if popcount(S) > best_size:
        best_S = S
        best_size = popcount(S)
        
    # 2. Reversed
    S = greedy_fill(0, list(range(n-1, -1, -1)))
    if popcount(S) > best_size:
        best_S = S
        best_size = popcount(S)

    # 3. Random Seeds
    random.seed(42)
    for _ in range(20):
        order = list(range(n))
        random.shuffle(order)
        S = greedy_fill(0, order)
        if popcount(S) > best_size:
            best_S = S
            best_size = popcount(S)

    # Local Search / Random Restart
    # We will perform "Kick and Refill"
    # Kick: Remove k random elements from best_S
    # Refill: Greedy fill the rest
    
    current_best_S = best_S
    current_best_size = best_size
    
    while time.time() - start_time < time_limit:
        # Choose a kick size
        # Larger kicks allow escaping deeper local minima but are riskier
        # Start with small kicks, increase if stuck
        # For simplicity, let's just vary it randomly between 1 and 5
        k = random.randint(1, 5)
        
        # Create a copy of the best set
        S = current_best_S
        
        # Remove k random elements
        # Get list of set bits
        # Efficient way to get set bits from a big int in Python?
        # S.bit_length() is small (512).
        # We can iterate through 0..n-1 and check bit.
        # Or use a helper to extract indices.
        
        # Since n=512, iterating 512 times is fast.
        set_bits = [i for i in range(n) if (S >> i) & 1]
        
        if len(set_bits) <= k:
            # Can't kick effectively, skip or restart
            continue
            
        to_remove = random.sample(set_bits, k)
        
        for x in to_remove:
            S &= ~(1 << x)
            
        # Refill
        order = list(range(n))
        random.shuffle(order)
        S = greedy_fill(S, order)
        
        new_size = popcount(S)
        
        if new_size > current_best_size:
            current_best_S = S
            current_best_size = new_size
            # If we improved, we can reset the "stuck" counter or just continue
            # We don't have a stuck counter, we just keep trying.

    # Convert best_S (bitmask) to sorted list
    result = [i + 1 for i in range(n) if (current_best_S >> i) & 1]
    return result
