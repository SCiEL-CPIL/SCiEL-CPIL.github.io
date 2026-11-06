import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    # Precompute conflict matrix using bitmasks for O(1) compatibility checks
    n = 256
    # conflict_mask[i] is a bitmask where bit j is set if i and j conflict
    conflict_mask = [0] * (n + 1)
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            if (i * j) % (i + j) == 0:
                conflict_mask[i] |= (1 << j)
                conflict_mask[j] |= (1 << i)
    
    # Helper: Check if x can be added to set represented by bitmask S_mask
    def can_add(x, S_mask):
        # x is compatible with S if no element in S conflicts with x
        # conflict_mask[x] has bits set for all elements that conflict with x
        # If (conflict_mask[x] & S_mask) == 0, then x is compatible
        return (conflict_mask[x] & S_mask) == 0
    
    # Greedy initialization: start with all odds, then greedily add compatible elements
    # Sort candidates by number of conflicts (fewer conflicts = more likely to be addable)
    S = set(range(1, n + 1, 2))
    S_mask = 0
    for x in S:
        S_mask |= (1 << x)
    
    # Try to add each even number if compatible
    for x in range(2, n + 1, 2):
        if can_add(x, S_mask):
            S.add(x)
            S_mask |= (1 << x)
    
    best_S = set(S)
    best_S_mask = S_mask
    best_size = len(best_S)
    
    # Simulated Annealing with bitmask operations
    T = 1.0
    cooling = 0.9995
    min_T = 0.005
    
    # Precompute list of numbers for random selection
    all_nums = list(range(1, n + 1))
    
    # Precompute for each x, the list of numbers that conflict with x (for quick swap checking)
    # Actually, we can use conflict_mask directly
    
    while time.time() < deadline:
        if T < min_T:
            T = min_T
            
        # Generate a move
        r = random.random()
        
        # Try to Add
        if r < 0.45:
            # Pick a random number not in S
            x = random.choice(all_nums)
            if not (S_mask & (1 << x)):  # x not in S
                if can_add(x, S_mask):
                    S.add(x)
                    S_mask |= (1 << x)
                    if len(S) > best_size:
                        best_size = len(S)
                        best_S = set(S)
                        best_S_mask = S_mask
        # Try to Remove
        elif r < 0.75:
            if S:
                u = random.choice(list(S))
                # Accept removal with probability exp(-1/T)
                if random.random() < math.exp(-1.0 / T):
                    S.remove(u)
                    S_mask &= ~(1 << u)
        # Try to Swap
        else:
            if S:
                u = random.choice(list(S))
                x = random.choice(all_nums)
                if x != u and not (S_mask & (1 << x)):  # x not in S and x != u
                    # Check if x is compatible with S \ {u}
                    # S_mask without u: S_mask & ~(1 << u)
                    S_mask_without_u = S_mask & ~(1 << u)
                    if can_add(x, S_mask_without_u):
                        # Size remains same. Accept to explore.
                        # Accept with probability that depends on T to allow exploration
                        # At high T, accept more swaps; at low T, accept fewer
                        swap_prob = min(1.0, 0.3 + 0.7 * T)
                        if random.random() < swap_prob:
                            S.remove(u)
                            S.add(x)
                            S_mask = S_mask_without_u | (1 << x)
        
        # Cool down
        T *= cooling
        
    return list(best_S)
