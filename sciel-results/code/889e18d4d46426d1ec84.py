import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    n = 256
    # conflict_mask[i] is a bitmask where bit j is 1 if i and j conflict
    # i, j in 1..256. We use 0-indexed bits for 1..256, so bit i-1 corresponds to number i.
    conflict_mask = [0] * (n + 1)
    
    for i in range(1, n + 1):
        mask = 0
        for j in range(1, n + 1):
            if i != j and (i * j) % (i + j) == 0:
                mask |= (1 << (j - 1))
        conflict_mask[i] = mask
        
    # Initial solution: All odd numbers
    # Bitmask representation
    S_mask = 0
    for i in range(1, n + 1, 2):
        S_mask |= (1 << (i - 1))
        
    best_mask = S_mask
    best_size = bin(S_mask).count('1')
    
    T = 10.0
    cooling = 0.9995
    min_T = 0.01
    
    # Helper to get size
    def size(mask):
        return bin(mask).count('1')
        
    # Helper to check if x is in mask
    def in_set(x, mask):
        return (mask & (1 << (x - 1))) != 0
        
    # Helper to remove x from mask
    def remove(x, mask):
        return mask & ~(1 << (x - 1))
        
    # Helper to add x to mask
    def add(x, mask):
        return mask | (1 << (x - 1))
        
    all_nums = list(range(1, n + 1))
    
    while time.time() < deadline:
        if T < min_T:
            # Restart with new T and maybe a perturbed state
            T = 10.0
            # Perturb best solution: remove 5 random elements
            S_mask = best_mask
            for _ in range(5):
                if S_mask:
                    # Pick random set bit
                    bits = []
                    temp = S_mask
                    while temp:
                        lsb = temp & -temp
                        idx = (lsb.bit_length() - 1) + 1
                        bits.append(idx)
                        temp ^= lsb
                    if bits:
                        x = random.choice(bits)
                        S_mask = remove(x, S_mask)
                        # Greedily re-add some
                        for y in all_nums:
                            if not in_set(y, S_mask) and (conflict_mask[y] & S_mask) == 0:
                                S_mask = add(y, S_mask)
            best_size = size(best_mask)
            # Ensure current S is not worse than best
            if size(S_mask) < best_size:
                 S_mask = best_mask
            continue

        # Generate move
        r = random.random()
        new_mask = S_mask
        delta = 0
        
        if r < 0.4:
            # Add
            # Find all x not in S_mask that are compatible
            # Optimization: Instead of scanning all 256, we can just pick a random x and check
            # But scanning is fast enough for 256.
            candidates = []
            for x in all_nums:
                if not in_set(x, S_mask):
                    if (conflict_mask[x] & S_mask) == 0:
                        candidates.append(x)
            if candidates:
                x = random.choice(candidates)
                new_mask = add(x, S_mask)
                delta = 1
        elif r < 0.7:
            # Remove
            if S_mask:
                # Pick random element in S_mask
                bits = []
                temp = S_mask
                while temp:
                    lsb = temp & -temp
                    idx = (lsb.bit_length() - 1) + 1
                    bits.append(idx)
                    temp ^= lsb
                if bits:
                    x = random.choice(bits)
                    new_mask = remove(x, S_mask)
                    delta = -1
        else:
            # Swap
            if S_mask:
                # Pick x in S_mask
                bits_in = []
                temp = S_mask
                while temp:
                    lsb = temp & -temp
                    idx = (lsb.bit_length() - 1) + 1
                    bits_in.append(idx)
                    temp ^= lsb
                
                if bits_in:
                    x = random.choice(bits_in)
                    # Find y not in S_mask compatible with S_mask \ {x}
                    S_minus_x = remove(x, S_mask)
                    candidates = []
                    for y in all_nums:
                        if not in_set(y, S_mask):
                            if (conflict_mask[y] & S_minus_x) == 0:
                                candidates.append(y)
                    if candidates:
                        y = random.choice(candidates)
                        new_mask = add(y, S_minus_x)
                        delta = 0 # Size unchanged, but structure changed
                        
        # Acceptance
        if delta > 0:
            S_mask = new_mask
            if size(S_mask) > best_size:
                best_size = size(S_mask)
                best_mask = S_mask
        elif delta < 0:
            # Prob exp(delta / T)
            if random.random() < math.exp(delta / T):
                S_mask = new_mask
        else:
            # Delta 0, accept with some prob to explore
            # If we accept all delta 0, we might cycle. 
            # Let's accept with prob 0.5 or based on T.
            # Actually, for delta 0, it's neutral. We can accept with prob 0.5 to allow mixing.
            if random.random() < 0.5:
                S_mask = new_mask
        
        T *= cooling
        
    return [i for i in range(1, n + 1) if in_set(i, best_mask)]
