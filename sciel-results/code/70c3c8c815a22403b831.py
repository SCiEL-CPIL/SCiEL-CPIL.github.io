import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    n = 256
    conflict_mask = [0] * (n + 1)
    
    for i in range(1, n + 1):
        mask = 0
        for j in range(1, n + 1):
            if i != j and (i * j) % (i + j) == 0:
                mask |= (1 << (j - 1))
        conflict_mask[i] = mask

    def size(mask):
        return mask.bit_count()
        
    def in_set(x, mask):
        return (mask & (1 << (x - 1))) != 0
        
    def remove(x, mask):
        return mask & ~(1 << (x - 1))
        
    def add(x, mask):
        return mask | (1 << (x - 1))
        
    def get_bits(mask):
        bits = []
        temp = mask
        while temp:
            lsb = temp & -temp
            idx = (lsb.bit_length() - 1) + 1
            bits.append(idx)
            temp ^= lsb
        return bits

    # Initial solution: All odds
    S_mask = 0
    for i in range(1, n + 1, 2):
        S_mask |= (1 << (i - 1))
        
    best_mask = S_mask
    best_size = size(S_mask)
    
    all_nums = list(range(1, n + 1))
    
    def local_polish(mask):
        nonlocal best_size, best_mask
        improved = True
        while improved:
            improved = False
            for x in all_nums:
                if not in_set(x, mask) and (conflict_mask[x] & mask) == 0:
                    mask = add(x, mask)
                    improved = True
                    if size(mask) > best_size:
                        best_size = size(mask)
                        best_mask = mask
        return mask

    S_mask = local_polish(S_mask)
    
    T = 10.0
    cooling = 0.9995
    min_T = 0.01
    
    while time.time() < deadline:
        if T < min_T:
            T = 10.0
            S_mask = best_mask
            # Perturb
            for _ in range(5):
                if S_mask:
                    bits = get_bits(S_mask)
                    if bits:
                        x = random.choice(bits)
                        S_mask = remove(x, S_mask)
            S_mask = local_polish(S_mask)
            continue

        r = random.random()
        new_mask = S_mask
        delta = 0
        
        if r < 0.4:
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
            if S_mask:
                bits = get_bits(S_mask)
                if bits:
                    x = random.choice(bits)
                    new_mask = remove(x, S_mask)
                    delta = -1
        else:
            if S_mask:
                bits_in = get_bits(S_mask)
                if bits_in:
                    x = random.choice(bits_in)
                    S_minus_x = remove(x, S_mask)
                    candidates = []
                    for y in all_nums:
                        if not in_set(y, S_mask):
                            if (conflict_mask[y] & S_minus_x) == 0:
                                candidates.append(y)
                    if candidates:
                        y = random.choice(candidates)
                        new_mask = add(y, S_minus_x)
                        delta = 0
                        
        if delta > 0:
            S_mask = new_mask
            if size(S_mask) > best_size:
                best_size = size(S_mask)
                best_mask = S_mask
                S_mask = local_polish(S_mask)
        elif delta < 0:
            if random.random() < math.exp(delta / T):
                S_mask = new_mask
        else:
            if random.random() < 0.5:
                S_mask = new_mask
        
        T *= cooling
        
    S_mask = local_polish(best_mask)
    
    return [i for i in range(1, n + 1) if in_set(i, best_mask)]
