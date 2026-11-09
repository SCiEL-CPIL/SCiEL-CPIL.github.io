import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    N = 32
    MAX_VAL = 512
    MAX_D = 255  # max possible d: 1 + 2*255 = 511 <= 512

    def time_ok():
        return time.time() < deadline

    # Bitmask approach: bit i represents element (i+1) in the set
    # So element value v corresponds to bit (v-1)
    
    def score_mask(mask):
        """Count distinct d > 0 such that a, a+d, a+2d all in mask."""
        count = 0
        for d in range(1, MAX_D + 1):
            if (mask & (mask >> d) & (mask >> (2 * d))) != 0:
                count += 1
        return count

    def list_to_mask(lst):
        m = 0
        for x in lst:
            m |= (1 << (x - 1))
        return m

    def mask_to_list(mask):
        return [i + 1 for i in range(MAX_VAL) if (mask >> i) & 1]

    def mask_popcount(mask):
        return bin(mask).count('1')

    # Fast delta: when replacing old_val with new_val, only check d's that involve old_val or new_val
    def delta_replace(mask, old_val, new_val):
        """Return (delta, new_mask) when replacing old_val with new_val."""
        new_mask = mask & ~(1 << (old_val - 1)) | (1 << (new_val - 1))
        
        # Find d's that could be affected
        affected_ds = set()
        # d's involving old_val
        for d in range(1, MAX_D + 1):
            # old_val as first: old_val, old_val+d, old_val+2d
            if old_val + 2 * d <= MAX_VAL:
                affected_ds.add(d)
            # old_val as middle: old_val-d, old_val, old_val+d
            if old_val - d >= 1 and old_val + d <= MAX_VAL:
                affected_ds.add(d)
            # old_val as last: old_val-2d, old_val-d, old_val
            if old_val - 2 * d >= 1:
                affected_ds.add(d)
        
        # d's involving new_val
        for d in range(1, MAX_D + 1):
            if new_val + 2 * d <= MAX_VAL:
                affected_ds.add(d)
            if new_val - d >= 1 and new_val + d <= MAX_VAL:
                affected_ds.add(d)
            if new_val - 2 * d >= 1:
                affected_ds.add(d)
        
        delta = 0
        for d in affected_ds:
            old_has = (mask & (mask >> d) & (mask >> (2 * d))) != 0
            new_has = (new_mask & (new_mask >> d) & (new_mask >> (2 * d))) != 0
            if old_has != new_has:
                delta += 1 if new_has else -1
        
        return delta, new_mask

    # Grid construction helper
    def grid_construct(r, c, R, C, offset):
        S = set()
        for i in range(r):
            for j in range(c):
                val = i * R + j * C + offset
                if 1 <= val <= 512:
                    S.add(val)
        return S

    # Generate initial candidates from grid structures
    best_mask = 0
    best_score = 0
    current_list = []

    # Try various grid configurations to get a strong initial seed
    random.seed(42)
    configs = []
    for r in range(2, 33):
        if 32 % r == 0:
            c = 32 // r
            configs.append((r, c))

    for (r, c) in configs:
        if not time_ok():
            break
        for _ in range(50):
            if not time_ok():
                break
            R = random.randint(c + 1, min(512 // r + c, 200))
            C = random.randint(1, min(512 // c + 1, 50))
            offset = random.randint(1, 10)
            S = grid_construct(r, c, R, C, offset)
            if len(S) != 32:
                continue
            m = list_to_mask(list(S))
            sc = score_mask(m)
            if sc > best_score:
                best_score = sc
                best_mask = m
                current_list = list(S)

    # If no good grid found, start random
    if best_score == 0:
        current_list = random.sample(range(1, 513), N)
        best_mask = list_to_mask(current_list)
        best_score = score_mask(best_mask)
        current_mask = best_mask
    else:
        current_mask = best_mask
        current_score = best_score

    # Simulated Annealing with fast delta evaluation
    T = 50.0
    T_min = 0.1
    cooling = 0.9995  # Slower cooling to explore more

    # Precompute complement for faster sampling
    # We'll maintain a list of current elements for easy removal
    current_list = mask_to_list(current_mask)
    
    # We need a set for O(1) lookup in current_list
    current_set = set(current_list)

    while time_ok():
        # Pick element to remove
        val_out = current_list[random.randint(0, N - 1)]
        
        # Pick element to add
        # Sample from 1..512 not in current_set
        val_in = random.randint(1, 512)
        while val_in in current_set:
            val_in = random.randint(1, 512)
        
        # Apply move using fast delta
        delta, new_mask = delta_replace(current_mask, val_out, val_in)
        new_score = current_score + delta
        
        # Accept if improvement or with probability exp(-delta/T) for worsening
        if delta > 0 or random.random() < pow(2.71828, -delta / T):
            # Accept
            current_mask = new_mask
            current_score = new_score
            
            # Update list and set
            idx = current_list.index(val_out)
            current_list[idx] = val_in
            current_set.discard(val_out)
            current_set.add(val_in)
            
            if current_score > best_score:
                best_score = current_score
                best_mask = current_mask
        
        # Cool down
        T *= cooling
        if T < T_min:
            T = T_min

    return mask_to_list(best_mask)
