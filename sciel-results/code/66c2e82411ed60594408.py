import time
import random
import numpy as np

def construct():
    N = 51975
    # Find divisors of 51975
    divisors = []
    for i in range(2, int(np.sqrt(N)) + 1):
        if N % i == 0:
            divisors.append(i)
            if i != N // i:
                divisors.append(N // i)
    divisors.append(N)
    divisors = sorted(list(set(divisors)))
    divisors = [d for d in divisors if d > 1]
    
    num_mods = len(divisors)
    
    # Precompute bitmasks for each (modulus, residue)
    # masks[i][r] is an integer where bit x is 1 if x % m == r
    # N=51975, so we need ~52000 bits. Python int handles this.
    masks = []
    for m in divisors:
        mod_masks = []
        for r in range(m):
            mask = 0
            # Efficiently build mask
            # Looping N/m times is fine since sum(N/m) is small
            for x in range(r, N, m):
                mask |= (1 << x)
            mod_masks.append(mask)
        masks.append(mod_masks)
        
    # Helper to count set bits in a Python int
    def popcount(x):
        return bin(x).count('1')
        
    # Helper to iterate set bits
    def iter_bits(mask):
        while mask:
            lsb = mask & -mask
            yield lsb.bit_length() - 1
            mask ^= lsb

    start_time = time.time()
    timeout = 115.0
    
    # State
    residues = [0] * num_mods
    counts = np.zeros(N, dtype=np.int32)
    uncovered_mask = (1 << N) - 1
    
    # Greedy Initialization
    # Order moduli by size? Larger moduli cover fewer residues, so they are more "special".
    # Usually, it's better to place larger moduli first or smaller?
    # Greedy: Pick residue that covers max uncovered.
    # Sort divisors in descending order? Let's try ascending (small moduli cover many, so they are easier to place greedily).
    # Actually, let's just iterate through them.
    
    for i in range(num_mods):
        m = divisors[i]
        best_r = 0
        best_gain = -1
        for r in range(m):
            mask = masks[i][r]
            # Gain is number of bits in mask that are in uncovered_mask
            gain = popcount(mask & uncovered_mask)
            if gain > best_gain:
                best_gain = gain
                best_r = r
                
        residues[i] = best_r
        mask = masks[i][best_r]
        # Update counts and uncovered_mask
        for x in iter_bits(mask):
            if counts[x] == 0:
                uncovered_mask &= ~(1 << x)
            counts[x] += 1
            
    current_score = popcount(uncovered_mask)
    best_score = current_score
    best_residues = residues.copy()
    
    # Local Search / Simulated Annealing
    T = 10.0
    alpha = 0.95
    T_min = 0.1
    
    while time.time() - start_time < timeout and T > T_min:
        # Perform a batch of moves
        for _ in range(1000):
            if time.time() - start_time > timeout:
                break
                
            i = random.randint(0, num_mods - 1)
            m = divisors[i]
            old_r = residues[i]
            
            # Find best new residue (Greedy)
            # We want to minimize score.
            # Score change = (newly uncovered) - (newly covered)
            # Newly covered: bits in M_new that are in uncovered_mask
            # Newly uncovered: bits in M_old that are NOT in M_new AND are covered ONLY by this class?
            # This is hard to compute directly with just masks.
            # We need counts.
            # Let's evaluate all r for this modulus i?
            # m can be up to 51975. Evaluating all r is too slow if m is large.
            # But for small m, it's fast.
            # For large m, N/m is small, so updating counts is fast.
            # However, checking all r for large m is expensive.
            
            # Strategy: 
            # If m is small, try all r.
            # If m is large, try a few random r.
            
            candidates = []
            if m <= 100:
                candidates = list(range(m))
            else:
                # Sample a few
                candidates = [old_r]
                for _ in range(10):
                    candidates.append(random.randint(0, m - 1))
            
            best_new_r = old_r
            best_delta = 0 # 0 means no change (keep old)
            
            M_old = masks[i][old_r]
            
            for new_r in candidates:
                if new_r == old_r:
                    continue
                M_new = masks[i][new_r]
                
                # Calculate delta
                # We need to know which elements in M_old become uncovered.
                # An element x in M_old becomes uncovered if:
                # 1. x is NOT in M_new (so this class stops covering it)
                # 2. counts[x] == 1 (so no other class covers it)
                
                # We need to iterate over M_old \ M_new
                # And check counts[x] == 1
                
                # And newly covered:
                # x in M_new \ M_old AND counts[x] == 0
                
                # This requires iterating over bits.
                # Cost: O(N/m)
                
                # Let's do it
                delta = 0
                
                # Elements that stop being covered by this class
                # M_old & ~M_new
                to_check_uncover = M_old & ~M_new
                # Elements that start being covered by this class
                # M_new & ~M_old
                to_check_cover = M_new & ~M_old
                
                # Iterate to_check_uncover
                # If counts[x] == 1, it becomes uncovered (bad, +1 to score)
                # If counts[x] > 1, it remains covered (no change)
                
                # Iterate to_check_cover
                # If counts[x] == 0, it becomes covered (good, -1 to score)
                # If counts[x] > 0, it remains covered (no change)
                
                # Note: counts[x] is the count BEFORE the move.
                
                mask_uncover = to_check_uncover
                while mask_uncover:
                    lsb = mask_uncover & -mask_uncover
                    x = lsb.bit_length() - 1
                    if counts[x] == 1:
                        delta += 1
                    mask_uncover ^= lsb
                    
                mask_cover = to_check_cover
                while mask_cover:
                    lsb = mask_cover & -mask_cover
                    x = lsb.bit_length() - 1
                    if counts[x] == 0:
                        delta -= 1
                    mask_cover ^= lsb
                    
                if delta < best_delta:
                    best_delta = delta
                    best_new_r = new_r
                    
            # Apply best move
            if best_new_r != old_r:
                # Accept if delta < 0 or with prob exp(-delta/T)
                accept = False
                if best_delta < 0:
                    accept = True
                else:
                    if T > 0:
                        accept = random.random() < np.exp(-best_delta / T)
                    
                if accept:
                    # Update state
                    M_old = masks[i][old_r]
                    M_new = masks[i][best_new_r]
                    
                    # Remove old
                    mask_uncover = M_old & ~M_new
                    while mask_uncover:
                        lsb = mask_uncover & -mask_uncover
                        x = lsb.bit_length() - 1
                        counts[x] -= 1
                        if counts[x] == 0:
                            uncovered_mask |= (1 << x)
                        mask_uncover ^= lsb
                        
                    # Add new
                    mask_cover = M_new & ~M_old
                    while mask_cover:
                        lsb = mask_cover & -mask_cover
                        x = lsb.bit_length() - 1
                        counts[x] += 1
                        if counts[x] == 1:
                            uncovered_mask &= ~(1 << x)
                        mask_cover ^= lsb
                        
                    residues[i] = best_new_r
                    current_score += best_delta
                    
                    if current_score < best_score:
                        best_score = current_score
                        best_residues = residues.copy()
                        
        T *= alpha

    classes = [[divisors[i], int(best_residues[i])] for i in range(num_mods)]
    return classes
