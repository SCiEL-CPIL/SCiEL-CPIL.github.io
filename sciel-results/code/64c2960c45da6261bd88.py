import time
import random
import numpy as np

def construct():
    n = 256
    deadline = time.time() + 110  # Leave 10s buffer for 120s limit
    
    # Precompute all 4-APs: a, a+d, a+2d, a+3d where 1 <= a <= n-3d
    # Using 0-indexed: 0 <= a <= n-1-3d
    ap_list = []
    for d in range(1, (n - 1) // 3 + 1):
        max_a = n - 1 - 3 * d
        for a in range(max_a + 1):
            ap_list.append((a, a + d, a + 2 * d, a + 3 * d))
    
    num_aps = len(ap_list)
    
    # For each bit position, precompute which APs it participates in
    bit_aps = [[] for _ in range(n)]
    for idx, ap in enumerate(ap_list):
        for bit in ap:
            bit_aps[bit].append(idx)
    
    def count_mono(coloring):
        """Count monochromatic 4-APs."""
        count = 0
        for ap in ap_list:
            a, b, c, d = ap
            if coloring[a] == coloring[b] == coloring[c] == coloring[d]:
                count += 1
        return count
    
    def count_mono_fast(coloring):
        """Faster count using precomputed bit_aps."""
        count = 0
        for ap in ap_list:
            a, b, c, d = ap
            if coloring[a] == coloring[b] and coloring[c] == coloring[d] and coloring[a] == coloring[c]:
                count += 1
        return count
    
    def delta_count(coloring, bit, new_val):
        """Compute change in mono count if we flip bit to new_val."""
        old_val = coloring[bit]
        if old_val == new_val:
            return 0
        delta = 0
        for ap_idx in bit_aps[bit]:
            ap = ap_list[ap_idx]
            # Check if this AP is currently mono
            was_mono = all(coloring[x] == old_val for x in ap)
            # Check if it would be mono with new_val
            would_be_mono = all((x == bit or coloring[x] == new_val) for x in ap)
            if was_mono and not would_be_mono:
                delta -= 1
            elif not was_mono and would_be_mono:
                delta += 1
        return delta
    
    # Start with Thue-Morse sequence
    def thue_morse():
        colors = []
        for i in range(n):
            # Number of 1s in binary representation of i, mod 2
            t = i
            cnt = 0
            while t:
                cnt += t & 1
                t >>= 1
            colors.append(cnt % 2)
        return colors
    
    best_coloring = thue_morse()
    best_count = count_mono(best_coloring)
    
    # Local search with simulated annealing
    rng = np.random.default_rng(42)
    
    current_coloring = best_coloring[:]
    current_count = best_count
    
    # Try multiple restarts
    num_restarts = 50
    for restart in range(num_restarts):
        if time.time() > deadline - 5:
            break
        
        if restart == 0:
            current_coloring = best_coloring[:]
            current_count = best_count
        else:
            # Start from a perturbed version of best or random
            if restart % 2 == 0:
                current_coloring = best_coloring[:]
                # Perturb
                num_flips = rng.integers(5, 20)
                for _ in range(num_flips):
                    bit = rng.integers(0, n)
                    current_coloring[bit] = 1 - current_coloring[bit]
            else:
                # Random start
                current_coloring = rng.integers(0, 2, n).tolist()
            current_count = count_mono(current_coloring)
        
        # Simulated annealing
        temp = 50.0
        cooling = 0.995
        min_temp = 0.1
        
        while temp > min_temp:
            if time.time() > deadline - 5:
                break
            
            # Try several moves at this temperature
            moves_at_temp = 100
            for _ in range(moves_at_temp):
                if time.time() > deadline - 5:
                    break
                
                bit = rng.integers(0, n)
                new_val = 1 - current_coloring[bit]
                delta = delta_count(current_coloring, bit, new_val)
                
                if delta < 0 or (delta == 0 and rng.random() < 0.1):
                    current_coloring[bit] = new_val
                    current_count += delta
                elif delta > 0 and rng.random() < np.exp(-delta / temp):
                    current_coloring[bit] = new_val
                    current_count += delta
                
                temp *= cooling
            
            # Check if we improved
            if current_count < best_count:
                best_count = current_count
                best_coloring = current_coloring[:]
        
        # If we found a better solution, continue from there
        if current_count < best_count:
            best_count = current_count
            best_coloring = current_coloring[:]
    
    return best_coloring
