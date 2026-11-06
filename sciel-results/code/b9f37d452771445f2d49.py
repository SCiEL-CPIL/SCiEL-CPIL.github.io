import numpy as np
import time
from functools import reduce

def construct():
    N = 256
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    # Precompute all 4-term APs: (a, a+d, a+2d, a+3d) with 1 <= a and a+3d <= N
    # Store as 0-indexed tuples
    aps = []
    for a in range(1, N + 1):
        for d in range(1, N + 1):
            if a + 3 * d <= N:
                aps.append((a - 1, a + d - 1, a + 2 * d - 1, a + 3 * d - 1))
    
    num_aps = len(aps)
    
    # Build adjacency: for each position i, list of (ap_index, position_in_ap)
    pos_to_aps = [[] for _ in range(N)]
    for idx, ap in enumerate(aps):
        for pos in ap:
            pos_to_aps[pos].append(idx)
    
    def count_mono(x):
        """Count monochromatic 4-APs given binary array x (0-indexed)."""
        count = 0
        for ap in aps:
            a, b, c, d = ap
            if x[a] == x[b] == x[c] == x[d]:
                count += 1
        return count
    
    def delta_for_flip(x, i):
        """Compute change in mono count if we flip bit i."""
        # For each AP containing i, check if flipping i changes mono status
        change = 0
        for ap_idx in pos_to_aps[i]:
            ap = aps[ap_idx]
            # Get the other 3 positions
            others = [ap[j] for j in range(4) if ap[j] != i]
            other_vals = [x[j] for j in others]
            cur_val = x[i]
            
            # Current: mono if cur_val == all other_vals
            cur_mono = (cur_val == other_vals[0] and cur_val == other_vals[1] and cur_val == other_vals[2])
            
            # After flip: new_val = 1 - cur_val
            new_val = 1 - cur_val
            new_mono = (new_val == other_vals[0] and new_val == other_vals[1] and new_val == other_vals[2])
            
            if cur_mono and not new_mono:
                change -= 1
            elif not cur_mono and new_mono:
                change += 1
        return change
    
    # Initialize with Thue-Morse sequence (known to have few monochromatic APs)
    def thue_morse(n):
        """Return Thue-Morse sequence of length n."""
        seq = [0]
        while len(seq) < n:
            seq = seq + [1 - b for b in seq]
        return seq[:n]
    
    x = np.array(thue_morse(N), dtype=np.int8)
    best_x = x.copy()
    best_count = count_mono(best_x)
    
    # Coordinate descent with multiple restarts
    max_restarts = 20
    max_iters_per_restart = 50000
    
    for restart in range(max_restarts):
        if restart > 0:
            # Random restart with some noise
            if time.time() > deadline - 10.0:
                break
            x = best_x.copy()
            # Flip a few random bits
            num_flips = np.random.randint(1, 10)
            flip_positions = np.random.choice(N, size=num_flips, replace=False)
            for pos in flip_positions:
                x[pos] = 1 - x[pos]
        
        improved = True
        iter_count = 0
        while improved and iter_count < max_iters_per_restart:
            improved = False
            iter_count += 1
            
            if time.time() > deadline - 5.0:
                break
            
            # Try flipping each position
            for i in range(N):
                d = delta_for_flip(x, i)
                if d < 0:
                    x[i] = 1 - x[i]
                    improved = True
                    cur_count = count_mono(x)
                    if cur_count < best_count:
                        best_count = cur_count
                        best_x = x.copy()
                elif d == 0:
                    # Consider flipping with small probability for diversity
                    if np.random.random() < 0.01:
                        x[i] = 1 - x[i]
        
        # If we hit the iteration limit, try a few more passes
        if iter_count >= max_iters_per_restart:
            for _ in range(1000):
                if time.time() > deadline - 5.0:
                    break
                improved = False
                for i in range(N):
                    d = delta_for_flip(x, i)
                    if d < 0:
                        x[i] = 1 - x[i]
                        improved = True
                        cur_count = count_mono(x)
                        if cur_count < best_count:
                            best_count = cur_count
                            best_x = x.copy()
                if not improved:
                    break
    
    # Final polish: one more pass of coordinate descent
    for _ in range(100):
        if time.time() > deadline - 2.0:
            break
        improved = False
        for i in range(N):
            d = delta_for_flip(best_x, i)
            if d < 0:
                best_x[i] = 1 - best_x[i]
                improved = True
                best_count = count_mono(best_x)
        if not improved:
            break
    
    # Return as list of ints
    return [int(v) for v in best_x]
