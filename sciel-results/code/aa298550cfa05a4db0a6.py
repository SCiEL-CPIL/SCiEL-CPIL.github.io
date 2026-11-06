import time
import numpy as np

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # Precompute all 4-term arithmetic progressions
    # AP is (a, a+d, a+2d, a+3d) with 0 <= a, a+3d < n
    aps = []
    for d in range(1, n // 3 + 1):
        max_a = n - 3 * d
        for a in range(max_a):
            aps.append((a, a + d, a + 2 * d, a + 3 * d))
    
    # Build index of APs containing each element
    aps_containing = [[] for _ in range(n)]
    for idx, ap in enumerate(aps):
        for val in ap:
            aps_containing[val].append(idx)

    def get_score(c):
        score = 0
        for ap in aps:
            a, b, c_val, d_val = ap
            if c[a] == c[b] == c_val == d_val:
                score += 1
        return score

    # Initialize with Thue-Morse
    # Thue-Morse: t(i) = parity of number of 1s in binary representation of i
    # i is 0-indexed here.
    c = np.array([bin(i).count('1') % 2 for i in range(n)], dtype=np.int8)
    
    current_score = get_score(c)
    
    # Local Search
    # We will perform passes until no improvement or time limit
    while True:
        if time.time() > deadline:
            break
            
        improved = False
        # Iterate through indices. Randomizing order might help escape local minima slightly, 
        # but sequential is deterministic and fast. 
        # Let's try sequential first.
        for i in range(n):
            if time.time() > deadline:
                break
                
            target_color = 1 - c[i]
            delta = 0
            
            # Check all APs containing i
            for ap_idx in aps_containing[i]:
                ap = aps[ap_idx]
                # Get the other 3 indices
                others = [x for x in ap if x != i]
                
                # Check if currently monochromatic
                # Since c[i] is the reference, we check if others match c[i]
                is_mono_now = True
                is_mono_after = True
                
                for j in others:
                    if c[j] != c[i]:
                        is_mono_now = False
                    if c[j] != target_color:
                        is_mono_after = False
                        
                if is_mono_now:
                    delta -= 1
                if is_mono_after:
                    delta += 1
                    
            if delta < 0:
                c[i] = target_color
                current_score += delta
                improved = True
        
        if not improved:
            break

    return c.tolist()
