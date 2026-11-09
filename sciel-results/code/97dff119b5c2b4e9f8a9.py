import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill
    n = 256
    
    # Precompute all 4-term APs as index tuples (0-indexed)
    # a, a+d, a+2d, a+3d with 1<=a, d>=1, a+3d<=256
    aps = []
    for d in range(1, n // 3 + 1):
        max_a = n - 3 * d
        for a in range(1, max_a + 1):
            i0 = a - 1
            i1 = a + d - 1
            i2 = a + 2 * d - 1
            i3 = a + 3 * d - 1
            aps.append((i0, i1, i2, i3))
            
    aps_arr = np.array(aps, dtype=np.int32)
    M = len(aps_arr)
    
    # Initial coloring: Thue-Morse
    def thue_morse(i):
        return bin(i).count('1') % 2
        
    colors = np.array([thue_morse(i) for i in range(n)], dtype=np.int8)
    
    # Vectorized count function
    def count_mono(c):
        return np.sum((c[aps_arr[:,0]] == c[aps_arr[:,1]]) & 
                      (c[aps_arr[:,1]] == c[aps_arr[:,2]]) & 
                      (c[aps_arr[:,2]] == c[aps_arr[:,3]]))
                      
    current_score = count_mono(colors)
    best_score = current_score
    best_colors = colors.copy()
    
    # Precompute for each position i, the list of AP indices where i appears
    # And the role of i in that AP (0, 1, 2, or 3)
    # This allows efficient delta calculation.
    # However, for n=256 and M~10k, a simpler approach is to just recompute 
    # or use a specific delta formula.
    
    # Let's use the explicit delta formula for flipping bit i.
    # Let c be the current colors.
    # Let flip i.
    # For an AP (j0, j1, j2, j3), if i is not in it, delta = 0.
    # If i is in it, say at position k:
    # The AP is mono if all 4 bits are equal.
    # Let b0, b1, b2, b3 be the bits.
    # If we flip bk, the new bits are b0, ..., 1-bk, ...
    # Old mono: b0==b1==b2==b3.
    # New mono: b0==...==1-bk==...
    
    # It is faster to just iterate over APs containing i.
    # Let's build the index lists.
    aps_containing = [[] for _ in range(n)]
    for idx, ap in enumerate(aps):
        for pos in range(4):
            aps_containing[ap[pos]].append((idx, pos))
            
    # Convert to numpy arrays for speed? Or just lists. Lists are fine for 256 items.
    
    def compute_best_flip(colors, current_score):
        """
        Returns (best_idx, best_delta) or (None, 0) if no improving flip.
        Delta = new_score - current_score. We want min delta.
        """
        best_idx = -1
        best_delta = 0
        
        for i in range(n):
            delta = 0
            ci = colors[i]
            # For each AP containing i
            for ap_idx, pos in aps_containing[i]:
                a0 = aps_arr[ap_idx, 0]
                a1 = aps_arr[ap_idx, 1]
                a2 = aps_arr[ap_idx, 2]
                a3 = aps_arr[ap_idx, 3]
                
                b0 = colors[a0]
                b1 = colors[a1]
                b2 = colors[a2]
                b3 = colors[a3]
                
                # Check if AP is currently mono
                is_mono_old = (b0 == b1) and (b1 == b2) and (b2 == b3)
                
                # Check if AP becomes mono after flipping i (at pos)
                # We don't need to actually flip, just check conditions.
                # The new bit at pos is 1-ci.
                # The other bits remain b0, b1, b2, b3 (with the one at pos effectively changed).
                
                # Let's just check the 4 values.
                # To avoid confusion, let's construct the new tuple values.
                # But that's slow.
                
                # Logic:
                # If pos == 0: new bits are (1-ci, b1, b2, b3)
                # If pos == 1: new bits are (b0, 1-ci, b2, b3)
                # etc.
                
                # We can check:
                # Are all 4 new bits equal?
                # This requires 3 comparisons.
                
                # Optimization:
                # If the other 3 bits are not all equal to each other, the new AP cannot be mono.
                # If the other 3 bits are all equal to X, then the new AP is mono if (1-ci) == X.
                
                # Let's extract the 3 other bits.
                if pos == 0:
                    others = [b1, b2, b3]
                elif pos == 1:
                    others = [b0, b2, b3]
                elif pos == 2:
                    others = [b0, b1, b3]
                else:
                    others = [b0, b1, b2]
                
                # Check if others are all equal
                if others[0] == others[1] and others[1] == others[2]:
                    val_other = others[0]
                    new_bit = 1 - ci
                    is_mono_new = (new_bit == val_other)
                else:
                    is_mono_new = False
                
                # Delta contribution:
                # If old was mono and new is not: -1
                # If old was not mono and new is mono: +1
                # Else: 0
                
                if is_mono_old and not is_mono_new:
                    delta -= 1
                elif not is_mono_old and is_mono_new:
                    delta += 1
            
            if delta < best_delta:
                best_delta = delta
                best_idx = i
                
        return best_idx, best_delta

    # Greedy Coordinate Descent
    # Loop until no improvement or time limit
    while time.time() < deadline:
        improved = False
        # One pass over all coordinates
        for i in range(n):
            if time.time() > deadline:
                break
            # Recompute delta for i specifically? 
            # The previous compute_best_flip did a full scan.
            # It's better to compute delta for i in O(deg(i)) and apply if negative.
            # This is true GCD.
            
            delta = 0
            ci = colors[i]
            for ap_idx, pos in aps_containing[i]:
                a0 = aps_arr[ap_idx, 0]
                a1 = aps_arr[ap_idx, 1]
                a2 = aps_arr[ap_idx, 2]
                a3 = aps_arr[ap_idx, 3]
                
                b0 = colors[a0]
                b1 = colors[a1]
                b2 = colors[a2]
                b3 = colors[a3]
                
                is_mono_old = (b0 == b1) and (b1 == b2) and (b2 == b3)
                
                if pos == 0:
                    others = [b1, b2, b3]
                elif pos == 1:
                    others = [b0, b2, b3]
                elif pos == 2:
                    others = [b0, b1, b3]
                else:
                    others = [b0, b1, b2]
                
                if others[0] == others[1] and others[1] == others[2]:
                    val_other = others[0]
                    new_bit = 1 - ci
                    is_mono_new = (new_bit == val_other)
                else:
                    is_mono_new = False
                
                if is_mono_old and not is_mono_new:
                    delta -= 1
                elif not is_mono_old and is_mono_new:
                    delta += 1
            
            if delta < 0:
                colors[i] = 1 - colors[i]
                current_score += delta
                improved = True
                if current_score < best_score:
                    best_score = current_score
                    best_colors = colors.copy()
        
        if not improved:
            # Local minimum reached.
            # To escape, we can try a random perturbation (kick) and restart GCD.
            # But first, let's see if we have time.
            if time.time() > deadline - 10:
                break
            
            # Random Kick: flip 5 random bits
            for _ in range(5):
                k = np.random.randint(n)
                colors[k] = 1 - colors[k]
            current_score = count_mono(colors)
            if current_score < best_score:
                best_score = current_score
                best_colors = colors.copy()
                
            # Continue loop to try GCD again from this new state
            
    return [int(c) for c in best_colors]
