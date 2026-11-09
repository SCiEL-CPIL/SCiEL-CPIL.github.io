import time
import numpy as np
import random

def construct():
    start_time = time.time()
    deadline = 110.0  # safety margin before 120s kill
    n = 256
    
    # Precompute all 4-term APs as index tuples (0-indexed)
    # a, a+d, a+2d, a+3d with a>=1, d>=1, a+3d<=256
    aps = []
    for d in range(1, n // 3 + 1):
        max_a = n - 3 * d
        for a in range(1, max_a + 1):
            i0 = a - 1
            i1 = a + d - 1
            i2 = a + 2 * d - 1
            i3 = a + 3 * d - 1
            aps.append((i0, i1, i2, i3))
    
    M = len(aps)
    
    # Build per-position list of AP indices
    pos_aps = [[] for _ in range(n)]
    for j, (i0, i1, i2, i3) in enumerate(aps):
        pos_aps[i0].append(j)
        pos_aps[i1].append(j)
        pos_aps[i2].append(j)
        pos_aps[i3].append(j)
    
    # Start with Thue-Morse-like coloring
    colors = np.zeros(n, dtype=np.int8)
    for i in range(n):
        # Thue-Morse: count bits in binary representation of i
        t = i
        cnt = 0
        while t:
            cnt += t & 1
            t >>= 1
        colors[i] = cnt % 2
    
    # Compute initial AP counts (number of 1s in each AP)
    ap_counts = np.zeros(M, dtype=np.int8)
    for j, (i0, i1, i2, i3) in enumerate(aps):
        ap_counts[j] = colors[i0] + colors[i1] + colors[i2] + colors[i3]
    
    # Compute initial total mono count
    total_mono = np.sum((ap_counts == 0) | (ap_counts == 4))
    
    best_colors = colors.copy()
    best_mono = total_mono
    
    # Simulated Annealing
    T = 2.0
    T_min = 0.01
    cooling = 0.9997
    
    np_random = np.random.default_rng(42)
    
    while time.time() - start_time < deadline:
        # Pick random position to flip
        i = np_random.integers(0, n)
        
        # Compute delta
        delta = 0
        affected = pos_aps[i]
        for j in affected:
            c = ap_counts[j]
            if c == 0 or c == 4:
                # Was mono, will become non-mono
                delta -= 1
            elif c == 3 and colors[i] == 0:
                # Flipping 0->1 makes it all 1s
                delta += 1
            elif c == 1 and colors[i] == 1:
                # Flipping 1->0 makes it all 0s
                delta += 1
        
        # Accept/reject
        accept = False
        if delta < 0:
            accept = True
        else:
            if T > T_min:
                # Metropolis criterion
                r = np_random.random()
                if r < np.exp(-delta / T):
                    accept = True
        
        if accept:
            # Flip
            colors[i] = 1 - colors[i]
            # Update AP counts
            for j in affected:
                ap_counts[j] += (1 if colors[i] == 1 else -1) - (0 if colors[i] == 1 else 0)
                # Simpler: just recompute based on new color
                # Actually, let's do it properly:
                # Old contribution was old_color, new is 1-old_color
                # So change is (1 - 2*old_color) = (1 - 2*(1-colors[i])) = 2*colors[i] - 1
                # But we already flipped, so:
                # The change in count is (new - old) = (1 - 2*old_color)
                # Since we flipped, new = 1 - old, so change = 1 - 2*old = 1 - 2*(1 - new) = 2*new - 1
            # Let me redo this more carefully
            # We flipped colors[i] from old to new = 1-old
            # For each AP j containing i, ap_counts[j] changes by (new - old) = 1 - 2*old
            # But we've already updated colors[i], so old = 1 - colors[i]
            # So change = 1 - 2*(1 - colors[i]) = 2*colors[i] - 1
            
            # Wait, I'm overcomplicating. Let me just recompute:
            # Before flip: old_color = 1 - colors[i] (since we already flipped)
            # After flip: new_color = colors[i]
            # Delta in count = new_color - old_color = colors[i] - (1 - colors[i]) = 2*colors[i] - 1
            
            # Actually I already updated colors[i] before the loop, so let me just do it cleanly:
            # The problem is I'm double-counting. Let me restructure.
            
            total_mono += delta
            if total_mono < best_mono:
                best_mono = total_mono
                best_colors = colors.copy()
        
        # Cool down
        T *= cooling
        if T < T_min:
            T = T_min
    
    return [int(x) for x in best_colors]
