import time
import random
import numpy as np

def construct():
    n = 256
    deadline = time.time() + 115.0  # safety margin
    
    # Precompute all 4-term arithmetic progressions
    # AP: (a, a+d, a+2d, a+3d) with 1-indexed positions, d >= 1
    # Convert to 0-indexed: (i, i+d, i+2d, i+3d) where i+3d < n
    aps = []
    for d in range(1, n // 3 + 1):
        for i in range(0, n - 3 * d):
            aps.append((i, i + d, i + 2 * d, i + 3 * d))
    
    aps = np.array(aps, dtype=np.int32)
    num_aps = len(aps)
    
    # For each position, precompute which APs it belongs to
    # pos_aps[pos] = list of AP indices containing position pos
    pos_aps = [[] for _ in range(n)]
    for idx in range(num_aps):
        for pos in aps[idx]:
            pos_aps[pos].append(idx)
    pos_aps = [np.array(p, dtype=np.int32) for p in pos_aps]
    
    def count_mono(color):
        """Count monochromatic 4-APs. color is 0/1 array of length n."""
        # For each AP, check if all 4 positions have the same color
        c0 = color[aps[:, 0]]
        c1 = color[aps[:, 1]]
        c2 = color[aps[:, 2]]
        c3 = color[aps[:, 3]]
        mono = ((c0 == c1) & (c1 == c2) & (c2 == c3))
        return int(np.sum(mono))
    
    # Initialize with Thue-Morse sequence (known to be good for AP avoidance)
    def thue_morse(n):
        return np.array([bin(i).count('1') % 2 for i in range(n)], dtype=np.int8)
    
    best_color = thue_morse(n).copy()
    best_score = count_mono(best_color)
    
    # Also try alternating blocks of 2 as initial
    alt_color = np.array([(i // 2) % 2 for i in range(n)], dtype=np.int8)
    alt_score = count_mono(alt_color)
    if alt_score < best_score:
        best_color = alt_color.copy()
        best_score = alt_score
    
    # Try random initializations too
    for _ in range(10):
        if time.time() > deadline - 5:
            break
        rand_color = np.random.randint(0, 2, size=n, dtype=np.int8)
        rand_score = count_mono(rand_color)
        if rand_score < best_score:
            best_color = rand_color.copy()
            best_score = rand_score
    
    # Local search with simulated annealing
    current_color = best_color.copy()
    current_score = best_score
    
    # Precompute delta for flipping each position
    # When we flip position p, the change in mono count for an AP containing p depends on
    # the other 3 positions' colors and whether the AP was mono before/after
    
    def compute_delta(p, color, score_cache=None):
        """Compute delta in mono count if we flip position p."""
        delta = 0
        for ap_idx in pos_aps[p]:
            ap = aps[ap_idx]
            # Get colors of the 4 positions
            c = [color[ap[0]], color[ap[1]], color[ap[2]], color[ap[3]]]
            # Find which position is p
            p_idx = list(ap).index(p)
            
            # Check if AP is currently monochromatic
            was_mono = (c[0] == c[1] == c[2] == c[3])
            
            # Flip p's color
            c[p_idx] = 1 - c[p_idx]
            is_mono_now = (c[0] == c[1] == c[2] == c[3])
            
            if was_mono and not is_mono_now:
                delta -= 1
            elif not was_mono and is_mono_now:
                delta += 1
        
        return delta
    
    # Simulated annealing
    T_start = 1.0
    T_end = 0.01
    cooling_rate = 0.9995
    T = T_start
    
    steps_in_epoch = 5000
    epoch = 0
    
    while time.time() < deadline - 2:
        for step in range(steps_in_epoch):
            if time.time() > deadline - 2:
                break
            
            # Pick random position
            p = np.random.randint(0, n)
            
            # Compute delta
            delta = compute_delta(p, current_color)
            
            # Acceptance criterion
            if delta < 0:
                accept = True
            elif delta == 0:
                accept = True
            else:
                accept = np.random.rand() < np.exp(-delta / max(T, 1e-10))
            
            if accept:
                current_color[p] = 1 - current_color[p]
                current_score += delta
                
                if current_score < best_score:
                    best_score = current_score
                    best_color = current_color.copy()
        
        T *= cooling_rate
        epoch += 1
    
    return list(best_color)
