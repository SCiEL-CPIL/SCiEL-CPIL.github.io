"""
Search-based optimizer for two-colourings of [1,256] minimizing monochromatic 4-APs.
Uses simulated annealing with efficient delta computation and a time budget.
"""
import time
import random
import numpy as np

def construct():
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    # Precompute all 4-term APs as index tuples (0-indexed)
    # A 4-AP is (x, x+d, x+2d, x+3d) with 1 <= x and x+3d <= 256, d >= 1
    # 0-indexed: (x-1, x-1+d, x-1+2d, x-1+3d)
    aps = []
    for d in range(1, 86):  # d from 1 to 85 (since 3*d <= 255)
        max_start = 256 - 3 * d
        for a in range(1, max_start + 1):
            i0 = a - 1
            i1 = i0 + d
            i2 = i0 + 2 * d
            i3 = i0 + 3 * d
            aps.append((i0, i1, i2, i3))
    
    num_aps = len(aps)
    
    # Build for each position, the list of AP indices it participates in
    # and the positions of the other 3 elements in each such AP
    pos_to_aps = [[] for _ in range(256)]
    for ap_idx, (i0, i1, i2, i3) in enumerate(aps):
        for pos in (i0, i1, i2, i3):
            others = [x for x in (i0, i1, i2, i3) if x != pos]
            pos_to_aps[pos].append((ap_idx, others))
    
    # Count monochromatic 4-APs
    def count_mono(colors):
        count = 0
        for (i0, i1, i2, i3) in aps:
            if colors[i0] == colors[i1] == colors[i2] == colors[i3]:
                count += 1
        return count
    
    # Compute delta in mono count when flipping position pos
    def calc_delta(colors, pos):
        delta = 0
        for ap_idx, others in pos_to_aps[pos]:
            # Check if all 4 are currently monochromatic
            all_same_now = (colors[others[0]] == colors[others[1]] == colors[others[2]] == colors[pos])
            # Check if all 4 would be monochromatic after flip
            new_color = 1 - colors[pos]
            all_same_after = (colors[others[0]] == colors[others[1]] == colors[others[2]] == new_color)
            
            if all_same_now and not all_same_after:
                delta -= 1  # We're removing a mono AP
            elif not all_same_now and all_same_after:
                delta += 1  # We're adding a mono AP
        return delta
    
    # Initialize with random coloring
    colors = np.random.randint(0, 2, size=256)
    current_count = count_mono(colors)
    best_colors = colors.copy()
    best_count = current_count
    
    # Simulated annealing
    T = 50.0  # Initial temperature
    T_min = 0.1
    cooling = 0.9995
    
    rng = random.Random(42)
    
    while time.time() < deadline and T > T_min:
        # Pick a random position to flip
        pos = rng.randint(0, 255)
        delta = calc_delta(colors, pos)
        
        if delta <= 0 or rng.random() < np.exp(-delta / T):
            colors[pos] = 1 - colors[pos]
            current_count += delta
            
            if current_count < best_count:
                best_count = current_count
                best_colors = colors.copy()
        
        T *= cooling
    
    # Final local search pass (greedy)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for pos in range(256):
            delta = calc_delta(colors, pos)
            if delta < 0:
                colors[pos] = 1 - colors[pos]
                current_count += delta
                improved = True
                if current_count < best_count:
                    best_count = current_count
                    best_colors = colors.copy()
    
    # Return as list of ints in {0,1}
    return [int(c) for c in best_colors]
