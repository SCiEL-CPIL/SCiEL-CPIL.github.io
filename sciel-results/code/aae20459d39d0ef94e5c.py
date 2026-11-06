import time
import sys
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    N = 128
    
    # Precompute all 4-term progressions
    aps = []
    for a in range(1, N + 1):
        for d in range(1, N // 3 + 1):
            end = a + 3 * d
            if end <= N:
                aps.append((a - 1, a + d - 1, a + 2 * d - 1, a + 3 * d - 1))
    
    # For each position, store APs where it appears and its index in the AP
    pos_ap_info = [[] for _ in range(N)]
    for idx, ap in enumerate(aps):
        for i, pos in enumerate(ap):
            pos_ap_info[pos].append((idx, i))
    
    def greedy_with_max_colors(max_colors):
        colors = [None] * N
        
        for pos in range(N):
            assigned = False
            for color in range(max_colors):
                valid = True
                for (ap_idx, i) in pos_ap_info[pos]:
                    ap = aps[ap_idx]
                    all_assigned = True
                    for j, p in enumerate(ap):
                        if j != i and colors[p] is None:
                            all_assigned = False
                            break
                    
                    if all_assigned:
                        ap_colors = []
                        for p in ap:
                            if p == pos:
                                ap_colors.append(color)
                            else:
                                ap_colors.append(colors[p])
                        if len(set(ap_colors)) < 3:
                            valid = False
                            break
                
                if valid:
                    colors[pos] = color
                    assigned = True
                    break
            
            if not assigned:
                colors[pos] = 0
        
        return colors
    
    for max_colors in [3, 4, 5, 6, 8, 16]:
        if time.time() > deadline - 10:
            break
        
        colors = greedy_with_max_colors(max_colors)
        
        valid = True
        for ap in aps:
            if len(set([colors[p] for p in ap])) < 3:
                valid = False
                break
        
        if valid:
            unique_colors = sorted(set(colors))
            if len(unique_colors) < max_colors:
                mapping = {c: i for i, c in enumerate(unique_colors)}
                colors = [mapping[c] for c in colors]
            return colors
    
    return [0] * N
