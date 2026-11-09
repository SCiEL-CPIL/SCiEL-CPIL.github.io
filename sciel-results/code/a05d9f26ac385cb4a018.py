import time

def construct():
    N = 128
    # Precompute all 4-APs
    aps = []
    for d in range(1, (N - 3) // 3 + 1):
        for start in range(N - 3 * d):
            aps.append((start, start + d, start + 2 * d, start + 3 * d))
    
    # For each position, precompute:
    # - ap_indices: list of AP indices where this position appears
    # - For each such AP, which position in the AP (0,1,2,3)
    index_to_aps = [[] for _ in range(N)]
    for idx, ap in enumerate(aps):
        for pos in ap:
            index_to_aps[pos].append(idx)
    
    # We'll try K from 3 upward
    for K in range(3, 15):
        colors = [-1] * N  # -1 means uncolored
        
        # Greedy: assign colors from left to right
        possible = True
        for i in range(N):
            valid_colors = []
            for c in range(K):
                # Check all APs that contain i and where the other 3 positions are already colored
                ok = True
                for ap_idx in index_to_aps[i]:
                    ap = aps[ap_idx]
                    # Check if all other positions are colored
                    other_positions = [pos for pos in ap if pos != i]
                    if all(colors[pos] != -1 for pos in other_positions):
                        # Form the set of colors in this AP with c at i
                        color_set = set()
                        for pos in ap:
                            if pos == i:
                                color_set.add(c)
                            else:
                                color_set.add(colors[pos])
                        if len(color_set) < 3:
                            ok = False
                            break
                if ok:
                    valid_colors.append(c)
            
            if not valid_colors:
                possible = False
                break
            colors[i] = valid_colors[0]  # Pick the smallest valid color
        
        if possible:
            # Verify the entire coloring
            valid = True
            for ap in aps:
                c0, c1, c2, c3 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
                if len({c0, c1, c2, c3}) < 3:
                    valid = False
                    break
            if valid:
                return colors
    
    # Fallback: use the original seed's greedy if we failed (should not happen)
    colors = []
    for x in range(1, 129):
        for color in range(128):
            if (all(len({color} | {colors[x - 1 - j * d] for j in (1, 2, 3)}) >= 3 for d in range(1, (x - 1) // 3 + 1))
                and all(any(colors[x - 1 - j * d] != color for j in (1, 2)) for d in range(1, (x - 1) // 2 + 1))):
                colors.append(color)
                break
    return colors
