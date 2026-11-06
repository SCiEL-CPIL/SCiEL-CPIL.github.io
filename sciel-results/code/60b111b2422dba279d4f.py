import random

def construct():
    n = 128
    num_colors = 3
    
    # Generate all 4-APs
    aps = []
    for a in range(1, n + 1):
        for d in range(1, 43):  # 3d <= 127 -> d <= 42
            e = a + 3 * d
            if e > n:
                break
            b = a + d
            c = a + 2 * d
            aps.append((a - 1, b - 1, c - 1, e - 1))
    
    num_aps = len(aps)
    
    # Position -> list of AP indices
    pos_to_aps = [[] for _ in range(n)]
    for idx, ap in enumerate(aps):
        for pos in ap:
            pos_to_aps[pos].append(idx)
    
    def check_ap(colors, ap):
        c0, c1, c2, c3 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
        # Count distinct colors
        # Fast check: are there at least 3 distinct?
        if c0 == c1 == c2 == c3:
            return False
        # Check if only 2 distinct
        if c0 == c1 and c0 == c2 and c0 == c3:
            return False
        if c0 == c1 and c0 == c2 and c1 != c3:
            return False  # 3 same, 1 diff = 2 colors
        if c0 == c1 and c0 != c2 and c2 == c3:
            return False  # 2,1,1 but c0=c1, c2=c3 -> 2 colors
        if c0 == c1 and c0 != c2 and c0 != c3 and c2 == c3:
            return False  # 2,1,1 -> 2 colors
        if c0 == c2 and c0 != c1 and c1 != c3 and c2 != c3:
            # c0=c2, c1, c3 all different from c0 and each other? 
            # c0=c2, c1!=c0, c3!=c0, c1!=c3 -> 3 colors
            return True
        # Let's just use a set for correctness
        return len(set((c0, c1, c2, c3))) >= 3
    
    # Initialize
    rng = random.Random(42)
    colors = [rng.randint(0, num_colors - 1) for _ in range(n)]
    
    ap_viol = [not check_ap(colors, ap) for ap in aps]
    cur_viol = sum(ap_viol)
    
    max_iters = 50000
    for it in range(max_iters):
        if cur_viol == 0:
            break
        
        # Pick random violating AP
        viol_indices = [i for i, v in enumerate(ap_viol) if v]
        if not viol_indices:
            break
        ap_idx = viol_indices[rng.randint(0, len(viol_indices) - 1)]
        ap = aps[ap_idx]
        
        best_delta = 0
        best_pos = -1
        best_color = -1
        
        for pos in ap:
            old_c = colors[pos]
            for new_c in range(num_colors):
                if new_c == old_c:
                    continue
                colors[pos] = new_c
                delta = 0
                for aidx in pos_to_aps[pos]:
                    old_v = ap_viol[aidx]
                    new_v = not check_ap(colors, aps[aidx])
                    if old_v and not new_v:
                        delta -= 1
                    elif not old_v and new_v:
                        delta += 1
                    ap_viol[aidx] = new_v
                if -delta > best_delta:
                    best_delta = -delta
                    best_pos = pos
                    best_color = new_c
                    best_delta_val = delta
                # Revert
                colors[pos] = old_c
                for aidx in pos_to_aps[pos]:
                    ap_viol[aidx] = not check_ap(colors, aps[aidx])
        
        if best_pos >= 0 and best_delta > 0:
            colors[best_pos] = best_color
            for aidx in pos_to_aps[best_pos]:
                old_v = ap_viol[aidx]
                new_v = not check_ap(colors, aps[aidx])
                if old_v and not new_v:
                    cur_viol -= 1
                elif not old_v and new_v:
                    cur_viol += 1
                ap_viol[aidx] = new_v
        else:
            # Random move
            pos = ap[rng.randint(0, 3)]
            old_c = colors[pos]
            new_c = rng.randint(0, num_colors - 1)
            if new_c != old_c:
                colors[pos] = new_c
                for aidx in pos_to_aps[pos]:
                    old_v = ap_viol[aidx]
                    new_v = not check_ap(colors, aps[aidx])
                    if old_v and not new_v:
                        cur_viol -= 1
                    elif not old_v and new_v:
                        cur_viol += 1
                    ap_viol[aidx] = new_v
    
    return colors
