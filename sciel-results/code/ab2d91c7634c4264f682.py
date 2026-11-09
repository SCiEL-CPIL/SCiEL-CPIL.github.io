import time
import numpy as np
import random

def construct():
    n = 256
    deadline = time.time() + 145.0  # 5s safety margin under 150s limit
    
    # Precompute all 4-term APs: (a, a+d, a+2d, a+3d) with 0-indexed
    aps = []
    for d in range(1, (n - 3) // 3 + 1):
        for a in range(n - 3 * d):
            aps.append((a, a + d, a + 2 * d, a + 3 * d))
    aps = np.array(aps, dtype=np.int32)  # shape (M, 4)
    M = len(aps)
    
    # Build for each position x, the list of AP indices it participates in
    # and its role (0,1,2,3) in that AP
    pos_aps = [[] for _ in range(n)]
    for idx, (a, b, c, dd) in enumerate(aps):
        pos_aps[a].append((idx, 0))
        pos_aps[b].append((idx, 1))
        pos_aps[c].append((idx, 2))
        pos_aps[dd].append((idx, 3))
    
    # For each position, precompute:
    # - the AP indices involving it
    # - for each such AP, the indices of the OTHER 3 positions (to check if they agree with new color)
    pos_ap_indices = []
    pos_other_positions = []  # shape (len, 3) for each position: the other 3 positions in each AP
    for x in range(n):
        if pos_aps[x]:
            arr = np.array(pos_aps[x], dtype=np.int32)
            idxs = arr[:, 0]
            roles = arr[:, 1]
            pos_ap_indices.append(idxs)
            # For each AP, get the 3 other positions
            others = np.empty((len(idxs), 3), dtype=np.int32)
            for j in range(len(idxs)):
                ap = aps[idxs[j]]
                r = roles[j]
                # Get the 3 positions other than role r
                others[j, 0] = ap[(r + 1) % 4]
                others[j, 1] = ap[(r + 2) % 4]
                others[j, 2] = ap[(r + 3) % 4]
            pos_other_positions.append(others)
        else:
            pos_ap_indices.append(np.array([], dtype=np.int32))
            pos_other_positions.append(np.empty((0, 3), dtype=np.int32))
    
    def count_mono(colors):
        """Count monochromatic 4-APs."""
        c0 = colors[aps[:, 0]]
        c1 = colors[aps[:, 1]]
        c2 = colors[aps[:, 2]]
        c3 = colors[aps[:, 3]]
        mono = (c0 == c1) & (c1 == c2) & (c2 == c3)
        return int(mono.sum())
    
    def delta_flip_fast(colors, x):
        """Compute change in cost if we flip position x. Fast version."""
        idxs = pos_ap_indices[x]
        if len(idxs) == 0:
            return 0
        others = pos_other_positions[x]  # shape (k, 3)
        
        # Current colors of the other 3 positions for each AP
        o0 = colors[others[:, 0]]
        o1 = colors[others[:, 1]]
        o2 = colors[others[:, 2]]
        
        # Before: AP is mono iff all 4 positions have same color
        # colors[x] is the current color of position x
        cx = colors[x]
        # All 4 agree iff all 3 others equal cx
        before_mono = (o0 == cx) & (o1 == cx) & (o2 == cx)
        
        # After: position x has color 1-cx
        # AP is mono-after iff all 3 others equal (1-cx)
        new_cx = 1 - cx
        after_mono = (o0 == new_cx) & (o1 == new_cx) & (o2 == new_cx)
        
        return int(after_mono.sum() - before_mono.sum())
    
    def thue_morse(n):
        return np.array([bin(i).count('1') % 2 for i in range(n)], dtype=np.int8)
    
    def greedy_local_search(colors, max_iters=5000):
        """Greedy: repeatedly flip the position with the best (most negative) delta."""
        current_cost = count_mono(colors)
        best_cost = current_cost
        best_colors = colors.copy()
        
        for _ in range(max_iters):
            if time.time() > deadline - 2.0:
                break
            best_delta = 0
            best_x = -1
            for x in range(n):
                d = delta_flip_fast(colors, x)
                if d < best_delta:
                    best_delta = d
                    best_x = x
            if best_x == -1:
                break
            colors[best_x] = 1 - colors[best_x]
            current_cost += best_delta
            if current_cost < best_cost:
                best_cost = current_cost
                best_colors = colors.copy()
        
        return best_colors, best_cost
    
    def iterated_local_search(colors, max_iters=100, perturb_strength=10):
        """ILS: local search, then perturb, repeat."""
        current_cost = count_mono(colors)
        best_cost = current_cost
        best_colors = colors.copy()
        
        for _ in range(max_iters):
            if time.time() > deadline - 3.0:
                break
            
            # Local search
            c, cost = greedy_local_search(colors.copy(), max_iters=3000)
            if cost < best_cost:
                best_cost = cost
                best_colors = c.copy()
            
            # Perturb: flip a random number of positions
            num_flips = random.randint(perturb_strength // 2, perturb_strength)
            flip_positions = random.sample(range(n), num_flips)
            colors = c.copy()
            for x in flip_positions:
                colors[x] = 1 - colors[x]
        
        return best_colors, best_cost
    
    # Multi-start: Thue-Morse + random perturbations
    best_overall = thue_morse(n)
    best_cost_overall = count_mono(best_overall)
    
    # Start 1: pure Thue-Morse with ILS
    c, cost = iterated_local_search(best_overall.copy(), max_iters=50, perturb_strength=8)
    if cost < best_cost_overall:
        best_cost_overall = cost
        best_overall = c
    
    # Additional starts with random perturbations of Thue-Morse
    tm = thue_morse(n)
    num_starts = 12
    for s in range(num_starts):
        if time.time() > deadline - 20.0:
            break
        # Perturb Thue-Morse with random flips
        colors = tm.copy()
        num_flips = random.randint(10, 40)
        flip_positions = random.sample(range(n), num_flips)
        for x in flip_positions:
            colors[x] = 1 - colors[x]
        
        c, cost = iterated_local_search(colors.copy(), max_iters=30, perturb_strength=6)
        if cost < best_cost_overall:
            best_cost_overall = cost
            best_overall = c
    
    # Also try some fully random starts
    for s in range(4):
        if time.time() > deadline - 15.0:
            break
        colors = np.random.randint(0, 2, n, dtype=np.int8)
        c, cost = iterated_local_search(colors.copy(), max_iters=25, perturb_strength=8)
        if cost < best_cost_overall:
            best_cost_overall = cost
            best_overall = c
    
    # Final greedy polish
    if time.time() < deadline - 5.0:
        c, cost = greedy_local_search(best_overall.copy(), max_iters=5000)
        if cost < best_cost_overall:
            best_cost_overall = cost
            best_overall = c
    
    return [int(x) for x in best_overall]
