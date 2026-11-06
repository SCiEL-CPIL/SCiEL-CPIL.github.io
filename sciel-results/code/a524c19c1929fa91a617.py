# EVOLVE-BLOCK-START
import random
import time

def construct():
    # Precompute all 4-term arithmetic progressions
    ap_indices = []
    for d in range(1, 43):
        for i in range(0, 128 - 3*d):
            ap_indices.append((i, i+d, i+2*d, i+3*d))
    
    n_aps = len(ap_indices)
    
    # Map index to list of AP indices containing it
    aps_containing = [[] for _ in range(128)]
    for idx, ap in enumerate(ap_indices):
        for pos in ap:
            aps_containing[pos].append(idx)

    def is_bad_fast(c1, c2, c3, c4):
        if c1 == c2:
            if c3 == c4:
                return 1
            else:
                if c1 == c3 or c1 == c4 or c3 == c4:
                    return 1
                return 0
        else:
            if c3 == c4:
                if c1 == c3 or c2 == c3 or c1 == c4 or c2 == c4:
                    return 1
                return 0
            else:
                if (c3 == c1 or c3 == c2) and (c4 == c1 or c4 == c2):
                    return 1
                return 0

    def calculate_initial_cost(colors):
        cost = 0
        ap_bad = [0] * n_aps
        for idx, ap in enumerate(ap_indices):
            if is_bad_fast(colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]):
                cost += 1
                ap_bad[idx] = 1
        return cost, ap_bad

    start_time = time.time()
    deadline = start_time + 45
    
    best_overall = None
    best_cost_overall = float('inf')

    for k in [3, 4, 5]:
        if time.time() > deadline - 5:
            break
            
        for restart in range(50):
            if time.time() > deadline - 5:
                break
                
            colors = [random.randint(0, k-1) for _ in range(128)]
            current_cost, ap_bad = calculate_initial_cost(colors)
            
            if current_cost == 0:
                return colors
            
            T = 1.0
            while time.time() < deadline - 5:
                if T < 0.01:
                    break
                
                idx = random.randint(0, 127)
                old_color = colors[idx]
                new_color = random.randint(0, k-1)
                if new_color == old_color:
                    continue
                
                colors[idx] = new_color
                
                affected_aps = aps_containing[idx]
                old_states = []
                delta_cost = 0
                for ap_idx in affected_aps:
                    ap = ap_indices[ap_idx]
                    old_bad = ap_bad[ap_idx]
                    new_bad = is_bad_fast(colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]])
                    old_states.append((ap_idx, old_bad))
                    if new_bad != old_bad:
                        delta_cost += (new_bad - old_bad)
                        ap_bad[ap_idx] = new_bad
                
                current_cost += delta_cost
                
                if current_cost == 0:
                    return colors
                
                if current_cost < best_cost_overall:
                    best_cost_overall = current_cost
                    best_overall = colors[:]
                
                if delta_cost < 0 or random.random() < pow(2.71828, -delta_cost / T):
                    pass
                else:
                    colors[idx] = old_color
                    for ap_idx, old_bad in old_states:
                        ap_bad[ap_idx] = old_bad
                    current_cost -= delta_cost
                
                T *= 0.999

    if best_overall:
        return best_overall
    return [0]*128
# EVOLVE-BLOCK-END
