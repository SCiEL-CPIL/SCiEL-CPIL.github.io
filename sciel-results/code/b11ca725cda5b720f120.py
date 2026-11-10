"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    n = 256
    
    # 1. Precomputation
    weights = [0] * (n + 1)
    for a in range(1, n + 1):
        for d in range(1, (n - a) // 2 + 1):
            b = a + d
            c = a + 2 * d
            if c <= n:
                weights[a] += 1
                weights[b] += 1
                weights[c] += 1
    
    four_aps = []
    vertex_to_4aps = [[] for _ in range(n + 1)]
    for d in range(1, (n - 1) // 3 + 1):
        for a in range(1, n - 3 * d + 1):
            ap = (a, a + d, a + 2 * d, a + 3 * d)
            idx = len(four_aps)
            four_aps.append(ap)
            for v in ap:
                vertex_to_4aps[v].append(idx)
                
    ap_masks = [0] * len(four_aps)
    for i, ap in enumerate(four_aps):
        m = 0
        for v in ap:
            m |= (1 << v)
        ap_masks[i] = m
        
    v_ap_info = [[] for _ in range(n + 1)]
    for i, ap in enumerate(four_aps):
        for v in ap:
            others_mask = ap_masks[i] ^ (1 << v)
            v_ap_info[v].append((i, others_mask))

    def set_to_mask(s):
        m = 0
        for v in s:
            m |= (1 << v)
        return m
        
    def mask_to_set(m):
        s = set()
        for v in range(1, n+1):
            if m & (1 << v):
                s.add(v)
        return s

    def count_3aps(s):
        s_set = set(s)
        count = 0
        for a in s:
            for d in range(1, (n - a) // 2 + 1):
                if a + d in s_set and a + 2 * d in s_set:
                    count += 1
        return count

    # 2. Greedy Initialization
    sorted_vertices = sorted(range(1, n + 1), key=lambda x: -weights[x])
    current_set = set()
    for v in sorted_vertices:
        can_add = True
        for ap_idx in vertex_to_4aps[v]:
            ap = four_aps[ap_idx]
            others = [u for u in ap if u != v]
            if all(u in current_set for u in others):
                can_add = False
                break
        if can_add:
            current_set.add(v)
            
    best_set = current_set
    best_score = count_3aps(best_set)
    global_best_score = best_score
    global_best_mask = set_to_mask(best_set)
    
    start_time = time.time()
    TIME_LIMIT = 55

    def bnb_optimized(candidates, current_mask, current_score):
        nonlocal global_best_score, global_best_mask
        
        if time.time() - start_time > TIME_LIMIT:
            return
            
        if not candidates:
            if current_score > global_best_score:
                global_best_score = current_score
                global_best_mask = current_mask
            return
            
        ub = current_score
        for v in candidates:
            ub += weights[v]
        if ub <= global_best_score:
            return
            
        v = max(candidates, key=lambda x: weights[x])
        
        can_add = True
        for ap_idx, others_mask in v_ap_info[v]:
            if (current_mask & others_mask) == others_mask:
                can_add = False
                break
                
        if can_add:
            new_mask = current_mask | (1 << v)
            new_score = current_score + weights[v]
            
            new_candidates = []
            for u in candidates:
                if u == v:
                    continue
                is_forbidden = False
                for ap_idx, others_mask in v_ap_info[v]:
                    if (others_mask >> u) & 1:
                        remaining_mask = others_mask ^ (1 << u)
                        if (current_mask & remaining_mask) == remaining_mask:
                            is_forbidden = True
                            break
                if not is_forbidden:
                    new_candidates.append(u)
                    
            bnb_optimized(new_candidates, new_mask, new_score)
            
        exclude_candidates = [u for u in candidates if u != v]
        bnb_optimized(exclude_candidates, current_mask, current_score)
        
    greedy_mask = set_to_mask(best_set)
    remaining = [v for v in range(1, n + 1) if v not in best_set]
    
    initial_candidates = []
    for v in remaining:
        can_add = True
        for ap_idx, others_mask in v_ap_info[v]:
            if (greedy_mask & others_mask) == others_mask:
                can_add = False
                break
        if can_add:
            initial_candidates.append(v)
            
    bnb_optimized(initial_candidates, greedy_mask, best_score)
    
    final_set = mask_to_set(global_best_mask)
    return sorted(final_set)

# EVOLVE-BLOCK-END
