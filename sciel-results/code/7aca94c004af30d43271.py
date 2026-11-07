import time
import random
import math

_start_time = time.time()

def construct():
    deadline = _start_time + 85
    n = 12
    m = 40
    total_masks = 1 << n  # 4096

    # Precompute comparability matrix using bytearray for memory efficiency
    # comp[a][b] = 1 if a and b are comparable (one is a strict subset of the other)
    # Since we only consider distinct sets, a != b, so strict subset is just subset
    comp = [bytearray(total_masks) for _ in range(total_masks)]
    for a in range(total_masks):
        for b in range(a + 1, total_masks):
            if (a | b) == a or (a | b) == b:
                comp[a][b] = 1
                comp[b][a] = 1

    # Precompute neighbor lists for fast iteration
    comp_list = [[] for _ in range(total_masks)]
    for a in range(total_masks):
        for b in range(total_masks):
            if comp[a][b]:
                comp_list[a].append(b)

    def mask_to_list(mask):
        return [int(b) for b in range(n) if (mask >> b) & 1]

    def list_to_mask(lst):
        mask = 0
        for b in lst:
            mask |= (1 << b)
        return mask

    # Helper to evaluate a family and return score, in_fam, sum_comp, contribs
    def evaluate(fam):
        in_fam = [False] * total_masks
        for mask in fam:
            in_fam[mask] = True
            
        sum_comp = [0] * total_masks
        for s in range(total_masks):
            sc = 0
            for x in fam:
                sc += comp[s][x]
            sum_comp[s] = sc
            
        contribs = [sum_comp[fam[i]] for i in range(m)]
        current_score = sum(contribs) // 2
        return current_score, in_fam, sum_comp, contribs

    # Greedy chain-based initialization
    def greedy_init():
        family = []
        seen = set()
        
        # Build a primary chain: 0, {0}, {0,1}, {0,1,2}, ..., {0,...,11}
        # That's 13 sets
        for sz in range(n + 1):
            mask = (1 << sz) - 1
            family.append(mask)
            seen.add(mask)
        
        # Now we have 13 sets. Need 27 more.
        # Greedily add sets that maximize comparability with existing family
        remaining = [mask for mask in range(total_masks) if mask not in seen]
        
        for _ in range(m - len(family)):
            best_mask = -1
            best_deg = -1
            for c in remaining:
                deg = 0
                for s in family:
                    if comp[c][s]:
                        deg += 1
                if deg > best_deg:
                    best_deg = deg
                    best_mask = c
            if best_mask == -1:
                break
            family.append(best_mask)
            seen.add(best_mask)
            remaining.remove(best_mask)
        
        # Ensure exactly 40
        while len(family) < m:
            for mask in range(total_masks):
                if mask not in seen:
                    family.append(mask)
                    seen.add(mask)
                    break
        
        return family

    # Multi-restart local search with simulated annealing
    def local_search(init_family, time_limit):
        fam = init_family[:]
        current_score, in_fam, sum_comp, contribs = evaluate(fam)
        best_family = fam[:]
        best_score = current_score
        best_in_fam = in_fam[:]
        best_sum_comp = sum_comp[:]
        best_contribs = contribs[:]
        
        T = 3.0
        cooling = 0.999
        T_min = 0.05
        no_improve = 0
        top_k = 3
        
        while time.time() < time_limit:
            # Find best 1-swap and collect top candidates for 2-swap
            best_1_swap_delta = -1000
            best_1_swap_i = -1
            best_1_swap_s = -1
            
            top_candidates = [[] for _ in range(m)]
            
            for i in range(m):
                old_val = fam[i]
                candidates = []
                
                for s in range(total_masks):
                    if in_fam[s]:
                        continue
                    val = sum_comp[s] - comp[s][old_val]
                    candidates.append((val, s))
                
                candidates.sort(key=lambda x: x[0], reverse=True)
                top_candidates[i] = candidates[:top_k]
                
                if candidates:
                    val, s = candidates[0]
                    delta = val - contribs[i]
                    if delta > best_1_swap_delta:
                        best_1_swap_delta = delta
                        best_1_swap_i = i
                        best_1_swap_s = s

            # Find best 2-swap using top candidates
            best_2_swap_delta = -1000
            best_2_swap_i1 = -1
            best_2_swap_i2 = -1
            best_2_swap_s1 = -1
            best_2_swap_s2 = -1
            
            for i1 in range(m):
                if not top_candidates[i1]:
                    continue
                for i2 in range(i1 + 1, m):
                    if not top_candidates[i2]:
                        continue
                    
                    old1 = fam[i1]
                    old2 = fam[i2]
                    
                    for delta1, s1 in top_candidates[i1]:
                        d1 = delta1 - contribs[i1]
                        for delta2, s2 in top_candidates[i2]:
                            if s1 == s2:
                                continue
                            d2 = delta2 - contribs[i2]
                            old_int = comp[old1][old2]
                            new_int = comp[s1][s2]
                            cross = comp[s1][old2] + comp[s2][old1]
                            
                            total_delta = d1 + d2 + old_int - cross + new_int
                            
                            if total_delta > best_2_swap_delta:
                                best_2_swap_delta = total_delta
                                best_2_swap_i1 = i1
                                best_2_swap_i2 = i2
                                best_2_swap_s1 = s1
                                best_2_swap_s2 = s2

            # Decide move
            accept = False
            move_type = 0
            move_i1 = -1
            move_i2 = -1
            move_s1 = -1
            move_s2 = -1
            move_delta = 0
            
            if best_2_swap_delta > best_1_swap_delta:
                if best_2_swap_delta > 0:
                    accept = True
                    move_type = 2
                    move_i1 = best_2_swap_i1
                    move_i2 = best_2_swap_i2
                    move_s1 = best_2_swap_s1
                    move_s2 = best_2_swap_s2
                    move_delta = best_2_swap_delta
                else:
                    if random.random() < math.exp(best_2_swap_delta / T):
                        accept = True
                        move_type = 2
                        move_i1 = best_2_swap_i1
                        move_i2 = best_2_swap_i2
                        move_s1 = best_2_swap_s1
                        move_s2 = best_2_swap_s2
                        move_delta = best_2_swap_delta
            else:
                if best_1_swap_delta > 0:
                    accept = True
                    move_type = 1
                    move_i1 = best_1_swap_i
                    move_s1 = best_1_swap_s
                    move_delta = best_1_swap_delta
                else:
                    if random.random() < math.exp(best_1_swap_delta / T):
                        accept = True
                        move_type = 1
                        move_i1 = best_1_swap_i
                        move_s1 = best_1_swap_s
                        move_delta = best_1_swap_delta

            if accept:
                if move_type == 1:
                    old_val = fam[move_i1]
                    new_val = move_s1
                    
                    fam[move_i1] = new_val
                    in_fam[old_val] = False
                    in_fam[new_val] = True
                    
                    set_new = set(comp_list[new_val])
                    set_old = set(comp_list[old_val])
                    
                    for s in set_new:
                        if s not in set_old:
                            sum_comp[s] += 1
                    for s in set_old:
                        if s not in set_new:
                            sum_comp[s] -= 1
                            
                    contribs[move_i1] = sum_comp[new_val]
                    for k in range(m):
                        if k == move_i1:
                            continue
                        contribs[k] += comp[new_val][fam[k]] - comp[old_val][fam[k]]
                            
                elif move_type == 2:
                    old1 = fam[move_i1]
                    old2 = fam[move_i2]
                    new1 = move_s1
                    new2 = move_s2
                    
                    fam[move_i1] = new1
                    fam[move_i2] = new2
                    
                    in_fam[old1] = False
                    in_fam[old2] = False
                    in_fam[new1] = True
                    in_fam[new2] = True
                    
                    n_old1 = set(comp_list[old1])
                    n_old2 = set(comp_list[old2])
                    n_new1 = set(comp_list[new1])
                    n_new2 = set(comp_list[new2])
                    
                    all_relevant = n_old1 | n_old2 | n_new1 | n_new2
                    for s in all_relevant:
                        old_c = comp[s][old1] + comp[s][old2]
                        new_c = comp[s][new1] + comp[s][new2]
                        sum_comp[s] += new_c - old_c
                        
                    contribs[move_i1] = sum_comp[new1]
                    contribs[move_i2] = sum_comp[new2]
                    
                    for k in range(m):
                        if k == move_i1 or k == move_i2:
                            continue
                        old_c = comp[old1][fam[k]] + comp[old2][fam[k]]
                        new_c = comp[new1][fam[k]] + comp[new2][fam[k]]
                        contribs[k] += new_c - old_c

                current_score = sum(contribs) // 2
                
                if current_score > best_score:
                    best_score = current_score
                    best_family = fam[:]
                    best_in_fam = in_fam[:]
                    best_sum_comp = sum_comp[:]
                    best_contribs = contribs[:]
                    no_improve = 0
                else:
                    no_improve += 1
                    
                T *= cooling
                if T < T_min:
                    T = T_min
                    
                # Restart logic
                if no_improve > 150:
                    fam = best_family[:]
                    in_fam = best_in_fam[:]
                    sum_comp = best_sum_comp[:]
                    contribs = best_contribs[:]
                    
                    for _ in range(5):
                        i = random.randint(0, m - 1)
                        old_val = fam[i]
                        while True:
                            s = random.randint(0, total_masks - 1)
                            if not in_fam[s]:
                                break
                        new_val = s
                        fam[i] = new_val
                        in_fam[old_val] = False
                        in_fam[new_val] = True
                        
                    current_score, in_fam, sum_comp, contribs = evaluate(fam)
                    
                    T = 2.0
                    no_improve = 0
            else:
                no_improve += 1
                T *= cooling
                if T < T_min:
                    T = T_min

        return best_family, best_score

    # Run multiple restarts with different initializations
    best_family = None
    best_score = -1
    
    # First, try the standard greedy init
    init1 = greedy_init()
    time_per_restart = 15.0
    end_time = deadline - 3.0
    
    fam1, score1 = local_search(init1, min(time.time() + time_per_restart, end_time))
    if score1 > best_score:
        best_score = score1
        best_family = fam1
    
    # Try a few more restarts with perturbed initializations
    for restart in range(3):
        if time.time() >= end_time - 2:
            break
        
        # Create a perturbed version of the best solution found so far
        perturbed = best_family[:]
        for _ in range(10):
            i = random.randint(0, m - 1)
            old_val = perturbed[i]
            used = set(perturbed)
            while True:
                s = random.randint(0, total_masks - 1)
                if s not in used:
                    break
            perturbed[i] = s
        
        # Evaluate perturbed
        p_score, _, _, _ = evaluate(perturbed)
        
        # Run local search on perturbed
        fam2, score2 = local_search(perturbed, min(time.time() + time_per_restart, end_time))
        if score2 > best_score:
            best_score = score2
            best_family = fam2
    
    # Final polish: try exhaustive 1-swaps on the best solution
    if time.time() < end_time - 1:
        polish_fam = best_family[:]
        polish_score, polish_in_fam, polish_sum_comp, polish_contribs = evaluate(polish_fam)
        
        for _ in range(5):
            if time.time() >= end_time - 0.5:
                break
            
            improved = False
            for i in range(m):
                if time.time() >= end_time - 0.5:
                    break
                
                old_val = polish_fam[i]
                best_gain = 0
                best_new = None
                
                for s in range(total_masks):
                    if polish_in_fam[s]:
                        continue
                    val = polish_sum_comp[s] - comp[s][old_val]
                    gain = val - polish_contribs[i]
                    if gain > best_gain:
                        best_gain = gain
                        best_new = s
                
                if best_new is not None and best_gain > 0:
                    old_val = polish_fam[i]
                    new_val = best_new
                    
                    polish_fam[i] = new_val
                    polish_in_fam[old_val] = False
                    polish_in_fam[new_val] = True
                    
                    set_new = set(comp_list[new_val])
                    set_old = set(comp_list[old_val])
                    
                    for s in set_new:
                        if s not in set_old:
                            polish_sum_comp[s] += 1
                    for s in set_old:
                        if s not in set_new:
                            polish_sum_comp[s] -= 1
                            
                    polish_contribs[i] = polish_sum_comp[new_val]
                    for k in range(m):
                        if k == i:
                            continue
                        polish_contribs[k] += comp[new_val][polish_fam[k]] - comp[old_val][polish_fam[k]]
                    
                    polish_score += best_gain
                    improved = True
                    if polish_score > best_score:
                        best_score = polish_score
                        best_family = polish_fam[:]
            
            if not improved:
                break
    
    # Final sanitization: ensure distinct sets
    if len(set(best_family)) < m:
        unique_masks = list(set(best_family))
        used_set = set(unique_masks)
        random.shuffle(unique_masks)
        
        while len(unique_masks) < m:
            s = random.randint(0, total_masks - 1)
            if s not in used_set:
                unique_masks.append(s)
                used_set.add(s)
        
        best_family = unique_masks[:m]
    
    result = []
    for mask in best_family:
        result.append(mask_to_list(mask))
        
    return result

# Quick self-test to ensure no crashes
if __name__ == "__main__":
    t0 = time.time()
    res = construct()
    t1 = time.time()
    assert len(res) == 40
    seen = set()
    for r in res:
        fr = frozenset(r)
        assert fr not in seen
        seen.add(fr)
        for v in r:
            assert isinstance(v, int)
            assert 0 <= v <= 11
    masks = []
    for r in res:
        m = 0
        for v in r:
            m |= (1 << v)
        masks.append(m)
    cnt = 0
    for i in range(40):
        for j in range(i+1, 40):
            a, b = masks[i], masks[j]
            if a | b == a or a | b == b:
                cnt += 1
    print(f"Time: {t1-t0:.2f}s, Comparables: {cnt}")
