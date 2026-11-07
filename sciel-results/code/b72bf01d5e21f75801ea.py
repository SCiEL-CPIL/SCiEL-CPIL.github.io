import time
import random

def construct():
    n = 12
    m = 40
    total_masks = 1 << n
    deadline = time.time() + 82  # Safety margin before 90s kill

    def is_comparable(a, b):
        if a == b:
            return False
        return (a & ~b) == 0 or (b & ~a) == 0

    def count_comparable_pairs(config):
        cnt = 0
        for i in range(len(config)):
            for j in range(i + 1, len(config)):
                if is_comparable(config[i], config[j]):
                    cnt += 1
        return cnt

    # Precompute comparability matrix for the full space? Too large (4096^2 = 16M).
    # Instead, we'll compute on the fly but cache for the current configuration.

    # Strategy:
    # 1. Greedy initialization in a dense subspace (subsets of first k elements)
    # 2. Full-space hill climbing with best-swap moves
    # 3. Periodic random restarts / multi-swap perturbations to escape local optima
    # 4. Track best configuration found

    best_overall_config = None
    best_overall_score = -1

    def run_search(seed_offset=0):
        random.seed(42 + seed_offset)
        
        # Greedy initialization in dense subspace
        k_init = 7
        init_pool = list(range(1 << k_init))
        
        selected = []
        available_init = set(init_pool)
        
        for _ in range(m):
            best_mask = -1
            best_score = -1
            for mask in available_init:
                score = 0
                for s in selected:
                    if is_comparable(mask, s):
                        score += 1
                if score > best_score:
                    best_score = score
                    best_mask = mask
            selected.append(best_mask)
            available_init.remove(best_mask)
            
        current_score = count_comparable_pairs(selected)
        best_config = selected[:]
        best_score = current_score

        selected_set = set(selected)
        
        # Local search: hill climbing with best-swap moves
        no_improve_passes = 0
        max_no_improve = 50
        
        while time.time() < deadline and no_improve_passes < max_no_improve:
            improved = False
            best_delta_this_pass = 0
            
            for i in range(m):
                if time.time() > deadline:
                    return best_config, best_score
                
                old_mask = selected[i]
                
                # Precompute loss for old_mask
                loss = 0
                for j in range(m):
                    if j != i and is_comparable(old_mask, selected[j]):
                        loss += 1
                
                best_new_mask = -1
                best_delta = 0
                
                for new_mask in range(total_masks):
                    if new_mask == old_mask:
                        continue
                    if new_mask in selected_set:
                        continue
                    
                    gain = 0
                    for j in range(m):
                        if j == i:
                            continue
                        if is_comparable(new_mask, selected[j]):
                            gain += 1
                    
                    delta = gain - loss
                    if delta > best_delta:
                        best_delta = delta
                        best_new_mask = new_mask
                
                if best_new_mask != -1 and best_delta > 0:
                    current_score += best_delta
                    selected[i] = best_new_mask
                    selected_set.remove(old_mask)
                    selected_set.add(best_new_mask)
                    improved = True
                    best_delta_this_pass += best_delta
            
            if improved:
                no_improve_passes = 0
                if current_score > best_score:
                    best_score = current_score
                    best_config = selected[:]
            else:
                no_improve_passes += 1
                
                # Try random perturbation to escape local minimum
                if no_improve_passes % 5 == 0 and random.random() < 0.3:
                    # Random 2-swap perturbation
                    i1 = random.randint(0, m - 1)
                    i2 = random.randint(0, m - 1)
                    if i1 != i2:
                        m1 = random.randint(0, total_masks - 1)
                        m2 = random.randint(0, total_masks - 1)
                        
                        if m1 not in selected_set or m1 == selected[i1]:
                            if m2 not in selected_set or m2 == selected[i2]:
                                if m1 != m2:
                                    old_m1 = selected[i1]
                                    old_m2 = selected[i2]
                                    
                                    selected[i1] = m1
                                    selected[i2] = m2
                                    selected_set.remove(old_m1)
                                    selected_set.remove(old_m2)
                                    selected_set.add(m1)
                                    selected_set.add(m2)
                                    
                                    new_score = count_comparable_pairs(selected)
                                    
                                    if new_score > current_score:
                                        current_score = new_score
                                        improved = True
                                        no_improve_passes = 0
                                    else:
                                        # Revert
                                        selected[i1] = old_m1
                                        selected[i2] = old_m2
                                        selected_set.remove(m1)
                                        selected_set.remove(m2)
                                        selected_set.add(old_m1)
                                        selected_set.add(old_m2)
        
        return best_config, best_score

    # Run multiple searches with different seeds to explore different basins
    num_restarts = 5
    for restart in range(num_restarts):
        if time.time() > deadline - 10:  # Leave time for final conversion
            break
        
        config, score = run_search(restart)
        if score > best_overall_score:
            best_overall_score = score
            best_overall_config = config[:]
        
        if time.time() > deadline - 5:
            break

    # Final polish: one more pass of best-swap hill climbing on the best config
    if best_overall_config is not None and time.time() < deadline - 2:
        selected = best_overall_config[:]
        current_score = best_overall_score
        selected_set = set(selected)
        
        for _ in range(10):  # Limited polish passes
            if time.time() > deadline - 1:
                break
            improved = False
            for i in range(m):
                old_mask = selected[i]
                loss = 0
                for j in range(m):
                    if j != i and is_comparable(old_mask, selected[j]):
                        loss += 1
                
                best_new_mask = -1
                best_delta = 0
                for new_mask in range(total_masks):
                    if new_mask == old_mask:
                        continue
                    if new_mask in selected_set:
                        continue
                    gain = 0
                    for j in range(m):
                        if j == i:
                            continue
                        if is_comparable(new_mask, selected[j]):
                            gain += 1
                    delta = gain - loss
                    if delta > best_delta:
                        best_delta = delta
                        best_new_mask = new_mask
                
                if best_new_mask != -1 and best_delta > 0:
                    current_score += best_delta
                    selected[i] = best_new_mask
                    selected_set.remove(old_mask)
                    selected_set.add(best_new_mask)
                    improved = True
            
            if not improved:
                break
            
            if current_score > best_overall_score:
                best_overall_score = current_score
                best_overall_config = selected[:]

    # Convert to lists of ints
    result = []
    for mask in best_overall_config:
        s = [i for i in range(n) if (mask >> i) & 1]
        result.append(s)
    
    return result
