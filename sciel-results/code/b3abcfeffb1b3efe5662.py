import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    n = 12
    m = 40
    total_masks = 1 << n
    
    # Precompute comparability
    # comp[i] is bitset of j such that i and j are comparable (strict subset in either direction)
    comp = [0] * total_masks
    for i in range(total_masks):
        ci = 0
        for j in range(total_masks):
            if i != j:
                if (i & j) == i or (i & j) == j:
                    ci |= (1 << j)
        comp[i] = ci

    def calc_score(sel_list, sel_mask):
        score = 0
        for s in sel_list:
            score += (comp[s] & sel_mask).bit_count()
        return score // 2

    def greedy_initialization():
        """
        Greedy initialization: pick 40 masks one by one, each time choosing
        the mask that maximizes comparability with already-picked masks.
        """
        selected = []
        sel_mask = 0
        remaining = list(range(total_masks))
        
        # To break symmetry and ensure diversity, we can shuffle remaining initially
        # But greedy will pick the best hubs anyway.
        
        for _ in range(m):
            best_cand = -1
            best_gain = -1
            
            # We can optimize this by noting that if we pick a hub, it connects to many.
            # Scanning all 4096 is fast enough: 40 * 4096 = 163k ops.
            for cand in remaining:
                gain = (comp[cand] & sel_mask).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_cand = cand
            
            if best_cand != -1:
                selected.append(best_cand)
                sel_mask |= (1 << best_cand)
                # Remove best_cand from remaining
                # O(N) removal is fine for 4096
                remaining.remove(best_cand)
        
        return selected, sel_mask

    def local_search(selected_list, selected_mask, current_score, deadline):
        """
        Perform hill climbing with single swaps.
        """
        best_score = current_score
        best_selection = list(selected_list)
        
        improved = True
        # We will iterate until no improvement or deadline
        # To avoid getting stuck in tiny local optima, we might need to allow some bad moves later (SA)
        # But first, pure hill climbing.
        
        while improved and time.time() < deadline:
            improved = False
            for idx in range(m):
                if time.time() >= deadline:
                    break
                old_mask = selected_list[idx]
                sel_without_old_mask = selected_mask & ~(1 << old_mask)
                loss = (comp[old_mask] & sel_without_old_mask).bit_count()
                
                best_new = -1
                best_delta = 0
                
                # Scan all possible new masks
                # Optimization: We can skip masks that are already in selected_list
                # We use selected_mask to check presence quickly.
                for cand in range(total_masks):
                    if (selected_mask >> cand) & 1:
                        continue
                    gain = (comp[cand] & sel_without_old_mask).bit_count()
                    delta = gain - loss
                    if delta > best_delta:
                        best_delta = delta
                        best_new = cand
                
                if best_new != -1 and best_delta > 0:
                    selected_list[idx] = best_new
                    selected_mask = sel_without_old_mask | (1 << best_new)
                    current_score += best_delta
                    improved = True
                    if current_score > best_score:
                        best_score = current_score
                        best_selection = list(selected_list)
        
        return best_score, best_selection, current_score, selected_mask

    def simulated_annealing_polish(selected_list, selected_mask, current_score, deadline):
        """
        Final polish with simulated annealing to escape local optima.
        """
        best_score = current_score
        best_selection = list(selected_list)
        temp = 20.0
        cooling = 0.95
        
        while time.time() < deadline and temp > 0.1:
            idx = random.randint(0, m - 1)
            old_mask = selected_list[idx]
            sel_without_old_mask = selected_mask & ~(1 << old_mask)
            loss = (comp[old_mask] & sel_without_old_mask).bit_count()
            
            cand = random.randint(0, total_masks - 1)
            if (selected_mask >> cand) & 1:
                continue
                
            gain = (comp[cand] & sel_without_old_mask).bit_count()
            delta = gain - loss
            
            if delta > 0 or random.random() < math.exp(delta / temp):
                selected_list[idx] = cand
                selected_mask = sel_without_old_mask | (1 << cand)
                current_score += delta
                if current_score > best_score:
                    best_score = current_score
                    best_selection = list(selected_list)
            
            temp *= cooling
            
        return best_score, best_selection, current_score, selected_mask

    best_overall_score = -1
    best_overall_selection = []
    
    # 1. Run Greedy Initialization
    selected, sel_mask = greedy_initialization()
    current_score = calc_score(selected, sel_mask)
    
    if current_score > best_overall_score:
        best_overall_score = current_score
        best_overall_selection = list(selected)
    
    # 2. Run Local Search on the Greedy result
    if time.time() < deadline - 5.0:
        ls_score, ls_selection, ls_current, ls_mask = local_search(
            list(selected), sel_mask, current_score, deadline - 5.0
        )
        if ls_score > best_overall_score:
            best_overall_score = ls_score
            best_overall_selection = ls_selection
    
    # 3. Run Simulated Annealing on the best result so far
    if time.time() < deadline - 1.0:
        sa_score, sa_selection, sa_current, sa_mask = simulated_annealing_polish(
            list(best_overall_selection), 
            0, # Reconstruct mask
            0, # Reconstruct score? No, we need the actual mask and score.
            deadline - 0.5
        )
        # Wait, I need to pass the correct mask and score from best_overall_selection
        # Let's recalculate or keep track.
        # In the block above, I didn't keep the mask for best_overall_selection if it came from greedy.
        # Let's fix the flow to keep track of (score, selection, mask).

    # Let's rewrite the main flow to be cleaner.

    best_state = None # (score, selection, mask)
    
    # Greedy
    sel_g, mask_g = greedy_initialization()
    score_g = calc_score(sel_g, mask_g)
    best_state = (score_g, list(sel_g), mask_g)
    
    # Local Search
    if time.time() < deadline - 10.0:
        ls_score, ls_sel, ls_cur, ls_mask = local_search(list(sel_g), mask_g, score_g, deadline - 10.0)
        if ls_score > best_state[0]:
            best_state = (ls_score, ls_sel, ls_mask)
    
    # Simulated Annealing
    if time.time() < deadline - 1.0:
        sa_score, sa_sel, sa_cur, sa_mask = simulated_annealing_polish(
            list(best_state[1]), best_state[2], best_state[0], deadline - 0.5
        )
        if sa_score > best_state[0]:
            best_state = (sa_score, sa_sel, sa_mask)
    
    # If we have time, try a few random restarts with local search?
    # The greedy start is usually very good.
    # Let's try 2 random restarts if time permits.
    
    time_remaining = deadline - time.time()
    if time_remaining > 15.0:
        for _ in range(2):
            if time.time() > deadline - 5.0:
                break
            # Random initialization
            random.seed()
            sel_r = random.sample(range(total_masks), m)
            mask_r = 0
            for s in sel_r:
                mask_r |= (1 << s)
            score_r = calc_score(sel_r, mask_r)
            
            # Quick local search
            ls_score, ls_sel, ls_cur, ls_mask = local_search(list(sel_r), mask_r, score_r, deadline - 2.0)
            if ls_score > best_state[0]:
                best_state = (ls_score, ls_sel, ls_mask)
                
    # Convert best_state selection to result
    result = []
    for mask in best_state[1]:
        result.append([b for b in range(n) if (mask >> b) & 1])
    
    return result
