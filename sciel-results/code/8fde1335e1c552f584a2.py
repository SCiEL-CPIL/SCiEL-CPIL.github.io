import time
import random

def construct():
    n = 12
    m = 40
    deadline = time.time() + 80  # Safety margin before 90s kill

    # Precompute masks for a smaller active set to ensure density,
    # but allow the search to expand into the full universe if beneficial.
    # However, the "core" insight from the current program is that restricting
    # to a small k (like 6 or 7) creates a dense comparability graph.
    # The inspirations use full n=12 search with greedy + local search.
    # We will combine:
    # 1. Greedy initialization using a restricted pool (e.g., k=7) to start in a dense region.
    # 2. Local search (hill climbing) over the full space (n=12) to refine.
    # 3. Efficient delta calculation for swaps.

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

    # Strategy:
    # 1. Define a candidate pool. Using the full 4096 masks is fine for n=12.
    # 2. Greedy Construction:
    #    - Start with an empty set.
    #    - Iteratively pick the mask that maximizes comparability with already selected sets.
    #    - To break ties and encourage diversity/density, we can bias towards smaller sets
    #      or use a restricted initial pool (e.g., subsets of {0..6}) to ensure a strong start.
    
    # Let's use a hybrid initialization:
    # First, greedily select from the power set of the first 7 elements (128 masks).
    # This ensures we get a very dense core.
    # Then, fill the remaining slots from the full space (or continue greedy from full space).
    # Actually, pure greedy on full space might get stuck in a suboptimal local minimum
    # if it picks a "bad" large set early.
    # Let's try: Greedy from the pool of subsets of {0..6} (k=7) first.
    # If we need more than 128, we expand. But we only need 40.
    # So, we can just run greedy on the pool of subsets of {0..6}.
    # Then, we run local search on the full space to see if swapping in elements from {7..11} helps.

    k_init = 7
    init_pool = list(range(1 << k_init))
    
    selected = []
    available_init = set(init_pool)
    
    # Greedy step 1: Select 40 from init_pool
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

    # Local Search: Hill Climbing
    # We allow swaps with any mask in the full universe (0 to 2^n - 1).
    # This allows the algorithm to "escape" the k=7 subspace if beneficial.
    
    selected_set = set(selected)
    no_improve_count = 0
    max_no_improve = 200 # Restart or stop if stuck for a while?
    # Since we have a deadline, we just keep going until time runs out or no improvement in a full pass.
    
    while time.time() < deadline:
        improved = False
        # Iterate through indices
        for i in range(m):
            if time.time() > deadline:
                break
            old_mask = selected[i]
            
            # Precompute loss for old_mask
            loss = 0
            for j in range(m):
                if j != i and is_comparable(old_mask, selected[j]):
                    loss += 1
            
            best_new_mask = -1
            best_delta = 0
            
            # Try all possible new masks in the full space
            # Optimization: We can skip masks that are in selected_set (except old_mask)
            # But checking membership is O(1).
            
            for new_mask in range(1 << n):
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
            
            if best_new_mask != -1:
                # Accept the best swap found for this i
                current_score += best_delta
                selected[i] = best_new_mask
                selected_set.remove(old_mask)
                selected_set.add(best_new_mask)
                improved = True
                # Note: We continue to the next i. The 'loss' for subsequent i's will be recalculated.
        
        if not improved:
            # Local optimum reached for single-swap moves.
            # Try 2-swap moves? Or random restarts?
            # Given the time constraint, single-swap hill climbing is robust.
            # If we are stuck, we might try a random perturbation to escape local minima.
            if random.random() < 0.1:
                # Randomly swap two elements with random new masks to escape
                i1 = random.randint(0, m - 1)
                i2 = random.randint(0, m - 1)
                if i1 != i2:
                    m1 = random.randint(0, (1 << n) - 1)
                    m2 = random.randint(0, (1 << n) - 1)
                    
                    # Ensure distinctness
                    if m1 in selected_set and m1 != selected[i1]:
                        continue
                    if m2 in selected_set and m2 != selected[i2]:
                        continue
                    if m1 == m2:
                        continue
                    
                    # Calculate new score
                    old_score = current_score
                    # Remove i1, i2
                    # Add m1, m2
                    # This is complex to do incrementally without errors.
                    # Simpler: just do the swap, calculate score, and revert if worse.
                    
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
                    else:
                        # Revert
                        selected[i1] = old_m1
                        selected[i2] = old_m2
                        selected_set.remove(m1)
                        selected_set.remove(m2)
                        selected_set.add(old_m1)
                        selected_set.add(old_m2)
            
            if not improved:
                # If we've been stuck for a few iterations without improvement (even with random kicks),
                # we might be at a strong local optimum.
                # We can break early if we've done several full passes with no improvement.
                # For now, let's just break if no improvement in a full pass AND random kicks failed.
                # To avoid infinite loops of random kicks failing, we track consecutive failures.
                # Actually, the loop above breaks if `improved` is False.
                # But if random kick succeeds, `improved` becomes True, and we loop again.
                # If random kick fails, `improved` remains False, and we exit the while loop.
                # This might be too aggressive. Let's allow a few random kicks.
                
                # Let's modify the logic:
                # If no single-swap improvement, try a few random 2-swaps.
                # If none work, break.
                
                break

        if current_score > best_score:
            best_score = current_score
            best_config = selected[:]

    # Convert to lists of ints
    result = []
    for mask in best_config:
        s = [i for i in range(n) if (mask >> i) & 1]
        result.append(s)
        
    return result
