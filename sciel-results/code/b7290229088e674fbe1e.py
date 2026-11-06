def construct():
    import time
    from fractions import Fraction
    
    start_time = time.time()
    deadline = start_time + 50
    
    denoms = list(range(2, 49))
    n = len(denoms)
    fracs = [Fraction(1, d) for d in denoms]
    
    def local_search(initial_selected, initial_sum):
        selected = set(initial_selected)
        current_sum = initial_sum
        best_sum = current_sum
        best_selected = set(selected)
        
        improved = True
        while improved:
            improved = False
            if time.time() > deadline:
                break
            
            best_move = None
            best_new_sum = current_sum
            best_new_selected = None
            
            # 1. Try Additions
            for j in range(n):
                if j in selected:
                    continue
                new_sum = current_sum + fracs[j]
                if new_sum < 1 and new_sum > best_new_sum:
                    best_new_sum = new_sum
                    best_move = ('add', j)
                    best_new_selected = selected | {j}
            
            # 2. Try Swaps
            if best_new_sum <= current_sum: # Only look for swaps if no addition improved it? 
                # Actually, a swap might be better than an addition? 
                # Addition increases sum by 1/d. Swap increases by 1/d_j - 1/d_i.
                # Since 1/d_j - 1/d_i < 1/d_j, addition is always a larger gain if possible.
                # So if an addition is possible, it's always better than any swap?
                # Not necessarily. An addition might block future better moves? 
                # But we are doing hill climbing on the sum itself. 
                # If we can add, we should add, because it strictly increases the sum.
                # So we only need to check swaps if NO addition is possible.
                pass
            
            if best_move is None:
                # No addition possible. Try swaps.
                for i in selected:
                    for j in range(n):
                        if j in selected:
                            continue
                        new_sum = current_sum - fracs[i] + fracs[j]
                        if new_sum < 1 and new_sum > best_new_sum:
                            best_new_sum = new_sum
                            best_move = ('swap', i, j)
                            best_new_selected = (selected - {i}) | {j}
            
            if best_move is not None:
                selected = best_new_selected
                current_sum = best_new_sum
                improved = True
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_selected = set(selected)
            else:
                # Local optimum reached
                break
                
        return best_sum, best_selected

    # Strategy: Run local search from multiple starting points
    # 1. Greedy Largest First
    sel_greedy = set()
    sum_greedy = Fraction(0)
    for i in range(n):
        if sum_greedy + fracs[i] < 1:
            sum_greedy += fracs[i]
            sel_greedy.add(i)
    
    best_overall_sum, best_overall_sel = local_search(sel_greedy, sum_greedy)
    
    # 2. Random Restarts
    import random
    random.seed(42) # Deterministic for reproducibility in eval? Usually good.
    
    # We have ~40-50s left.
    # Each local search is very fast (N=47, swaps are 47*47 ~ 2200 checks, additions 47).
    # We can do many restarts.
    
    while time.time() < deadline - 5:
        # Random initialization
        # Pick a random subset? 
        # Better: Pick a random "seed" subset and let local search fix it.
        # Or just random 0/1 assignment.
        rand_sel = set()
        rand_sum = Fraction(0)
        for i in range(n):
            if random.random() < 0.5:
                # Only add if it doesn't exceed 1? 
                # If we start with a valid subset, local search is simpler.
                # If we start with invalid, we need to fix it.
                # Let's start with valid subsets.
                if rand_sum + fracs[i] < 1:
                    rand_sel.add(i)
                    rand_sum += fracs[i]
        
        s, sel = local_search(rand_sel, rand_sum)
        if s > best_overall_sum:
            best_overall_sum = s
            best_overall_sel = sel
            
    return list(best_overall_sel)
