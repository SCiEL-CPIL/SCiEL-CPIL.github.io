import time
import random
import math

def _score_mask(M):
    """
    Calculate the number of distinct d > 0 such that there exists a
    with a, a+d, a+2d in the set represented by bitmask M.
    M is an integer where bit i is 1 if i is in the set.
    Indices 1..512 are used.
    """
    count = 0
    for d in range(1, 256):
        if (M & (M >> d) & (M >> (2 * d))) != 0:
            count += 1
    return count

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    N = 32
    MAXV = 512
    
    # --- Phase 1: Greedy Initialization ---
    # Start with empty set
    current_mask = 0
    current_set = set()
    
    # We want to add 32 elements
    for _ in range(N):
        best_x = -1
        best_score = -1
        
        # Try all possible x not in current_set
        # Optimization: Iterate 1..512, skip if in set
        for x in range(1, MAXV + 1):
            if x in current_set:
                continue
            
            new_mask = current_mask | (1 << x)
            # Score calculation
            # We can optimize this by noting that we only need to check
            # if the score increased, but full score is needed for comparison
            # since we are maximizing total score.
            sc = _score_mask(new_mask)
            
            if sc > best_score:
                best_score = sc
                best_x = x
        
        if best_x != -1:
            current_set.add(best_x)
            current_mask |= (1 << best_x)
            
    current_score = best_score
    best_mask = current_mask
    best_score = current_score
    best_set = set(current_set)
    
    # Check time
    if time.time() > deadline - 10.0:
         return sorted(list(best_set))

    # --- Phase 2: Simulated Annealing ---
    T = 10.0
    T_min = 0.1
    # We will use time-based cooling to adapt to hardware speed
    # Or step-based. Let's use time-based for robustness.
    # We want T to drop from 10 to 0.1 over the available time.
    
    sa_deadline = deadline - 5.0 # Leave 5s for LS
    
    # Estimate steps per second to calibrate cooling?
    # Or just use a fixed cooling rate per step, and run until time limit.
    # Fixed cooling rate is risky if hardware is slow/fast.
    # Let's use a hybrid: run SA until sa_deadline.
    # Use a cooling factor that ensures we don't get stuck in high T too long
    # or drop too fast.
    
    # A common heuristic: T = T_start * exp(-alpha * time_elapsed)
    # Or simply T *= cooling_factor each step.
    # If we do ~10k steps/sec, and have 100s, that's 1M steps.
    # 0.9995^1000000 is very small.
    # Let's use time-based cooling.
    
    # To make it robust, let's run SA for a fixed number of steps OR until time limit,
    # but adjust T based on progress.
    
    # Let's stick to step-based but estimate steps.
    # Actually, simpler: Just run the loop, decrement T by a small amount each step.
    # If we do 1M steps, and start T=10, end T=0.1.
    # Ratio 100. 100^(1/1000000) is ~1.000005.
    # So cooling = 0.999995.
    
    # Let's try a different approach:
    # Run SA for 50% of the time, then LS for the rest?
    # No, LS is expensive (O(N * MAXV * ScoreCost)).
    # LS: 32 * 480 swaps * 255 ops = ~4M ops per full pass.
    # If we do 10 passes, that's 40M ops. Might take 5-10 seconds.
    # So LS is significant.
    
    # Let's allocate 80% time to SA, 20% to LS.
    total_time = 115.0
    sa_time_limit = total_time * 0.8
    ls_time_limit = total_time * 0.2
    
    sa_end_time = start_time + sa_time_limit
    
    # Use a cooling factor that depends on estimated steps
    # Or just use a very small cooling factor and let it run.
    # If T drops too low, we stop improving.
    # Let's use T = T_start * (T_end/T_start)^(elapsed/sa_time_limit)
    
    while time.time() < sa_end_time:
        # Pick random element to remove
        # Using list(current_set) is O(N). N=32.
        # Picking random index is O(1).
        idx = random.randint(0, N - 1)
        pos_remove = list(current_set)[idx]
        
        # Pick random element to add
        # Expected ~1.07 tries
        while True:
            pos_add = random.randint(1, MAXV)
            if pos_add not in current_set:
                break
        
        if pos_remove == pos_add:
            continue
        
        # Apply move
        new_mask = current_mask
        new_mask &= ~(1 << pos_remove)
        new_mask |= (1 << pos_add)
        
        new_score = _score_mask(new_mask)
        delta = new_score - current_score
        
        # Calculate T
        elapsed = time.time() - start_time
        progress = elapsed / sa_time_limit
        if progress > 1.0:
            progress = 1.0
        
        T = 10.0 * (0.1 / 10.0) ** progress
        
        # Acceptance criterion
        if delta > 0 or random.random() < math.exp(delta / T):
            current_mask = new_mask
            current_set.remove(pos_remove)
            current_set.add(pos_add)
            current_score = new_score
            
            if current_score > best_score:
                best_score = current_score
                best_mask = current_mask
                best_set = set(current_set)
        
    # --- Phase 3: Local Search Polish ---
    # Use remaining time to polish the best solution found.
    # Iterative best-improvement local search.
    
    # Work on a copy of the best solution
    ls_mask = best_mask
    ls_score = best_score
    ls_set = set(best_set)
    
    ls_deadline = deadline
    
    improved = True
    while improved and time.time() < ls_deadline:
        improved = False
        best_new_mask = ls_mask
        best_new_score = ls_score
        best_new_set = set(ls_set)
        
        # Iterate over all possible swaps
        # To speed up, we can iterate over elements in ls_set and potential additions
        # But checking all 32 * 480 swaps is ~15k checks.
        # Each check is ~255 ops. Total ~4M ops.
        # This should take < 1 second.
        
        for rem in ls_set:
            if time.time() > ls_deadline:
                break
                
            rem_bit = 1 << rem
            base_mask = ls_mask & ~rem_bit
            
            for add in range(1, MAXV + 1):
                if add in ls_set:
                    continue
                
                add_bit = 1 << add
                cand_mask = base_mask | add_bit
                cand_score = _score_mask(cand_mask)
                
                if cand_score > best_new_score:
                    best_new_score = cand_score
                    best_new_mask = cand_mask
                    best_new_set = set(ls_set)
                    best_new_set.remove(rem)
                    best_new_set.add(add)
                    improved = True
                    # We don't break, we find the BEST neighbor in this round
                    # to ensure steady progress and avoid getting stuck in shallow local minima
                    # due to order of checks.
        
        if improved:
            ls_mask = best_new_mask
            ls_score = best_new_score
            ls_set = best_new_set

    # Final check
    if ls_score > best_score:
        best_score = ls_score
        best_set = ls_set

    result = sorted(list(best_set))
    return result
