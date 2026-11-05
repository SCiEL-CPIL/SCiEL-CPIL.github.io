import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    n = 128
    min_sum = 128
    max_sum = 256
    
    def get_cost(A):
        A = sorted(A)
        m = len(A)
        if m == 0:
            return (0, 129, 0)
        
        counts = [0] * 257
        
        for i in range(m):
            ai = A[i]
            s_ii = ai + ai
            if min_sum <= s_ii <= max_sum:
                counts[s_ii] += 1
            
            for j in range(i + 1, m):
                s_ij = ai + A[j]
                if min_sum <= s_ij <= max_sum:
                    counts[s_ij] += 1
        
        unc = 0
        peak = 0
        for s in range(min_sum, max_sum + 1):
            c = counts[s]
            if c == 0:
                unc += 1
            elif c > peak:
                peak = c
        
        return (peak, unc, m)

    def score_tuple(A):
        peak, unc, m = get_cost(A)
        if unc > 0:
             return (1, unc, peak, m)
        else:
             return (0, 0, peak, m)

    best_score = (1, 129, 0, 0)
    best_set = []
    
    # Phase 1: Targeted Greedy
    A = []
    
    # We need to cover all sums.
    # Start by picking elements that cover many sums.
    # Randomize the order of sums to avoid bias?
    # Or just iterate.
    
    # Let's try a few random restarts of greedy
    for restart in range(10):
        if time.time() > deadline - 20:
            break
        random.seed(42 + restart)
        A = []
        
        # To break symmetry, we can pick a random first element?
        # Or just start empty.
        
        for _ in range(50):
            current_score = score_tuple(A)
            if current_score[0] == 0:
                break
            
            # Find uncovered sums
            # We need counts to know which are uncovered.
            # get_cost doesn't return counts. Let's modify it or re-calc.
            # For speed, let's just try all candidates.
            
            best_next = None
            best_next_score = current_score
            
            # Optimization: Only try candidates that might help.
            # If we have uncovered sum S, we need a+b=S.
            # If A is small, most a,b are not in A.
            
            for x in range(0, 129):
                if x in A:
                    continue
                new_A = A + [x]
                s = score_tuple(new_A)
                if s < best_next_score:
                    best_next_score = s
                    best_next = new_A
            
            if best_next is None:
                break
            A = best_next
            
        s_final = score_tuple(A)
        if s_final < best_score:
            best_score = s_final
            best_set = A.copy()
            
    # If we still don't have a valid solution, use the seed as fallback
    if best_score[0] != 0:
        A_seed = list(range(64, 129))
        s_seed = score_tuple(A_seed)
        if s_seed < best_score:
            best_score = s_seed
            best_set = A_seed.copy()
            
    # Phase 2: Local Search
    current_A = best_set.copy()
    current_score = best_score
    
    last_improvement_time = time.time()
    
    while time.time() < deadline - 2:
        improved = False
        
        if current_score[0] != 0:
            # Invalid: Try to fix coverage
            # Get counts to find uncovered sums
            # Re-calc counts
            counts = [0] * 257
            A_sorted = sorted(current_A)
            m = len(A_sorted)
            for i in range(m):
                ai = A_sorted[i]
                s_ii = ai + ai
                if min_sum <= s_ii <= max_sum:
                    counts[s_ii] += 1
                for j in range(i + 1, m):
                    s_ij = ai + A_sorted[j]
                    if min_sum <= s_ij <= max_sum:
                        counts[s_ij] += 1
            
            uncovered = [s for s in range(min_sum, max_sum + 1) if counts[s] == 0]
            
            if uncovered:
                target_s = random.choice(uncovered)
                # Find a, b such that a+b = target_s
                # Try adding a or b
                # We can try all a in [0, 128]
                # But that's 129 trials.
                # Let's pick a random a, b = target_s - a.
                # If a in [0,128] and b in [0,128], try adding a or b.
                
                # Try a few random decompositions
                for _ in range(5):
                    a = random.randint(0, 128)
                    b = target_s - a
                    if 0 <= b <= 128:
                        # Try adding a
                        if a not in current_A:
                            trial_A = current_A + [a]
                            s = score_tuple(trial_A)
                            if s < current_score:
                                current_A = trial_A
                                current_score = s
                                improved = True
                                last_improvement_time = time.time()
                                if s < best_score:
                                    best_score = s
                                    best_set = current_A.copy()
                                break
                        # Try adding b
                        if b not in current_A:
                            trial_A = current_A + [b]
                            s = score_tuple(trial_A)
                            if s < current_score:
                                current_A = trial_A
                                current_score = s
                                improved = True
                                last_improvement_time = time.time()
                                if s < best_score:
                                    best_score = s
                                    best_set = current_A.copy()
                                break
        
        else:
            # Valid: Try to reduce peak
            # Random move
            if len(current_A) > 0:
                idx = random.randint(0, len(current_A) - 1)
                old_val = current_A[idx]
                new_val = random.randint(0, 128)
                
                if new_val != old_val and new_val not in current_A:
                    trial_A = current_A.copy()
                    trial_A[idx] = new_val
                    s = score_tuple(trial_A)
                    
                    if s < current_score:
                        current_A = trial_A
                        current_score = s
                        improved = True
                        last_improvement_time = time.time()
                        if s < best_score:
                            best_score = s
                            best_set = current_A.copy()
                
                # Try removing
                if len(current_A) > 1:
                    idx = random.randint(0, len(current_A) - 1)
                    trial_A = current_A[:idx] + current_A[idx+1:]
                    s = score_tuple(trial_A)
                    if s < current_score:
                        current_A = trial_A
                        current_score = s
                        improved = True
                        last_improvement_time = time.time()
                        if s < best_score:
                            best_score = s
                            best_set = current_A.copy()
        
        if not improved and time.time() - last_improvement_time > 3.0:
            # Stuck, reset to best and try a different random seed for the next phase?
            # Or just continue.
            pass

    return best_set
