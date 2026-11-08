import time
from math import log
import random

def score_set(U):
    """
    Calculate the score for a set U.
    Score = 1 + log(|U-U| / |U+U|) / log(2 * max(U) + 1)
    """
    U = sorted(set(U))
    if 0 not in U:
        return -1
    n = len(U)
    if n < 2:
        return -1
    
    sums = set()
    diffs = set()
    
    # Optimization: Use lists for speed, then convert to sets
    # But for small n, sets are fine.
    for i in range(n):
        ui = U[i]
        for j in range(i, n):
            sums.add(ui + U[j])
        for j in range(i + 1, n):
            d = U[j] - ui
            if d != 0:
                diffs.add(d)
                diffs.add(-d)
    diffs.add(0)
    
    sum_count = len(sums)
    diff_count = len(diffs)
    
    if sum_count == 0:
        return -1
        
    ratio = diff_count / sum_count
    if ratio <= 1:
        return -1
        
    max_u = max(U)
    D = 2 * max_u + 1
    if D <= 1:
        return -1
        
    return 1 + log(ratio) / log(D)

def compute_sets(U):
    """
    Compute sumset and diffset for a sorted list U.
    Returns (sum_set, diff_set, sum_count, diff_count, max_u)
    """
    sums = set()
    diffs = set()
    n = len(U)
    for i in range(n):
        ui = U[i]
        for j in range(i, n):
            sums.add(ui + U[j])
        for j in range(i + 1, n):
            d = U[j] - ui
            if d != 0:
                diffs.add(d)
                diffs.add(-d)
    diffs.add(0)
    return sums, diffs, len(sums), len(diffs), max(U)

def greedy_construct(start_time, deadline):
    """
    Greedy construction: Start with [0], iteratively add the element
    that maximizes the score improvement, or more specifically,
    maximizes the ratio of new differences to new sums.
    """
    current = [0]
    current_sums, current_diffs, s_cnt, d_cnt, max_u = compute_sets(current)
    current_score = score_set(current)
    
    # We want to grow the set.
    # Candidate generation:
    # 1. Small integers (1, 2, 3...)
    # 2. Geometric progressions from current max
    # 3. Random integers
    
    best_score = current_score
    best_set = list(current)
    
    # Pre-define a range of potential candidates to explore
    # The "size" of the set matters. Let's aim for size 10-20.
    
    while time.time() < deadline - 5:
        n = len(current)
        if n > 20: # Cap size to keep computation fast and avoid dilution
            break
            
        # Generate candidates
        candidates = set()
        
        # 1. Small integers
        for i in range(1, 50):
            candidates.add(i)
            
        # 2. Based on current max
        max_val = max(current)
        for i in range(1, 10):
            candidates.add(max_val + i)
            if max_val > 1:
                candidates.add(max_val * i)
                
        # 3. Random
        for _ in range(50):
            candidates.add(random.randint(1, max_val * 2 + 10))
            
        best_candidate = None
        best_candidate_score = current_score
        best_candidate_sums = None
        best_candidate_diffs = None
        best_candidate_s_cnt = None
        best_candidate_d_cnt = None
        best_candidate_max = None
        
        # Evaluate candidates
        # To speed up, we can estimate the gain first, but exact is safer.
        # We will compute the full score for the top few, or just all if fast enough.
        
        # Optimization: We can update sums/diffs incrementally?
        # New sums: {x + u for x in current}
        # New diffs: {x - u, u - x for u in current}
        
        for c in candidates:
            if c in current:
                continue
                
            # Incremental update
            new_sums = set(current_sums)
            new_diffs = set(current_diffs)
            
            for u in current:
                new_sums.add(c + u)
                d = c - u
                new_diffs.add(d)
                new_diffs.add(-d)
                
            new_s_cnt = len(new_sums)
            new_d_cnt = len(new_diffs)
            
            if new_s_cnt == 0:
                continue
                
            new_max = max(max_u, c)
            D = 2 * new_max + 1
            if D <= 1:
                continue
                
            ratio = new_d_cnt / new_s_cnt
            if ratio <= 1:
                continue
                
            new_score = 1 + log(ratio) / log(D)
            
            if new_score > best_candidate_score:
                best_candidate_score = new_score
                best_candidate = c
                best_candidate_sums = new_sums
                best_candidate_diffs = new_diffs
                best_candidate_s_cnt = new_s_cnt
                best_candidate_d_cnt = new_d_cnt
                best_candidate_max = new_max
        
        if best_candidate is None:
            # No improvement found with current candidates
            # Try larger jumps or break
            if n < 5:
                # Force add a random larger number to escape local trap
                c = random.randint(max_u + 1, max_u * 2 + 10)
                current.append(c)
                current.sort()
                current_sums, current_diffs, s_cnt, d_cnt, max_u = compute_sets(current)
                current_score = score_set(current)
            else:
                break
        else:
            # Accept the best candidate
            current.append(best_candidate)
            current.sort()
            current_sums = best_candidate_sums
            current_diffs = best_candidate_diffs
            s_cnt = best_candidate_s_cnt
            d_cnt = best_candidate_d_cnt
            max_u = best_candidate_max
            current_score = best_candidate_score
            
            if current_score > best_score:
                best_score = current_score
                best_set = list(current)
                
        # If we are stuck for a while (no score improvement in last 5 steps), 
        # we might want to restart or perturb. 
        # For now, simple greedy is the core strategy.
        
    return best_set

def local_search_refine(U, start_time, deadline):
    """
    Refine the set found by greedy using local search.
    """
    current = list(U)
    current_score = score_set(current)
    best_set = list(current)
    best_score = current_score
    
    # Simple hill climbing with random restarts
    while time.time() < deadline - 2:
        improved = False
        n = len(current)
        
        # 1. Try shifting elements
        for i in range(1, n):
            if time.time() > deadline - 2:
                break
            old_val = current[i]
            for delta in range(-5, 6):
                if delta == 0:
                    continue
                new_val = old_val + delta
                if new_val <= 0 or new_val in current:
                    continue
                new_set = [current[j] if j != i else new_val for j in range(n)]
                new_set.sort()
                new_score = score_set(new_set)
                if new_score > current_score:
                    current = new_set
                    current_score = new_score
                    improved = True
                    if current_score > best_score:
                        best_score = current_score
                        best_set = list(current)
                    break # Restart loop with new current
            if improved:
                break
                
        # 2. Try adding an element
        if not improved and n < 25:
            max_val = max(current)
            for new_val in range(1, max_val + 10):
                if new_val in current:
                    continue
                new_set = sorted(current + [new_val])
                new_score = score_set(new_set)
                if new_score > current_score:
                    current = new_set
                    current_score = new_score
                    improved = True
                    if current_score > best_score:
                        best_score = current_score
                        best_set = list(current)
                    break
                    
        # 3. Try removing an element
        if not improved and n > 3:
            for i in range(1, n):
                new_set = [current[j] for j in range(n) if j != i]
                new_score = score_set(new_set)
                if new_score > current_score:
                    current = new_set
                    current_score = new_score
                    improved = True
                    if current_score > best_score:
                        best_score = current_score
                        best_set = list(current)
                    break
                    
        if not improved:
            # Local optimum reached. Try a random perturbation (kick)
            if random.random() < 0.5:
                # Shift a random element by a larger amount
                if n > 1:
                    idx = random.randint(1, n - 1)
                    delta = random.randint(-10, 10)
                    new_val = current[idx] + delta
                    if new_val > 0 and new_val not in current:
                        current[idx] = new_val
                        current.sort()
                        current_score = score_set(current)
                    else:
                        break # No valid kick, stop
            else:
                break
                
    return best_set

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    best_overall = None
    best_overall_score = -1
    
    # Run multiple greedy instances with different random seeds/strategies
    for _ in range(5):
        if time.time() > deadline - 20:
            break
            
        # Greedy phase
        g_set = greedy_construct(start_time, deadline - 10)
        g_score = score_set(g_set)
        
        if g_score > best_overall_score:
            best_overall_score = g_score
            best_overall = list(g_set)
            
        # Refine phase
        r_set = local_search_refine(g_set, start_time, deadline)
        r_score = score_set(r_set)
        
        if r_score > best_overall_score:
            best_overall_score = r_score
            best_overall = list(r_set)
            
    # If we have time, do a final global polish on the best set
    if time.time() < deadline - 5 and best_overall:
        final_set = local_search_refine(best_overall, start_time, deadline)
        final_score = score_set(final_set)
        if final_score > best_overall_score:
            best_overall = final_set
            best_overall_score = final_score
            
    if best_overall is None:
        # Fallback
        best_overall = [0, 1, 4, 9, 11]
        
    return best_overall

if __name__ == "__main__":
    U = construct()
    print("Result:", U)
    print("Score:", score_set(U))
