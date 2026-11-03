"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

def _score_set(U):
    """Compute the lower bound 1 + log(|U-U|/|U+U|)/log(2*max(U)+1)."""
    if not U:
        return 0.0
    U = list(U)
    if 0 not in U:
        U = U + [0]
    n = len(U)
    maxU = max(U)
    if maxU == 0:
        return 0.0
    
    # Compute sumset U+U
    sumset = set()
    diffset = set()
    for a in U:
        for b in U:
            sumset.add(a + b)
            diffset.add(a - b)
    
    n_sum = len(sumset)
    n_diff = len(diffset)
    
    if n_sum == 0:
        return 0.0
    
    # The score is 1 + log(n_diff / n_sum) / log(2*maxU + 1)
    # We want to maximize this.
    # Note: n_diff is always >= n_sum for sets containing 0? 
    # Actually, |U-U| >= |U+U| is not always true, but for sets with 0, 
    # U+U = {a+b}, U-U = {a-b}. 
    # If U has 0, then for every s in U+U, s = a+b. 
    # This doesn't immediately imply a bijection to U-U.
    # However, the formula is fixed.
    
    log_denom = math.log(2 * maxU + 1)
    if log_denom == 0:
        return 0.0
    
    ratio = n_diff / n_sum
    if ratio < 0:
        return 0.0
    if ratio == 0:
        return 0.0
        
    val = 1.0 + math.log(ratio) / log_denom
    return val

def _improve_set(U, max_iters=5000, deadline=None):
    """Local search to improve the set U."""
    U = list(U)
    if 0 not in U:
        U.append(0)
    U = sorted(set(U))
    
    best_score = _score_set(U)
    best_U = U[:]
    
    # Determine search range based on current max
    # We can add elements or remove elements (except 0)
    # Adding elements increases maxU, which increases the denominator log(2*maxU+1).
    # This hurts the score unless |U-U|/|U+U| increases significantly.
    # So we should be careful about adding large elements.
    
    # Strategy: 
    # 1. Try adding small elements that increase the ratio.
    # 2. Try removing elements that decrease the ratio.
    # 3. Try swapping elements.
    
    # To speed up, we can precompute or use sets.
    # Since U is small (likely < 50 in early stages), O(n^2) is fine.
    
    current_U = best_U[:]
    current_score = best_score
    
    # Define a helper to get the set of sums and diffs efficiently?
    # For now, recompute.
    
    def get_metrics(U):
        sumset = set()
        diffset = set()
        for a in U:
            for b in U:
                sumset.add(a + b)
                diffset.add(a - b)
        return len(sumset), len(diffset), max(U)
    
    n_sum, n_diff, maxU = get_metrics(current_U)
    
    # Candidate pool for addition
    # Elements up to 2*maxU + 10?
    # If we add x, new max is max(maxU, x).
    # New sumset size <= n_sum + 2*len(U) + 1
    # New diffset size <= n_diff + 2*len(U) + 1
    
    # Let's try a simple hill climbing.
    # Moves:
    # 1. Add a random integer from 1 to maxU + 10
    # 2. Remove a random non-zero element
    # 3. Replace a random non-zero element with another random integer
    
    # To avoid timeout, we check time.
    start_time = time.time()
    stall_count = 0
    max_stall = 100
    
    while stall_count < max_stall:
        if deadline and time.time() > deadline - 5:
            break
            
        # Random move
        move_type = random.random()
        
        new_U = current_U[:]
        
        if move_type < 0.3:
            # Add element
            # Range: 1 to maxU + 20
            upper = max(maxU + 20, 10)
            x = random.randint(1, upper)
            if x not in new_U:
                new_U.append(x)
            else:
                continue
        elif move_type < 0.6:
            # Remove element
            if len(new_U) <= 2:
                continue
            # Remove a non-zero element
            non_zero = [i for i, v in enumerate(new_U) if v != 0]
            if not non_zero:
                continue
            idx = random.choice(non_zero)
            new_U.pop(idx)
        else:
            # Replace element
            if len(new_U) <= 2:
                continue
            non_zero = [i for i, v in enumerate(new_U) if v != 0]
            if not non_zero:
                continue
            idx = random.choice(non_zero)
            old_val = new_U[idx]
            upper = max(maxU + 20, 10)
            new_val = random.randint(1, upper)
            if new_val not in new_U:
                new_U[idx] = new_val
            else:
                continue
        
        new_U = sorted(set(new_U))
        
        # Evaluate
        n_sum_new, n_diff_new, maxU_new = get_metrics(new_U)
        
        if n_sum_new == 0:
            continue
            
        log_denom_new = math.log(2 * maxU_new + 1)
        if log_denom_new == 0:
            continue
            
        ratio_new = n_diff_new / n_sum_new
        if ratio_new <= 0:
            continue
            
        score_new = 1.0 + math.log(ratio_new) / log_denom_new
        
        if score_new > current_score:
            current_U = new_U
            current_score = score_new
            stall_count = 0
            
            if current_score > best_score:
                best_score = current_score
                best_U = current_U[:]
                n_sum, n_diff, maxU = n_sum_new, n_diff_new, maxU_new
        else:
            stall_count += 1
    
    return best_U, best_score

def construct():
    """
    Construct a finite set U of nonnegative integers containing 0 to maximize
    the sumset/difference-set exponent lower bound.
    """
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Initial set
    # Start with a small set that has a good ratio
    # [0, 1, 3] gives:
    # U+U = {0, 1, 2, 3, 4, 6} -> size 6
    # U-U = {-3, -2, -1, 0, 1, 2, 3} -> size 7
    # maxU = 3, log(7)
    # ratio = 7/6
    # score = 1 + log(7/6)/log(7)
    
    # Let's try to find a better initial seed via small exhaustive search
    # or use a known good set.
    
    # Let's run local search from a few different starts
    best_U = [0, 1, 3]
    best_score = _score_set(best_U)
    
    # Try a few random starts
    for _ in range(20):
        if time.time() > deadline - 10:
            break
            
        # Random set of size 5-15
        size = random.randint(5, 15)
        # Elements from 0 to 50
        U = set([0])
        for _ in range(size - 1):
            U.add(random.randint(1, 50))
        U = list(U)
        
        # Improve
        U_improved, score_improved = _improve_set(U, max_iters=200, deadline=deadline)
        
        if score_improved > best_score:
            best_score = score_improved
            best_U = U_improved
    
    # One more polish on the best found
    if time.time() < deadline - 5:
        best_U, best_score = _improve_set(best_U, max_iters=5000, deadline=deadline)
    
    return best_U
# EVOLVE-BLOCK-END
