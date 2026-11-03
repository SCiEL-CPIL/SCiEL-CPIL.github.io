import time
import random
from math import log
import numpy as np

def fast_score(U):
    """
    Calculate the score for a set U.
    Score = 1 + log(|U-U|/|U+U|) / log(2*max(U)+1)
    """
    U = sorted(set(U))
    if 0 not in U:
        return -1
    n = len(U)
    if n < 2:
        return 0.0
    
    U_arr = np.array(U)
    
    # Compute all pairwise sums (i <= j)
    i_idx, j_idx = np.triu_indices(n)
    sums = U_arr[i_idx] + U_arr[j_idx]
    num_sums = len(np.unique(sums))
    
    # Compute all pairwise differences (i < j)
    i_idx_d, j_idx_d = np.triu_indices(n, k=1)
    diffs = U_arr[j_idx_d] - U_arr[i_idx_d]
    # Unique positive diffs
    num_pos_diffs = len(np.unique(diffs))
    # Total diffs = 2 * num_pos_diffs + 1 (for 0)
    num_diffs = 2 * num_pos_diffs + 1
    
    ratio = num_diffs / num_sums
    if ratio <= 1:
        return -1
    max_u = max(U)
    D = 2 * max_u + 1
    if D <= 1:
        return -1
    return 1 + log(ratio) / log(D)

def local_search(U, deadline, max_iter=500):
    """
    Local search to improve the set U.
    """
    current = sorted(set(U))
    current_score = fast_score(current)
    if current_score < 0:
        return current, current_score
    
    best = current
    best_score = current_score
    
    start_time = time.time()
    
    for _ in range(max_iter):
        if time.time() > deadline:
            break
            
        n = len(current)
        if n < 2:
            break
            
        # Move: perturb one element
        i = random.randint(1, n-1)
        old_val = current[i]
        max_val = max(current)
        scale = max(1, int(max_val * 0.2))
        new_val = old_val + random.randint(-scale, scale)
        
        if new_val <= 0:
            new_val = old_val + random.randint(1, scale)
            
        new_set = current[:]
        new_set[i] = new_val
        new_set = sorted(set(new_set))
        
        # If size changed, we might want to try adding/removing separately
        # For now, if size changed, skip or accept?
        # Let's stick to fixed size for perturb to keep it simple.
        if len(new_set) != n:
            continue
            
        new_score = fast_score(new_set)
        
        if new_score > current_score:
            current = new_set
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best = current
        # Simple hill climbing, no acceptance of worse moves to save time
        # Or accept with small prob?
        # Given short time, pure hill climbing on perturb is fast.
        
    return best, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    # Candidate sets
    candidates = [
        [0, 1, 3],
        [0, 1, 4, 6],
        [0, 1, 4, 9, 11],
        [0, 1, 4, 10, 12, 17],
        [0, 1, 4, 10, 18, 23, 25],
        [0, 1, 3, 7, 10],
        [0, 2, 5, 11, 15], # Random guess
    ]
    
    # Generate some random sparse sets
    random.seed(42)
    for _ in range(50):
        n_elem = random.randint(3, 6)
        max_val = random.randint(10, 50)
        U = [0] + sorted(random.sample(range(1, max_val), n_elem - 1))
        candidates.append(U)
        
    best_set = [0, 1, 3]
    best_score = fast_score(best_set)
    
    # Evaluate candidates
    scored_candidates = []
    for U in candidates:
        s = fast_score(U)
        if s > best_score:
            best_score = s
            best_set = U
        if s > 0:
            scored_candidates.append((s, U))
            
    scored_candidates.sort(reverse=True)
    
    # Run local search on top candidates
    for score, U in scored_candidates[:10]:
        if time.time() > deadline - 10:
            break
        refined, refined_score = local_search(U, deadline, max_iter=200)
        if refined_score > best_score:
            best_score = refined_score
            best_set = refined
            
    # Final polish on the best set
    if time.time() < deadline - 5:
        refined, refined_score = local_search(best_set, deadline, max_iter=1000)
        if refined_score > best_score:
            best_score = refined_score
            best_set = refined
            
    return best_set

# Test
if __name__ == "__main__":
    print(f"Best set: {construct()}")
