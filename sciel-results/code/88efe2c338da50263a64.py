import time
from math import log, floor
import random
import numpy as np

def score_set(U):
    """Compute the score for a set U."""
    U = sorted(set(U))
    if 0 not in U:
        return -1
    n = len(U)
    if n < 2:
        return -1
    
    # Use numpy for vectorized computation
    arr = np.array(U, dtype=np.int64)
    
    # Compute all pairwise sums (i<=j) and differences (i<j)
    # Sums: U[i] + U[j] for i <= j
    # Diffs: U[j] - U[i] for i < j (and negatives)
    
    # For efficiency with small n, use broadcasting
    # n is at most ~30-50, so n^2 is at most 2500
    sums_matrix = arr[:, None] + arr[None, :]
    diffs_matrix = arr[:, None] - arr[None, :]
    
    sum_count = len(np.unique(sums_matrix))
    diff_count = len(np.unique(diffs_matrix))
    
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

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit

    # Start with some good seed sets
    seed_sets = [
        [0, 1, 4, 9, 16, 25, 36, 49, 64, 81],  # Squares
        [0, 1, 2, 4, 8, 16, 32, 64, 128],  # Powers of 2
        [0, 1, 3, 9, 27, 81, 243],  # Powers of 3
        [0, 1, 3, 7, 15, 31, 63, 127],  # Mersenne-like
        [0, 1, 4, 10, 20, 35, 56, 84],  # Triangular
    ]
    
    best_set = list(seed_sets[0])
    best_score = score_set(best_set)
    
    # Evaluate all seeds
    for s in seed_sets:
        sc = score_set(s)
        if sc > best_score:
            best_score = sc
            best_set = list(s)
    
    # Local search with multiple strategies
    def local_search(U, deadline, max_val_limit=1000, max_n_limit=40):
        U = sorted(set(U))
        current_score = score_set(U)
        best_local_score = current_score
        best_local_set = list(U)
        
        no_improvement_count = 0
        max_no_improvement = 50
        
        while time.time() < deadline - 2:
            if no_improvement_count > max_no_improvement:
                break
                
            improved = False
            move_type = random.random()
            
            if move_type < 0.4:
                # Try to ADD an element
                if len(U) < max_n_limit:
                    current_max = max(U)
                    # Generate candidates around existing structure
                    candidates = set()
                    for a in U:
                        for b in U:
                            s = a + b
                            d = abs(a - b)
                            if 0 < s <= max_val_limit and s not in U:
                                candidates.add(s)
                            if 0 < d <= max_val_limit and d not in U:
                                candidates.add(d)
                            # Also try nearby values
                            for delta in range(-5, 6):
                                ns = s + delta
                                if 0 < ns <= max_val_limit and ns not in U:
                                    candidates.add(ns)
                                nd = d + delta
                                if 0 < nd <= max_val_limit and nd not in U:
                                    candidates.add(nd)
                    
                    if candidates:
                        cand_list = list(candidates)
                        random.shuffle(cand_list)
                        for c in cand_list[:20]:
                            new_U = sorted(U + [c])
                            new_score = score_set(new_U)
                            if new_score > current_score:
                                U = new_U
                                current_score = new_score
                                improved = True
                                break
            elif move_type < 0.6:
                # Try to REMOVE an element
                if len(U) > 2:
                    removable = [x for x in U if x != 0]
                    if removable:
                        c = random.choice(removable)
                        new_U = [x for x in U if x != c]
                        new_score = score_set(new_U)
                        if new_score > current_score:
                            U = new_U
                            current_score = new_score
                            improved = True
            else:
                # Try to REPLACE an element
                if len(U) > 1:
                    idx = random.randint(1, len(U) - 1)
                    old_val = U[idx]
                    # Try nearby values
                    for delta in range(-15, 16):
                        if delta == 0:
                            continue
                        new_val = old_val + delta
                        if new_val <= 0 or new_val in U:
                            continue
                        new_U = U.copy()
                        new_U[idx] = new_val
                        new_U.sort()
                        new_score = score_set(new_U)
                        if new_score > current_score:
                            U = new_U
                            current_score = new_score
                            improved = True
                            break
            
            if improved:
                no_improvement_count = 0
                if current_score > best_local_score:
                    best_local_score = current_score
                    best_local_set = list(U)
            else:
                no_improvement_count += 1
                
        return best_local_score, best_local_set
    
    # Run local search with multiple restarts
    restarts = 0
    while time.time() < deadline - 15:
        restarts += 1
        # Use best set or a random seed
        if restarts % 3 == 0:
            base = random.choice(seed_sets)
        else:
            base = list(best_set)
        
        local_score, local_set = local_search(base, deadline)
        
        if local_score > best_score:
            best_score = local_score
            best_set = local_set
    
    # Final polish: fine-grained local search
    if time.time() < deadline - 2:
        current = list(best_set)
        current_score = best_score
        improved = True
        iteration = 0
        while improved and time.time() < deadline - 1 and iteration < 300:
            improved = False
            iteration += 1
            n = len(current)
            
            # Try replacing each element with nearby values
            for i in range(1, n):
                if time.time() > deadline - 1:
                    break
                old_val = current[i]
                for delta in range(-8, 9):
                    if delta == 0:
                        continue
                    new_val = old_val + delta
                    if new_val <= 0 or new_val in current:
                        continue
                    new_set = [current[j] if j != i else new_val for j in range(n)]
                    new_set = sorted(set(new_set))
                    if len(new_set) != n:
                        continue
                    new_score = score_set(new_set)
                    if new_score > current_score:
                        current = new_set
                        current_score = new_score
                        improved = True
                        break
                if improved:
                    break
            
            # Try adding an element
            if not improved and n < 25 and time.time() < deadline - 1:
                max_val = max(current)
                for new_val in range(max(1, max_val - 15), max_val + 16):
                    if new_val in current:
                        continue
                    new_set = sorted(current + [new_val])
                    new_score = score_set(new_set)
                    if new_score > current_score:
                        current = new_set
                        current_score = new_score
                        improved = True
                        break
            
            # Try removing an element
            if not improved and n > 3 and time.time() < deadline - 1:
                for i in range(1, n):
                    new_set = [current[j] for j in range(n) if j != i]
                    new_score = score_set(new_set)
                    if new_score > current_score:
                        current = new_set
                        current_score = new_score
                        improved = True
                        break
        
        if current_score > best_score:
            best_score = current_score
            best_set = current
    
    return best_set

if __name__ == "__main__":
    U = construct()
    print("Result:", U)
    print("Score:", score_set(U))
