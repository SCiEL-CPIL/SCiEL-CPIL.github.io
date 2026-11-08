import time
from math import log, floor
import random
import numpy as np
from itertools import combinations

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
    
    # Compute all pairwise sums and differences
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
    deadline = start_time + 145  # Safety margin before 150s limit

    # Phase 1: Exhaustive search over small sets (n=3..6) to find a strong seed
    # This is inspired by the brute-force approach that found good small sets
    best_score = -1
    best_set = [0, 1]
    
    max_val_small = 60
    max_n_small = 6
    
    for n in range(3, max_n_small + 1):
        range_vals = range(1, max_val_small + 1)
        for comb in combinations(range_vals, n - 1):
            U = [0] + list(comb)
            s = score_set(U)
            if s > best_score:
                best_score = s
                best_set = list(U)
    
    # Also try some structured sets
    structured_sets = [
        [0, 1, 3, 7],
        [0, 1, 3, 8],
        [0, 1, 3, 9],
        [0, 1, 4, 9],
        [0, 1, 4, 16],
        [0, 1, 2, 7],
        [0, 1, 3, 7, 15],
        [0, 1, 3, 8, 16],
        [0, 1, 4, 9, 16],
        [0, 1, 2, 4, 8],
        [0, 1, 2, 4, 9],
        [0, 1, 3, 6, 10],
        [0, 1, 4, 9, 16, 25],
        [0, 1, 3, 7, 15, 31],
    ]
    for s_set in structured_sets:
        s = score_set(s_set)
        if s > best_score:
            best_score = s
            best_set = list(s_set)
    
    # Phase 2: Local search with multiple strategies to expand/refine
    # Inspired by the dynamic local search with candidate generation
    
    def generate_candidates(U, limit_val):
        """Generate candidate elements for addition based on sums and differences."""
        U_set = set(U)
        candidates = set()
        for a in U:
            for b in U:
                s = a + b
                d = abs(a - b)
                if 0 < s <= limit_val and s not in U_set:
                    candidates.add(s)
                if 0 < d <= limit_val and d not in U_set:
                    candidates.add(d)
        return list(candidates)
    
    def local_search(U, deadline, max_val_limit=500, max_n_limit=25):
        """Local search with add/remove/replace/double-swap moves."""
        U = sorted(set(U))
        current_score = score_set(U)
        best_local_score = current_score
        best_local_set = list(U)
        
        no_improvement_count = 0
        max_no_improvement = 60
        
        candidates = generate_candidates(U, max_val_limit)
        
        while time.time() < deadline - 5:
            if no_improvement_count > max_no_improvement:
                break
            
            improved = False
            move_type = random.random()
            
            # Refresh candidates periodically
            if not candidates or random.random() < 0.03:
                current_max = max(U) if U else 0
                new_limit = max(max_val_limit, int(current_max * 1.5))
                candidates = generate_candidates(U, new_limit)
            
            if move_type < 0.45:
                # ADD an element
                if len(U) < max_n_limit and candidates:
                    random.shuffle(candidates)
                    for c in candidates[:20]:
                        if c in U:
                            continue
                        new_U = sorted(U + [c])
                        new_score = score_set(new_U)
                        if new_score > current_score:
                            U = new_U
                            current_score = new_score
                            improved = True
                            candidates = generate_candidates(U, max(max_val_limit, max(U) * 2))
                            break
            elif move_type < 0.6:
                # REMOVE an element
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
                            candidates = generate_candidates(U, max_val_limit)
            elif move_type < 0.8:
                # REPLACE an element
                if len(U) > 1 and candidates:
                    idx = random.randint(1, len(U) - 1)
                    old_val = U[idx]
                    random.shuffle(candidates)
                    for new_val in candidates[:20]:
                        if new_val == old_val or new_val in U:
                            continue
                        new_U = U.copy()
                        new_U[idx] = new_val
                        new_U.sort()
                        new_score = score_set(new_U)
                        if new_score > current_score:
                            U = new_U
                            current_score = new_score
                            improved = True
                            candidates = generate_candidates(U, max(max_val_limit, max(U) * 2))
                            break
            else:
                # DOUBLE SWAP: replace 2 elements simultaneously
                if len(U) > 3 and candidates:
                    idxs = random.sample(range(1, len(U)), 2)
                    random.shuffle(candidates)
                    for new_val1 in candidates[:10]:
                        if new_val1 in U:
                            continue
                        for new_val2 in candidates[:10]:
                            if new_val2 == new_val1 or new_val2 in U:
                                continue
                            new_U = U.copy()
                            new_U[idxs[0]] = new_val1
                            new_U[idxs[1]] = new_val2
                            new_U.sort()
                            if len(new_U) != len(U):
                                continue
                            new_score = score_set(new_U)
                            if new_score > current_score:
                                U = new_U
                                current_score = new_score
                                improved = True
                                candidates = generate_candidates(U, max(max_val_limit, max(U) * 2))
                                break
                        if improved:
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
    while time.time() < deadline - 20:
        restarts += 1
        
        if restarts % 4 == 0:
            # Occasionally restart from a random structured set
            base = random.choice(structured_sets) if structured_sets else list(best_set)
        else:
            base = list(best_set)
        
        local_score, local_set = local_search(base, deadline)
        
        if local_score > best_score:
            best_score = local_score
            best_set = local_set
    
    # Phase 3: Fine-grained polish
    if time.time() < deadline - 3:
        current = list(best_set)
        current_score = best_score
        improved = True
        iteration = 0
        while improved and time.time() < deadline - 1 and iteration < 200:
            improved = False
            iteration += 1
            n = len(current)
            
            # Try replacing each element with nearby values
            for i in range(1, n):
                if time.time() > deadline - 1:
                    break
                old_val = current[i]
                for delta in range(-10, 11):
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
            
            # Try adding an element near the max
            if not improved and n < 20 and time.time() < deadline - 1:
                max_val = max(current)
                for new_val in range(max(1, max_val - 10), max_val + 11):
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
