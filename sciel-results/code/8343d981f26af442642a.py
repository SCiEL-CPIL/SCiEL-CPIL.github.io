import math
import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110

    def _compute_score(U):
        if not U or 0 not in U:
            return 0.0
        U = list(set(U))
        n = len(U)
        if n < 2:
            return 0.0
        max_u = max(U)
        if max_u == 0:
            return 0.0
        
        sumset = set()
        for a in U:
            for b in U:
                sumset.add(a + b)
        
        diffset = set()
        for a in U:
            for b in U:
                diffset.add(a - b)
        
        num_sum = len(sumset)
        num_diff = len(diffset)
        
        if num_sum == 0:
            return 0.0
        
        ratio = num_diff / num_sum
        log_arg = 2 * max_u + 1
        if log_arg <= 1:
            return 0.0
        
        score = 1.0 + math.log(ratio) / math.log(log_arg)
        return score

    best_set = [0, 1, 3, 7, 12]
    best_score = _compute_score(best_set)
    
    # Known good candidates
    candidates = [
        [0, 1, 3],
        [0, 1, 3, 7],
        [0, 1, 3, 7, 12],
        [0, 1, 3, 7, 15],
        [0, 1, 3, 7, 12, 22],
        [0, 1, 3, 7, 12, 25],
        [0, 1, 3, 7, 12, 27],
        [0, 1, 4, 6, 10],
        [0, 1, 4, 6, 11],
        [0, 1, 4, 6, 12],
        [0, 1, 4, 9, 11],
        [0, 1, 5, 6, 10],
        [0, 2, 3, 7, 12],
        [0, 1, 3, 8, 12],
        [0, 1, 3, 9, 12],
        [0, 1, 4, 7, 12],
        [0, 1, 4, 8, 12],
        [0, 1, 5, 7, 12],
        [0, 2, 5, 7, 12],
    ]
    
    for cand in candidates:
        if time.time() > deadline:
            break
        score = _compute_score(cand)
        if score > best_score:
            best_score = score
            best_set = list(cand)
            
    # Local search
    def local_search(U, max_iter=200):
        U = list(U)
        current_score = _compute_score(U)
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            improved = False
            # Try adding an element
            max_u = max(U) if U else 0
            for e in range(1, max_u + 15):
                if e in U:
                    continue
                new_U = U + [e]
                s = _compute_score(new_U)
                if s > current_score + 1e-12:
                    U = new_U
                    current_score = s
                    improved = True
                    break
            if improved:
                continue
            # Try removing an element
            for i in range(len(U)):
                if U[i] == 0:
                    continue
                new_U = [x for j, x in enumerate(U) if j != i]
                if len(new_U) < 2:
                    continue
                s = _compute_score(new_U)
                if s > current_score + 1e-12:
                    U = new_U
                    current_score = s
                    improved = True
                    break
            # Try swapping
            if not improved:
                for i in range(len(U)):
                    if U[i] == 0:
                        continue
                    for j in range(len(U)):
                        if U[j] == 0:
                            continue
                        old_val = U[i]
                        U[i] = U[j]
                        U[j] = old_val
                        s = _compute_score(U)
                        if s > current_score + 1e-12:
                            current_score = s
                            improved = True
                            break
                        else:
                            U[i] = U[j]
                            U[j] = old_val
                    if improved:
                        break
            if not improved:
                break
        return U, current_score

    # Run local search on best found
    if time.time() < deadline:
        ls_set, ls_score = local_search(best_set, max_iter=100)
        if ls_score > best_score:
            best_score = ls_score
            best_set = list(ls_set)
            
    # Random restarts
    if time.time() < deadline:
        random.seed(42)
        for trial in range(100):
            if time.time() > deadline:
                break
            size = random.randint(3, 7)
            max_val = random.randint(10, 30)
            U = [0]
            for _ in range(size - 1):
                e = random.randint(1, max_val)
                if e not in U:
                    U.append(e)
            U.sort()
            ls_set, ls_score = local_search(U, max_iter=50)
            if ls_score > best_score:
                best_score = ls_score
                best_set = list(ls_set)
                
    return best_set
