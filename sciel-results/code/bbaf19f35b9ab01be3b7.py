import math
import random
import time

def _score_set_fast(U):
    U_set = set(U)
    n = len(U_set)
    if n == 0:
        return 0.0
    max_u = max(U_set)
    if max_u == 0:
        return 1.0
    
    sumset = set()
    diffset = set()
    U_list = list(U_set)
    for i in range(n):
        a = U_list[i]
        for j in range(n):
            b = U_list[j]
            sumset.add(a + b)
            diffset.add(a - b)
            
    sumset_size = len(sumset)
    diffset_size = len(diffset)
    
    if sumset_size == 0:
        return 0.0
        
    ratio = diffset_size / sumset_size
    if ratio <= 0:
        return 0.0
    
    log_ratio = math.log(ratio)
    log_denom = math.log(2 * max_u + 1)
    if log_denom == 0:
        return 1.0
    
    return 1.0 + log_ratio / log_denom

def _local_search(U, deadline, max_iters=100):
    U = sorted(set(U))
    if not U or U[0] != 0:
        U = [0] + U
    best_score = _score_set_fast(U)
    best_U = U[:]
    
    max_u = max(U)
    upper_bound = max_u + 20
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
            
        current_U = best_U[:]
        n = len(current_U)
        improved = False
        
        # Try removing elements
        for i in range(1, n):
            new_U = current_U[:i] + current_U[i+1:]
            s = _score_set_fast(new_U)
            if s > best_score + 1e-12:
                best_score = s
                best_U = new_U
                improved = True
                break
        
        # Try modifying elements
        if not improved and n > 1:
            for idx in range(1, n):
                old_val = current_U[idx]
                candidates = set()
                for d in range(-2, 3):
                    nv = old_val + d
                    if nv > 0:
                        candidates.add(nv)
                
                for new_val in candidates:
                    if new_val in current_U and new_val != old_val:
                        continue
                    new_U = current_U[:]
                    new_U[idx] = new_val
                    new_U = sorted(set(new_U))
                    if len(new_U) != len(current_U):
                        continue
                    s = _score_set_fast(new_U)
                    if s > best_score + 1e-12:
                        best_score = s
                        best_U = new_U
                        improved = True
                        break
                if improved:
                    break
        
        if not improved:
            break
            
    return best_U, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    # Hardcoded known good sets
    # The set U = {0, 1, 3} has score ~1.0
    # The set U = {0, 1, 4, 6} (Golomb ruler)
    # The set U = {0, 1, 4, 10, 11}
    
    # Let's try to find a set with higher score.
    # Score = 1 + log(|U-U|/|U+U|) / log(2*max(U)+1)
    # We want high ratio and low max.
    
    # Candidate: {0, 1, 3} -> n=3, max=3, log(7)=1.94
    # U-U: {0, 1, 2, 3, -1, -2, -3} -> 7
    # U+U: {0, 1, 2, 3, 4, 6} -> 6
    # Ratio 7/6 = 1.166. log(1.166)=0.153. Score = 1 + 0.153/1.94 = 1.079
    
    # Candidate: {0, 1, 4, 6} -> n=4, max=6, log(13)=2.56
    # U-U: {0, 1, 2, 3, 4, 5, 6, -1, -2, -3, -4, -5, -6} -> 13
    # U+U: {0, 1, 2, 4, 5, 6, 7, 8, 10, 12} -> 10
    # Ratio 13/10 = 1.3. log(1.3)=0.262. Score = 1 + 0.262/2.56 = 1.102
    
    # Candidate: {0, 1, 4, 10, 11} -> n=5, max=11, log(23)=3.13
    # U-U size?
    # U+U size?
    # Let's just run local search on these.
    
    candidates = [
        [0, 1, 3],
        [0, 1, 4, 6],
        [0, 1, 4, 10, 11],
        [0, 1, 4, 10, 12, 17],
        [0, 1, 4, 10, 12, 17, 22],
        [0, 1, 4, 10, 12, 17, 22, 29],
        [0, 1, 4, 10, 12, 17, 22, 29, 36],
        [0, 1, 4, 10, 12, 17, 22, 29, 36, 44],
        [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52],
        [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60],
    ]
    
    best_U = [0, 1, 4, 6]
    best_score = _score_set_fast(best_U)
    
    for U in candidates:
        if time.time() > deadline - 10:
            break
        s = _score_set_fast(U)
        if s > best_score:
            best_score = s
            best_U = U[:]
            
    # Local search on best candidate
    if time.time() < deadline - 5:
        polished_U, polished_score = _local_search(best_U, deadline=deadline - 1)
        if polished_score > best_score:
            best_score = polished_score
            best_U = polished_U
            
    # Try a few random perturbations of the best candidate
    random.seed(42)
    for _ in range(10):
        if time.time() > deadline - 2:
            break
        mutated = best_U[:]
        idx = random.randint(1, len(mutated)-1)
        delta = random.randint(-3, 3)
        mutated[idx] += delta
        if mutated[idx] <= 0:
            mutated[idx] = 1
        mutated = sorted(set(mutated))
        s = _score_set_fast(mutated)
        if s > best_score:
            best_score = s
            best_U = mutated
    
    return sorted(set(best_U))
