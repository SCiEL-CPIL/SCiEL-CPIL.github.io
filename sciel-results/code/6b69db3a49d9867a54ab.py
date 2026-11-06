"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def _score_set(S):
    """Count distinct d > 0 such that a, a+d, a+2d all in S for some a."""
    S_set = set(S)
    mn = min(S)
    mx = max(S)
    distinct_d = set()
    for d in range(1, (mx - mn) // 2 + 1):
        for a in S:
            if a + d in S_set and a + 2 * d in S_set:
                distinct_d.add(d)
                break
    return len(distinct_d)

def _local_search(S, time_limit):
    """Local search: try swaps to improve score."""
    start = time.time()
    best_S = list(S)
    best_score = _score_set(best_S)
    n = len(best_S)
    best_S_set = set(best_S)
    
    while time.time() - start < time_limit:
        improved = False
        for i in range(n):
            if time.time() - start > time_limit:
                break
            old_val = best_S[i]
            for _ in range(80):
                new_val = random.randint(1, 512)
                if new_val == old_val or new_val in best_S_set:
                    continue
                best_S[i] = new_val
                best_S_set.discard(old_val)
                best_S_set.add(new_val)
                new_score = _score_set(best_S)
                if new_score > best_score:
                    best_score = new_score
                    improved = True
                else:
                    best_S[i] = old_val
                    best_S_set.discard(new_val)
                    best_S_set.add(old_val)
        
        if not improved:
            for i in range(n):
                if random.random() < 0.3:
                    new_val = random.randint(1, 512)
                    if new_val not in best_S_set:
                        best_S_set.discard(best_S[i])
                        best_S[i] = new_val
                        best_S_set.add(new_val)
            best_score = _score_set(best_S)
    
    return best_S, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    best_S = list(range(1, 33))
    best_score = _score_set(best_S)
    
    random.seed(42)
    for restart in range(300):
        if time.time() > deadline:
            break
        
        S = random.sample(range(1, 513), 32)
        score = _score_set(S)
        if score > best_score:
            best_S = list(S)
            best_score = score
        
        remaining = deadline - time.time()
        if remaining < 2:
            break
        time_budget = min(3.0, remaining * 0.8)
        
        S2, score2 = _local_search(S, time_budget)
        if score2 > best_score:
            best_S = list(S2)
            best_score = score2
        
        remaining = deadline - time.time()
        if remaining < 2:
            break
        time_budget = min(3.0, remaining * 0.8)
        S3, score3 = _local_search(best_S, time_budget)
        if score3 > best_score:
            best_S = list(S3)
            best_score = score3
    
    remaining = deadline - time.time()
    if remaining > 5:
        S4, score4 = _local_search(best_S, remaining * 0.9)
        if score4 > best_score:
            best_S = list(S4)
            best_score = score4
    
    return best_S
# EVOLVE-BLOCK-END
