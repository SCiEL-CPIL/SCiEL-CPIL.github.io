import time
import random

def score_set(S):
    S_set = set(S)
    diffs = set()
    for a in S:
        # Max d is (512 - a) // 2
        max_d = (512 - a) // 2
        for d in range(1, max_d + 1):
            if a + d in S_set and a + 2*d in S_set:
                diffs.add(d)
    return len(diffs)

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    # Phase 1: Structured search
    base = list(range(1, 31))
    best_S = base + [31, 32]
    best_score = score_set(best_S)
    
    # Search x, y in 31..100
    for x in range(31, 101):
        for y in range(x + 1, 101):
            S = base + [x, y]
            s = score_set(S)
            if s > best_score:
                best_score = s
                best_S = S.copy()
                
        if time.time() > deadline - 10:
            break
            
    if time.time() > deadline - 5:
        return best_S

    # Phase 2: Local search
    S = best_S.copy()
    current_score = best_score
    
    # Candidate pool
    # Focus on numbers that might improve
    # Random swaps
    while time.time() < deadline - 2:
        improved = False
        for _ in range(200):
            i = random.randint(0, 31)
            j = random.randint(1, 512)
            
            if j in S:
                continue
                
            old_val = S[i]
            S[i] = j
            
            s = score_set(S)
            if s > current_score:
                current_score = s
                best_S = S.copy()
                best_score = s
                improved = True
            else:
                S[i] = old_val
                
        if not improved:
            # Perturb
            # Swap two random elements
            i1 = random.randint(0, 31)
            i2 = random.randint(0, 31)
            S[i1], S[i2] = S[i2], S[i1]
            current_score = score_set(S)
            if current_score > best_score:
                best_score = current_score
                best_S = S.copy()

    return best_S
