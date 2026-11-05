"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import math

def score_set(S):
    """Count distinct d > 0 such that a, a+d, a+2d all in S for some a."""
    S_set = set(S)
    distinct_d = set()
    for a in S:
        for b in S:
            if b > a:
                d = b - a
                if a + 2 * d in S_set:
                    distinct_d.add(d)
    return len(distinct_d)

def score_set_fast(S):
    """Faster scoring using set membership."""
    S_set = frozenset(S)
    distinct_d = set()
    for a in S:
        for b in S:
            if b > a:
                d = b - a
                if a + 2 * d in S_set:
                    distinct_d.add(d)
    return len(distinct_d)

def greedy_polish(S, max_iter=200):
    """Try single-element moves to improve score."""
    S = list(S)
    current_score = score_set(S)
    n = 512
    improved = True
    iters = 0
    while improved and iters < max_iter:
        improved = False
        iters += 1
        for i in range(len(S)):
            old_val = S[i]
            for new_val in range(1, n + 1):
                if new_val == old_val or new_val in S:
                    continue
                S[i] = new_val
                new_score = score_set(S)
                if new_score > current_score:
                    current_score = new_score
                    improved = True
                    break
                else:
                    S[i] = old_val
    return S, current_score

def simulated_annealing(S, time_limit, rng):
    """SA with 1-swap and 2-swap moves."""
    S = list(S)
    current_score = score_set(S)
    best_S = list(S)
    best_score = current_score
    
    start_time = time.time()
    T = 5.0
    T_min = 0.01
    cooling = 0.999
    
    while T > T_min:
        if time.time() - start_time > time_limit:
            break
        
        # Choose move type: 70% 1-swap, 30% 2-swap
        move_type = rng.random()
        
        if move_type < 0.7:
            # 1-swap: move one element to a new position
            i = rng.randint(0, len(S) - 1)
            new_val = rng.randint(1, 512)
            if new_val in S:
                T *= cooling
                continue
            old_val = S[i]
            S[i] = new_val
            new_score = score_set(S)
            delta = new_score - current_score
            if delta >= 0 or rng.random() < math.exp(delta / T):
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_S = list(S)
            else:
                S[i] = old_val
        else:
            # 2-swap: swap two elements with random new values
            i, j = rng.sample(range(len(S)), 2)
            new_i = rng.randint(1, 512)
            new_j = rng.randint(1, 512)
            # Ensure no duplicates
            if new_i in S or new_j in S or new_i == new_j:
                T *= cooling
                continue
            old_i, old_j = S[i], S[j]
            S[i], S[j] = new_i, new_j
            new_score = score_set(S)
            delta = new_score - current_score
            if delta >= 0 or rng.random() < math.exp(delta / T):
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_S = list(S)
            else:
                S[i], S[j] = old_i, old_j
        
        T *= cooling
    
    return best_S, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill
    
    rng = random.Random(42)
    
    best_S = list(range(1, 33))
    best_score = score_set(best_S)  # Should be 31
    
    # Try multiple random initializations
    num_starts = 8
    
    for s in range(num_starts):
        if time.time() > deadline - 5:
            break
        
        # Random initialization
        S = rng.sample(range(1, 513), 32)
        
        # Greedy polish
        S, sc = greedy_polish(S, max_iter=100)
        if sc > best_score:
            best_score = sc
            best_S = list(S)
        
        # SA
        remaining = deadline - time.time() - 2
        if remaining > 5:
            S, sc = simulated_annealing(S, remaining, rng)
            if sc > best_score:
                best_score = sc
                best_S = list(S)
        
        # Final polish on best
        if time.time() < deadline - 1:
            S, sc = greedy_polish(best_S, max_iter=50)
            if sc > best_score:
                best_score = sc
                best_S = list(S)
    
    # Final greedy polish
    if time.time() < deadline:
        best_S, sc = greedy_polish(best_S, max_iter=200)
        if sc > best_score:
            best_score = sc
            best_S = list(S)
    
    return best_S
# EVOLVE-BLOCK-END
