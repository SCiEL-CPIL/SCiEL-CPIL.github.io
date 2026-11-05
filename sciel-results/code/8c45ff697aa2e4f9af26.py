"""Optimized program: multi-restart SA + greedy polish for 3-AP differences."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def _score_set(S):
    """Count distinct d > 0 such that a, a+d, a+2d all in S."""
    s = set(S)
    diffs = set()
    for a in s:
        for d in range(1, 256):
            if a + d in s and a + 2 * d in s:
                diffs.add(d)
    return len(diffs)


def _score_set_fast(S_list):
    """Faster scoring using sorted list and set lookup."""
    s = set(S_list)
    diffs = set()
    for a in S_list:
        # max d such that a+2d <= 512
        max_d = (512 - a) // 2
        for d in range(1, max_d + 1):
            if a + d in s and a + 2 * d in s:
                diffs.add(d)
    return len(diffs)


def _greedy_polish(S, max_iters=200):
    """Try to improve by swapping elements greedily."""
    S = list(S)
    current_score = _score_set_fast(S)
    s = set(S)
    
    for _ in range(max_iters):
        improved = False
        # Try removing each element and adding a new one
        for i, elem in enumerate(S):
            s.remove(elem)
            # Try adding each candidate not in s
            for cand in range(1, 513):
                if cand in s:
                    continue
                s.add(cand)
                new_score = _score_set_fast(list(s))
                if new_score > current_score:
                    S[i] = cand
                    current_score = new_score
                    improved = True
                    break
                s.remove(cand)
            if improved:
                break
        if not improved:
            break
    return S, current_score


def _simulated_annealing(S_init, time_budget, rng):
    """SA with swap moves."""
    S = list(S_init)
    current_score = _score_set_fast(S)
    best_S = list(S)
    best_score = current_score
    
    start_time = time.time()
    T = 10.0
    cooling = 0.9995
    
    while time.time() - start_time < time_budget:
        T *= cooling
        if T < 0.01:
            break
        
        # Pick a random element to remove
        i = rng.randint(0, 31)
        old_elem = S[i]
        
        # Pick a random element to add
        new_elem = rng.randint(1, 512)
        if new_elem == old_elem:
            continue
        
        # Check if new_elem already in S (excluding position i)
        if any(S[j] == new_elem for j in range(32) if j != i):
            continue
        
        S[i] = new_elem
        new_score = _score_set_fast(S)
        delta = new_score - current_score
        
        if delta > 0 or rng.random() < np.exp(delta / T):
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_S = list(S)
        else:
            S[i] = old_elem
    
    return best_S, best_score


def _random_init(rng):
    return sorted(rng.sample(range(1, 513), 32))


def _structured_init(rng):
    """Generate structured initial sets with various patterns."""
    pattern = rng.randint(0, 4)
    if pattern == 0:
        # Arithmetic progression with random step
        start = rng.randint(1, 20)
        step = rng.randint(10, 15)
        S = [start + i * step for i in range(32)]
        # Clamp to range
        S = [max(1, min(512, x)) for x in S]
        # Remove duplicates and fill
        s = sorted(set(S))
        while len(s) < 32:
            cand = rng.randint(1, 512)
            if cand not in s:
                s.append(cand)
        return sorted(s[:32])
    elif pattern == 1:
        # Two interleaved APs
        start1 = rng.randint(1, 10)
        step1 = rng.randint(15, 25)
        start2 = rng.randint(1, 10)
        step2 = rng.randint(15, 25)
        S = []
        for i in range(16):
            S.append(start1 + i * step1)
            S.append(start2 + i * step2)
        S = [max(1, min(512, x)) for x in S]
        s = sorted(set(S))
        while len(s) < 32:
            cand = rng.randint(1, 512)
            if cand not in s:
                s.append(cand)
        return sorted(s[:32])
    elif pattern == 2:
        # Random with bias toward small values (more APs fit)
        S = sorted(rng.sample(range(1, 200), 32))
        return S
    elif pattern == 3:
        # Random with bias toward spread
        S = sorted(rng.sample(range(1, 513), 32))
        return S
    else:
        # Mixed: some structured, some random
        S = []
        base = rng.randint(1, 10)
        for i in range(16):
            S.append(base + i * rng.randint(12, 20))
        for i in range(16):
            S.append(rng.randint(1, 512))
        S = [max(1, min(512, x)) for x in S]
        s = sorted(set(S))
        while len(s) < 32:
            cand = rng.randint(1, 512)
            if cand not in s:
                s.append(cand)
        return sorted(s[:32])


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    rng = random.Random(42)
    
    best_S = list(range(1, 33))
    best_score = _score_set_fast(best_S)
    
    # Try multiple restarts with different initializations
    num_restarts = 10
    per_restart_time = 12.0  # seconds per restart
    
    for restart in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
        
        # Generate initial set
        if restart == 0:
            S_init = list(range(1, 33))
        elif restart % 2 == 0:
            S_init = _random_init(rng)
        else:
            S_init = _structured_init(rng)
        
        # Run SA
        remaining = min(per_restart_time, deadline - time.time() - 5.0)
        if remaining < 1.0:
            break
        
        S, score = _simulated_annealing(S_init, remaining, rng)
        
        # Greedy polish
        if time.time() < deadline - 3.0:
            S, score = _greedy_polish(S, max_iters=100)
        
        if score > best_score:
            best_score = score
            best_S = S
    
    # Final greedy polish on best
    if time.time() < deadline - 1.0:
        best_S, best_score = _greedy_polish(best_S, max_iters=150)
    
    return sorted(best_S)
# EVOLVE-BLOCK-END
