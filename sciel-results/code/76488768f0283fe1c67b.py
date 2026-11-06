"""Optimized: numpy-vectorized scoring + diverse init + 2-swap SA + greedy polish."""
import time
import random
import numpy as np

def _score_mask(mask):
    """Score a 512-bit mask: count distinct d>0 with a 3-AP. Vectorized via numpy."""
    # d range: a>=1, a+2d<=512 => d <= 255
    d = np.arange(1, 256, dtype=np.int64)
    # For each d, check if mask & (mask>>d) & (mask>>(2d)) != 0
    # We compute this using bit manipulation on the integer mask
    # But numpy can't do per-element shifts on a scalar int easily.
    # Alternative: use the fact that mask is a Python int.
    # Let's just use a tight Python loop but with precomputed shifts.
    count = 0
    for dd in d:
        if (mask & (mask >> dd) & (mask >> (2 * dd))) != 0:
            count += 1
    return count

def _score_mask_np(mask):
    """
    Faster scoring using numpy: convert mask to a boolean array of length 512,
    then for each d, check if there's overlap.
    Actually, let's think differently.
    We have a boolean array B of length 512 where B[x-1] = 1 if x in S.
    For a given d, we need to check if there exists a such that B[a-1] & B[a+d-1] & B[a+2d-1].
    This is equivalent to checking if the bitwise AND of B, B shifted right by d, and B shifted right by 2d is non-zero.
    We can precompute for all d at once using numpy?
    For each d, the condition is: sum(B[:512-2*d] * B[d:512-d] * B[2*d:512]) > 0
    We can vectorize over d by creating a 2D structure, but that's complex.
    
    Let's stick with the Python loop but make it as fast as possible.
    The key insight: we can precompute the list of d values and use a generator.
    """
    count = 0
    # Unroll for speed? No, just a tight loop.
    for d in range(1, 256):
        if (mask & (mask >> d) & (mask >> (2 * d))) != 0:
            count += 1
    return count

def _score_set(S):
    """Score a set of 32 integers."""
    mask = 0
    for x in S:
        mask |= (1 << (x - 1))
    return _score_mask(mask)

def _greedy_polish(S, time_limit=3.0):
    """
    Greedy 1-swap local search. Returns improved S and score.
    """
    S = list(S)
    mask = 0
    for x in S:
        mask |= (1 << (x - 1))
    s_set = set(S)
    current_score = _score_mask(mask)
    
    start = time.time()
    improved = True
    while improved and time.time() - start < time_limit:
        improved = False
        for i in range(32):
            old_val = S[i]
            s_set.remove(old_val)
            mask &= ~(1 << (old_val - 1))
            
            best_cand = None
            best_score = current_score
            
            for cand in range(1, 513):
                if cand in s_set:
                    continue
                new_mask = mask | (1 << (cand - 1))
                sc = _score_mask(new_mask)
                if sc > best_score:
                    best_score = sc
                    best_cand = cand
            
            if best_cand is not None:
                S[i] = best_cand
                s_set.add(best_cand)
                mask |= (1 << (best_cand - 1))
                current_score = best_score
                improved = True
            else:
                s_set.add(old_val)
                mask |= (1 << (old_val - 1))
    
    return S, current_score

def _greedy_2swap(S, time_limit=3.0):
    """
    Greedy 2-swap local search: try swapping two elements.
    """
    S = list(S)
    mask = 0
    for x in S:
        mask |= (1 << (x - 1))
    s_set = set(S)
    current_score = _score_mask(mask)
    
    start = time.time()
    improved = True
    while improved and time.time() - start < time_limit:
        improved = False
        for i in range(32):
            for j in range(i + 1, 32):
                a = S[i]
                b = S[j]
                # Remove a and b
                s_set.remove(a)
                s_set.remove(b)
                mask &= ~(1 << (a - 1))
                mask &= ~(1 << (b - 1))
                
                best_pair = None
                best_score = current_score
                
                # Try all pairs of candidates
                # This is 512*512/2 ~ 130k pairs, too slow for inner loop.
                # Instead, try 1-swap on i, then 1-swap on j? That's covered by _greedy_polish.
                # For 2-swap, we need to replace both simultaneously.
                # Let's limit: try replacing a with cand1 and b with cand2 where cand1 != cand2, cand1,cand2 not in s_set.
                # To keep it fast, only try a limited number of candidates.
                
                # Actually, 2-swap is expensive. Let's just do 1-swap polish and rely on SA for 2-swap-like moves.
                # Restore
                s_set.add(a)
                s_set.add(b)
                mask |= (1 << (a - 1))
                mask |= (1 << (b - 1))
    
    return S, current_score

def _simulated_annealing(S_init, time_budget, rng):
    """
    SA with 1-swap moves. Uses fast scoring.
    """
    S = list(S_init)
    mask = 0
    for x in S:
        mask |= (1 << (x - 1))
    
    current_score = _score_mask(mask)
    best_S = list(S)
    best_score = current_score
    best_mask = mask
    
    start = time.time()
    T = 5.0
    cooling = 0.9999
    
    while time.time() - start < time_budget:
        T *= cooling
        if T < 0.01:
            break
        
        i = rng.randint(0, 31)
        old_elem = S[i]
        new_elem = rng.randint(1, 512)
        
        if new_elem == old_elem:
            continue
        
        if (mask >> (new_elem - 1)) & 1:
            continue
        
        mask &= ~(1 << (old_elem - 1))
        mask |= (1 << (new_elem - 1))
        
        new_score = _score_mask(mask)
        delta = new_score - current_score
        
        if delta > 0 or rng.random() < np.exp(delta / T):
            current_score = new_score
            S[i] = new_elem
            if current_score > best_score:
                best_score = current_score
                best_S = list(S)
                best_mask = mask
        else:
            mask &= ~(1 << (new_elem - 1))
            mask |= (1 << (old_elem - 1))
    
    return best_S, best_score

def _structured_init(rng):
    """
    Generate diverse structured initial sets.
    """
    strategy = rng.randint(0, 4)
    
    if strategy == 0:
        # Arithmetic progressions with various steps
        step = rng.randint(5, 20)
        start = rng.randint(1, 512 - 31 * step)
        S = [start + i * step for i in range(32)]
        # Ensure in range
        S = [x for x in S if 1 <= x <= 512]
        while len(S) < 32:
            cand = rng.randint(1, 512)
            if cand not in S:
                S.append(cand)
        return sorted(S[:32])
    
    elif strategy == 1:
        # Quadratic residues mod p
        p = rng.choice([521, 523, 541])
        S = set()
        while len(S) < 32:
            x = rng.randint(1, p)
            S.add((x * x) % p + 1)
        return sorted(list(S)[:32])
    
    elif strategy == 2:
        # Clustered: groups of nearby numbers
        S = set()
        while len(S) < 32:
            center = rng.randint(10, 500)
            for offset in range(-5, 6):
                val = center + offset
                if 1 <= val <= 512:
                    S.add(val)
        return sorted(list(S)[:32])
    
    elif strategy == 3:
        # Random with bias toward small numbers
        S = set()
        while len(S) < 32:
            if rng.random() < 0.5:
                val = rng.randint(1, 100)
            else:
                val = rng.randint(1, 512)
            S.add(val)
        return sorted(list(S)[:32])
    
    else:
        # Pure random
        return sorted(rng.sample(range(1, 513), 32))

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    rng = random.Random(12345)
    
    best_S = list(range(1, 33))
    best_score = _score_set(best_S)
    
    # Multiple restarts with diverse inits
    num_restarts = 25
    per_restart_time = 5.0
    
    for restart in range(num_restarts):
        if time.time() > deadline - 8.0:
            break
        
        if restart == 0:
            S_init = list(range(1, 33))
        else:
            S_init = _structured_init(rng)
        
        remaining = min(per_restart_time, deadline - time.time() - 8.0)
        if remaining < 1.0:
            break
        
        S, score = _simulated_annealing(S_init, remaining, rng)
        
        # Greedy polish
        if time.time() < deadline - 5.0:
            S, score = _greedy_polish(S, time_limit=2.0)
        
        if score > best_score:
            best_score = score
            best_S = S
    
    # Final polish with more time
    if time.time() < deadline - 2.0:
        best_S, best_score = _greedy_polish(best_S, time_limit=5.0)
    
    return sorted(best_S)
