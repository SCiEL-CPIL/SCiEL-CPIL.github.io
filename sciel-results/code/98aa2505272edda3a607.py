"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import isqrt

P = {'n': 256}

def _is_valid(S):
    """Check if set S satisfies the condition: no a divides b*c for distinct a,b,c in S."""
    s_list = list(S)
    n = len(s_list)
    if n < 3:
        return True
    # Optimization: sort to help with early termination? 
    # We need to check all triples (a,b,c) distinct.
    # a | b*c
    for i in range(n):
        a = s_list[i]
        for j in range(n):
            if i == j: continue
            b = s_list[j]
            for k in range(n):
                if k == i or k == j: continue
                c = s_list[k]
                if (b * c) % a == 0:
                    return False
    return True

def _try_add(S, x):
    """Check if adding x to S keeps it valid. S is a set, x not in S."""
    if x in S:
        return S
    # Check all pairs (b,c) in S such that x | b*c
    s_list = list(S)
    for i in range(len(s_list)):
        b = s_list[i]
        for j in range(len(s_list)):
            if i == j: continue
            c = s_list[j]
            if (b * c) % x == 0:
                return None # Invalid
    return S | {x}

def _get_primes(n):
    """Return list of primes up to n."""
    is_prime = [True] * (n + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, isqrt(n) + 1):
        if is_prime[i]:
            for j in range(i*i, n + 1, i):
                is_prime[j] = False
    return [i for i in range(2, n + 1) if is_prime[i]]

def construct():
    n = P['n']
    start_time = time.time()
    deadline = start_time + 85.0 # Safety margin
    
    # Initial solution: Primes
    primes = _get_primes(n)
    S_best = set(primes)
    best_size = len(S_best)
    
    # Candidates: All integers in 2..n not in S_best
    candidates = [x for x in range(2, n + 1) if x not in S_best]
    
    # Local Search with Swaps
    # We try to improve the set size.
    # Strategy: 
    # 1. Try to add any candidate.
    # 2. If no candidate can be added, try swapping an element in S with a candidate.
    #    If the swap is valid, do it, and then try to add again.
    
    # To speed up, we can precompute conflicts? 
    # But N is small.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Phase 1: Greedy Add
        added_any = False
        for x in candidates[:]: # Iterate over copy
            if x in S_best: continue
            new_S = _try_add(S_best, x)
            if new_S is not None:
                S_best = new_S
                candidates.remove(x)
                best_size += 1
                improved = True
                added_any = True
        if added_any:
            continue # Restart loop to try more adds
        
        # Phase 2: Swap
        # If we couldn't add anything, try swapping.
        # We want to swap x in S_best with y in candidates such that S' is valid.
        # We don't strictly need size increase from swap itself (it's 0), 
        # but it might unlock additions.
        
        # To be efficient, only try swaps that might be useful?
        # Or just random swaps.
        
        # Let's try a limited number of random swaps to escape local optima.
        if len(S_best) > 0 and len(candidates) > 0:
            # Shuffle to randomize order
            s_list = list(S_best)
            random.shuffle(s_list)
            c_list = candidates[:]
            random.shuffle(c_list)
            
            # Limit swaps per iteration to avoid timeout
            max_swaps_to_try = 100
            swap_count = 0
            found_swap = False
            
            for x in s_list:
                if swap_count >= max_swaps_to_try: break
                for y in c_list:
                    if swap_count >= max_swaps_to_try: break
                    # Try swap x -> y
                    S_trial = S_best - {x}
                    # Check if y can be added to S_trial
                    new_S = _try_add(S_trial, y)
                    if new_S is not None:
                        # Valid swap
                        S_best = new_S
                        # Update candidates list
                        if x in candidates:
                            candidates.remove(x)
                        if y not in candidates:
                            candidates.append(y)
                        improved = True
                        found_swap = True
                        swap_count += 1
                        # After a swap, the set changed, so we should break inner loops 
                        # and restart the outer "improved" loop to try adds again.
                        break
                if found_swap:
                    break
            
            if found_swap:
                continue

    # Final check and return
    # Ensure types are correct
    result = [int(x) for x in S_best]
    return result

# EVOLVE-BLOCK-END
