import time
import numpy as np
import random

def _primes_up_to(n):
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = [False] * len(sieve[i*i::i])
    return [i for i in range(n + 1) if sieve[i]]

_PRIMES = _primes_up_to(1000)
_NPRIMES = len(_PRIMES)

def _build_count_np(residues, L_MAX):
    count = np.zeros(L_MAX + 1, dtype=np.int32)
    for i, p in enumerate(_PRIMES):
        r = residues[i]
        start = p if r == 0 else r
        if start <= L_MAX:
            count[start::p] += 1
    return count

def _find_prefix_L_np(count, L_MAX):
    if L_MAX < 1:
        return 0
    below = np.where(count[1:L_MAX+1] < 2)[0]
    if len(below) == 0:
        return L_MAX
    return int(below[0])

def _apply_move(count, residues, i, new_r, L_LIMIT):
    p = _PRIMES[i]
    old_r = residues[i]
    if old_r == new_r:
        return
    start_old = p if old_r == 0 else old_r
    start_new = p if new_r == 0 else new_r
    if start_old <= L_LIMIT:
        count[start_old::p] -= 1
    if start_new <= L_LIMIT:
        count[start_new::p] += 1
    residues[i] = new_r

def _revert_move(count, residues, i, old_r, L_LIMIT):
    p = _PRIMES[i]
    new_r = residues[i]
    start_old = p if old_r == 0 else old_r
    start_new = p if new_r == 0 else new_r
    if start_new <= L_LIMIT:
        count[start_new::p] -= 1
    if start_old <= L_LIMIT:
        count[start_old::p] += 1
    residues[i] = old_r

def _greedy_init_small_primes(L_SMALL=500):
    residues = [0] * _NPRIMES
    count = np.zeros(L_SMALL + 1, dtype=np.int32)
    
    for i, p in enumerate(_PRIMES):
        if p > L_SMALL:
            break
            
        best_r = 0
        best_score = -1
        
        for r in range(p):
            start = p if r == 0 else r
            if start > L_SMALL:
                continue
            positions = np.arange(start, L_SMALL + 1, p)
            if len(positions) == 0:
                continue
            current_counts = count[positions]
            would_reach_2 = np.sum(current_counts == 1)
            stays_2 = np.sum(current_counts >= 2)
            score = would_reach_2 * 10 + stays_2
            if score > best_score:
                best_score = score
                best_r = r
        
        residues[i] = best_r
        start = p if best_r == 0 else best_r
        if start <= L_SMALL:
            count[start::p] += 1
    
    for i, p in enumerate(_PRIMES):
        if p > L_SMALL:
            residues[i] = 0
            
    return residues

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    random.seed(42)
    
    # Phase 1: Greedy initialization
    L_SMALL = 500
    residues = _greedy_init_small_primes(L_SMALL)
    
    # Phase 2: Build count and find initial L
    L_LIMIT = 5000
    count = _build_count_np(residues, L_LIMIT)
    current_L = _find_prefix_L_np(count, L_LIMIT)
    
    best_L = current_L
    best_residues = list(residues)
    
    # Phase 3: Greedy repair with targeted local search
    stagnation_count = 0
    max_stagnation = 30
    
    while time.time() < deadline - 15:
        # Extend window if needed
        if current_L > L_LIMIT - 500:
            new_limit = min(L_LIMIT * 2, 100000)
            count = _build_count_np(residues, new_limit)
            L_LIMIT = new_limit
            current_L = _find_prefix_L_np(count, L_LIMIT)
        
        gap = current_L + 1
        
        if gap > L_LIMIT:
            new_limit = min(L_LIMIT * 2, 100000)
            count = _build_count_np(residues, new_limit)
            L_LIMIT = new_limit
            current_L = _find_prefix_L_np(count, L_LIMIT)
            continue
        
        # Check if gap is already covered
        if count[gap] >= 2:
            while gap <= L_LIMIT and count[gap] >= 2:
                current_L = gap
                gap += 1
            if current_L > best_L:
                best_L = current_L
                best_residues = list(residues)
            stagnation_count = 0
            continue
        
        # Gap not covered. Find best move to fix it.
        best_move = None
        min_cost = float('inf')
        
        prime_order = list(range(_NPRIMES))
        prime_order.sort(key=lambda i: _PRIMES[i])
        
        for i in prime_order:
            p = _PRIMES[i]
            old_r = residues[i]
            new_r = gap % p
            
            if old_r == new_r:
                continue
            
            start_x = p if old_r == 0 else old_r
            if start_x > current_L:
                cost = 0
            else:
                positions = np.arange(start_x, current_L + 1, p)
                cost = np.sum(count[positions] == 2)
            
            if cost < min_cost:
                min_cost = cost
                best_move = (i, new_r)
                if cost == 0:
                    break
        
        if best_move is not None:
            i, new_r = best_move
            old_r = residues[i]
            p = _PRIMES[i]
            
            _apply_move(count, residues, i, new_r, L_LIMIT)
            
            new_L = _find_prefix_L_np(count, L_LIMIT)
            
            if new_L > current_L:
                current_L = new_L
                stagnation_count = 0
                if current_L > best_L:
                    best_L = current_L
                    best_residues = list(residues)
            else:
                _revert_move(count, residues, i, old_r, L_LIMIT)
                stagnation_count += 1
        else:
            stagnation_count += 1
        
        # If stuck, do targeted SA on bottleneck primes
        if stagnation_count > max_stagnation:
            gap = current_L + 1
            bottleneck_primes = []
            for i, p in enumerate(_PRIMES):
                if (gap % p) != residues[i]:
                    bottleneck_primes.append(i)
            
            if bottleneck_primes:
                # Randomly pick a prime and try a random residue
                i = random.choice(bottleneck_primes)
                p = _PRIMES[i]
                old_r = residues[i]
                new_r = random.randint(0, p - 1)
                
                if new_r != old_r:
                    _apply_move(count, residues, i, new_r, L_LIMIT)
                    new_L = _find_prefix_L_np(count, L_LIMIT)
                    
                    if new_L > current_L:
                        current_L = new_L
                        stagnation_count = 0
                        if current_L > best_L:
                            best_L = current_L
                            best_residues = list(residues)
                    else:
                        _revert_move(count, residues, i, old_r, L_LIMIT)
                        stagnation_count += 1
            else:
                # No bottleneck primes? Try random moves on any prime
                i = random.randint(0, _NPRIMES - 1)
                p = _PRIMES[i]
                old_r = residues[i]
                new_r = random.randint(0, p - 1)
                
                if new_r != old_r:
                    _apply_move(count, residues, i, new_r, L_LIMIT)
                    new_L = _find_prefix_L_np(count, L_LIMIT)
                    
                    if new_L > current_L:
                        current_L = new_L
                        stagnation_count = 0
                        if current_L > best_L:
                            best_L = current_L
                            best_residues = list(residues)
                    else:
                        _revert_move(count, residues, i, old_r, L_LIMIT)
                        stagnation_count += 1

    # Final check: ensure we return the best found
    if best_L > current_L:
        residues = best_residues
    else:
        # Re-verify current residues score
        final_count = _build_count_np(residues, max(L_LIMIT, 10000))
        final_L = _find_prefix_L_np(final_count, max(L_LIMIT, 10000))
        if final_L > best_L:
            best_L = final_L
            best_residues = list(residues)
    
    # Ensure return type is list of Python ints
    return [int(r) for r in best_residues]
