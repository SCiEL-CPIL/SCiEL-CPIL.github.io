import time
import numpy as np
import random

def _primes_up_to(n):
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            for j in range(i * i, n + 1, i):
                sieve[j] = False
    return [i for i in range(2, n + 1) if sieve[i]]

def _compute_L_and_vulnerable(counts, max_L):
    below = np.where(counts[1:max_L + 1] < 2)[0]
    if len(below) == 0:
        L = max_L
    else:
        L = int(below[0])
    
    vulnerable = np.sum(counts[1:max_L + 1] == 1)
    
    return L, vulnerable

def _build_counts(residues, primes, L_max):
    counts = np.zeros(L_max + 1, dtype=np.int32)
    for i, p in enumerate(primes):
        r = int(residues[i])
        if r == 0:
            if p <= L_max:
                counts[p::p] += 1
        else:
            if r <= L_max:
                counts[r::p] += 1
    return counts

def _update_counts(counts, p, old_r, new_r, L_max):
    if old_r == 0:
        if p <= L_max:
            counts[p::p] -= 1
    else:
        if old_r <= L_max:
            counts[old_r::p] -= 1
            
    if new_r == 0:
        if p <= L_max:
            counts[p::p] += 1
    else:
        if new_r <= L_max:
            counts[new_r::p] += 1

def construct():
    start_time = time.time()
    deadline = start_time + 118.0
    
    primes = _primes_up_to(1000)
    n_primes = len(primes)
    L_search = 50000

    best_residues = None
    best_L = 0
    best_vulnerable = float('inf')
    best_counts = None
    
    inits = [
        lambda i, p: ((i + 2) // 2) % p,
        lambda i, p: 0,
        lambda i, p: 1,
        lambda i, p: (i * 7 + 3) % p,
        lambda i, p: (i * 13 + 5) % p,
    ]
    
    for init_func in inits:
        residues = np.array([init_func(i, p) for i, p in enumerate(primes)], dtype=np.int64)
        counts = _build_counts(residues, primes, L_search)
        L, vuln = _compute_L_and_vulnerable(counts, L_search)
        if L > best_L or (L == best_L and vuln < best_vulnerable):
            best_L = L
            best_vulnerable = vuln
            best_residues = residues.copy()
            best_counts = counts.copy()

    residues = best_residues.copy()
    counts = best_counts.copy()
    current_L = best_L
    current_vulnerable = best_vulnerable

    T_start = 50.0
    T_end = 0.5
    random.seed(42)
    
    while time.time() < deadline - 2.0:
        elapsed = time.time() - start_time
        total_time_budget = deadline - 2.0 - start_time
        if total_time_budget <= 0:
            break
        progress = min(1.0, elapsed / total_time_budget)
        T = T_start * (T_end / T_start) ** progress
        
        target_x = current_L + 1
        if target_x > L_search:
            break

        move_accepted = False
        attempts = 0
        max_attempts = 20
        
        while attempts < max_attempts and not move_accepted:
            attempts += 1
            i = random.randint(0, n_primes - 1)
            p = primes[i]
            old_r = int(residues[i])
            
            if random.random() < 0.7:
                new_r = target_x % p
            else:
                new_r = random.randint(0, p - 1)
                
            if old_r == new_r:
                continue

            _update_counts(counts, p, old_r, new_r, L_search)
            residues[i] = new_r
            
            new_L, new_vuln = _compute_L_and_vulnerable(counts, L_search)
            
            accept = False
            if new_L > current_L:
                accept = True
            elif new_L == current_L:
                if new_vuln < current_vulnerable:
                    accept = True
                elif new_vuln == current_vulnerable:
                    if random.random() < 0.1:
                        accept = True
            else:
                delta_L = current_L - new_L
                if random.random() < np.exp(-delta_L / T):
                    accept = True

            if accept:
                current_L = new_L
                current_vulnerable = new_vuln
                if current_L > best_L or (current_L == best_L and current_vulnerable < best_vulnerable):
                    best_L = current_L
                    best_vulnerable = current_vulnerable
                    best_residues = residues.copy()
                    best_counts = counts.copy()
                move_accepted = True
            else:
                _update_counts(counts, p, new_r, old_r, L_search)
                residues[i] = old_r
                move_accepted = True

        if not move_accepted:
            i = random.randint(0, n_primes - 1)
            p = primes[i]
            old_r = int(residues[i])
            new_r = random.randint(0, p - 1)
            if new_r != old_r:
                _update_counts(counts, p, old_r, new_r, L_search)
                residues[i] = new_r
                new_L, new_vuln = _compute_L_and_vulnerable(counts, L_search)
                current_L = new_L
                current_vulnerable = new_vuln
                if current_L > best_L or (current_L == best_L and current_vulnerable < best_vulnerable):
                    best_L = current_L
                    best_vulnerable = current_vulnerable
                    best_residues = residues.copy()
                    best_counts = counts.copy()

    # Final verification
    L_verify = 100000
    if time.time() < deadline - 1.0:
        counts_v = _build_counts(best_residues, primes, L_verify)
        final_L, _ = _compute_L_and_vulnerable(counts_v, L_verify)
        
    return [int(r) for r in best_residues]
