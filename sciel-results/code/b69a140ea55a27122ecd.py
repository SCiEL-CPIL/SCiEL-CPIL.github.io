import time
import random
import numpy as np

def _primes_upto(n):
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = [False] * len(sieve[i*i::i])
    return [i for i, is_p in enumerate(sieve) if is_p]

PRIMES = _primes_upto(1000)
N_PRIMES = len(PRIMES)
MAX_CHECK = 2000 # Safe upper bound for search

def _score(residues):
    """
    Computes the largest L such that every integer 1..L is covered by at least 2 congruences.
    """
    cov = np.zeros(MAX_CHECK + 1, dtype=np.int8)
    
    for i, p in enumerate(PRIMES):
        r = residues[i]
        start = r % p
        if start == 0:
            start = p
        
        if start <= MAX_CHECK:
            cov[start::p] += 1
            
    valid = cov[1:] >= 2
    if np.all(valid):
        return MAX_CHECK
    
    first_fail_idx = np.argmax(~valid)
    return int(first_fail_idx)

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140 seconds
    
    # Initial residues: Random
    residues = np.array([random.randint(0, p - 1) for p in PRIMES], dtype=np.int32)
    
    best_residues = residues.copy()
    best_score = _score(best_residues)
    
    current_residues = residues.copy()
    current_score = best_score
    
    stuck_count = 0
    max_stuck = 2000
    
    while time.time() < deadline:
        gap = current_score + 1
        
        # Generate candidates
        candidates = []
        
        # 1. Target the gap with small primes
        for i in range(min(50, N_PRIMES)):
            p = PRIMES[i]
            target_r = gap % p
            if current_residues[i] != target_r:
                candidates.append((i, target_r))
                
        # 2. Target the gap with random primes
        for _ in range(20):
            i = random.randint(0, N_PRIMES - 1)
            p = PRIMES[i]
            target_r = gap % p
            if current_residues[i] != target_r:
                candidates.append((i, target_r))
                
        # 3. Random perturbations if stuck
        if stuck_count > 50:
            for _ in range(10):
                i = random.randint(0, N_PRIMES - 1)
                p = PRIMES[i]
                new_r = random.randint(0, p - 1)
                if current_residues[i] != new_r:
                    candidates.append((i, new_r))
        
        best_move_score = current_score
        best_move_idx = -1
        best_move_r = -1
        
        for i, new_r in candidates:
            old_r = current_residues[i]
            current_residues[i] = new_r
            
            s = _score(current_residues)
            
            current_residues[i] = old_r
            
            if s > best_move_score:
                best_move_score = s
                best_move_idx = i
                best_move_r = new_r
        
        if best_move_idx != -1:
            current_residues[best_move_idx] = best_move_r
            current_score = best_move_score
            stuck_count = 0
            
            if current_score > best_score:
                best_score = current_score
                best_residues = current_residues.copy()
        else:
            stuck_count += 1
            if stuck_count > max_stuck:
                # Random restart
                for i in range(N_PRIMES):
                    current_residues[i] = random.randint(0, PRIMES[i] - 1)
                current_score = _score(current_residues)
                stuck_count = 0
                if current_score > best_score:
                    best_score = current_score
                    best_residues = current_residues.copy()

    return [int(r) for r in best_residues]
