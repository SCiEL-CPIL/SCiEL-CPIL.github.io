import time
import random
import numpy as np

def _get_primes():
    limit = 1000
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, limit + 1, i):
                sieve[j] = False
    return [i for i, is_prime in enumerate(sieve) if is_prime]

_primes = _get_primes()
_n_primes = len(_primes)

def _score(residues, max_L):
    indices = np.arange(1, max_L + 1)
    coverage = np.zeros(max_L, dtype=np.int32)
    for i in range(_n_primes):
        p = _primes[i]
        r = residues[i]
        coverage += (indices % p == r)
    bad = coverage < 2
    if not np.any(bad):
        return max_L
    return int(np.argmax(bad))

def construct():
    deadline = time.time() + 145.0
    random.seed(42)
    
    # Initial residues: Seed-like or random
    # Let's start with the seed from the prompt to be safe, or random.
    # Seed: ((i + 2) // 2) % p
    residues = [((i + 2) // 2) % p for i, p in enumerate(_primes)]
    
    current_max_check = 1000
    best_L = _score(residues, current_max_check)
    best_residues = residues[:]
    
    # If we hit the limit, increase it
    if best_L == current_max_check:
        current_max_check = 5000
        best_L = _score(residues, current_max_check)
        if best_L == current_max_check:
            current_max_check = 20000
            best_L = _score(residues, current_max_check)
            if best_L == current_max_check:
                current_max_check = 100000
                best_L = _score(residues, current_max_check)
                
    improved = True
    no_improve_count = 0
    
    while time.time() < deadline - 2.0:
        # Re-evaluate current score with current max_check
        current_L = _score(residues, current_max_check)
        
        if current_L > best_L:
            best_L = current_L
            best_residues = residues[:]
            no_improve_count = 0
            # If we hit the ceiling, try to raise it
            if best_L == current_max_check and current_max_check < 100000:
                if current_max_check < 5000:
                    current_max_check = 5000
                elif current_max_check < 20000:
                    current_max_check = 20000
                else:
                    current_max_check = 100000
                best_L = _score(residues, current_max_check)
                if best_L > best_L: # This logic is slightly off, best_L is already updated
                    pass 
                # Just update best_L to the new check
                best_L = current_max_check # Assuming it filled it
                # Actually, let's just re-score
                best_L = _score(residues, current_max_check)
                if best_L == current_max_check:
                    # We are at the cap, keep searching but score is capped
                    pass
        
        if best_L >= 100000:
            break
            
        gap_x = best_L + 1
        if gap_x > current_max_check:
             # This shouldn't happen if best_L < current_max_check
             pass

        # Find primes that do NOT cover gap_x
        # Coverage of gap_x:
        # count = sum(1 for i in range(_n_primes) if gap_x % _primes[i] == residues[i])
        # We only want to change primes where gap_x % p != residues[i]
        
        candidates = []
        
        # 1. Targeted moves: Fix the gap
        for i in range(_n_primes):
            p = _primes[i]
            target_r = gap_x % p
            if target_r == residues[i]:
                continue # Already covers, changing it would lose coverage
            
            # Try this move
            old_r = residues[i]
            residues[i] = target_r
            new_L = _score(residues, current_max_check)
            residues[i] = old_r
            
            if new_L > best_L:
                candidates.append((new_L, i, target_r))
                
        # 2. Random moves if no targeted improvement or to diversify
        if not candidates:
            # Try a few random moves
            for _ in range(10):
                if time.time() > deadline - 2.0:
                    break
                i = random.randint(0, _n_primes - 1)
                new_r = random.randint(0, _primes[i] - 1)
                if new_r == residues[i]:
                    continue
                old_r = residues[i]
                residues[i] = new_r
                new_L = _score(residues, current_max_check)
                residues[i] = old_r
                if new_L > best_L:
                    candidates.append((new_L, i, new_r))
                    break # Accept first random improvement
        
        if candidates:
            candidates.sort(reverse=True)
            best_new_L, best_i, best_r = candidates[0]
            residues[best_i] = best_r
            if best_new_L > best_L:
                best_L = best_new_L
                best_residues = residues[:]
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            no_improve_count += 1
            
        # If stuck for too long, try a larger random perturbation
        if no_improve_count > 50:
            # Change 5 random residues
            for _ in range(5):
                i = random.randint(0, _n_primes - 1)
                residues[i] = random.randint(0, _primes[i] - 1)
            new_L = _score(residues, current_max_check)
            if new_L > best_L:
                best_L = new_L
                best_residues = residues[:]
            else:
                # Revert to best
                residues = best_residues[:]
            no_improve_count = 0

    return best_residues
