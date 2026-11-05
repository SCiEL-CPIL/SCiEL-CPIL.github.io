import numpy as np
import time
import random

def get_primes(n):
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            for j in range(i * i, n + 1, i):
                sieve[j] = False
    return [i for i, is_prime in enumerate(sieve) if is_prime]

PRIMES = get_primes(1000)
N_PRIMES = len(PRIMES)
M = 20000

def evaluate(residues):
    counts = np.zeros(M + 1, dtype=np.int32)
    for i, p in enumerate(PRIMES):
        r = residues[i]
        counts[r::p] += 1
        
    invalid = counts < 2
    first_invalid_idx = np.argmax(invalid[1:])
    if invalid[1:][first_invalid_idx]:
        return first_invalid_idx
    else:
        return M

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    residues = [((i + 2) // 2) % p for i, p in enumerate(PRIMES)]
    current_score = evaluate(residues)
    best_residues = residues.copy()
    best_score = current_score
    
    T = 50.0
    cooling_rate = 0.995
    min_T = 0.1
    
    while time.time() < deadline and T > min_T:
        # Strategy 1: Targeted move to fix the first failure
        # Find first failure
        counts = np.zeros(M + 1, dtype=np.int32)
        for i, p in enumerate(PRIMES):
            counts[residues[i]::p] += 1
        
        invalid = counts < 2
        first_invalid_idx = np.argmax(invalid[1:])
        if invalid[1:][first_invalid_idx]:
            x_fail = first_invalid_idx + 1
            # Find primes that cover x_fail
            covering_primes = []
            non_covering_primes = []
            for i, p in enumerate(PRIMES):
                if x_fail % p == residues[i]:
                    covering_primes.append(i)
                else:
                    non_covering_primes.append(i)
            
            if len(covering_primes) < 2 and non_covering_primes:
                # Pick a random non-covering prime and force it to cover x_fail
                i = random.choice(non_covering_primes)
                p = PRIMES[i]
                old_r = residues[i]
                new_r = x_fail % p
                
                residues[i] = new_r
                new_score = evaluate(residues)
                
                if new_score > current_score:
                    current_score = new_score
                    if new_score > best_score:
                        best_score = new_score
                        best_residues = residues.copy()
                else:
                    delta = new_score - current_score
                    if delta < 0:
                        try:
                            prob = np.exp(delta / T)
                        except OverflowError:
                            prob = 0.0
                        if random.random() < prob:
                            current_score = new_score
                        else:
                            residues[i] = old_r
                    else:
                        current_score = new_score
            else:
                # If already >= 2 covering, or no non-covering (impossible), do random move
                i = random.randint(0, N_PRIMES - 1)
                p = PRIMES[i]
                old_r = residues[i]
                new_r = random.randint(0, p - 1)
                if new_r != old_r:
                    residues[i] = new_r
                    new_score = evaluate(residues)
                    if new_score > current_score:
                        current_score = new_score
                        if new_score > best_score:
                            best_score = new_score
                            best_residues = residues.copy()
                    else:
                        delta = new_score - current_score
                        if delta < 0:
                            try:
                                prob = np.exp(delta / T)
                            except OverflowError:
                                prob = 0.0
                            if random.random() < prob:
                                current_score = new_score
                            else:
                                residues[i] = old_r
                        else:
                            current_score = new_score
        else:
            # No failure found in 1..M, score is M
            # Just do random moves to see if we can go higher? 
            # But M is the cap for evaluation. 
            # If we reach M, we are likely optimal for this M.
            # We can try to increase M? No, M is fixed.
            # Just break or do random moves.
            i = random.randint(0, N_PRIMES - 1)
            p = PRIMES[i]
            old_r = residues[i]
            new_r = random.randint(0, p - 1)
            if new_r != old_r:
                residues[i] = new_r
                new_score = evaluate(residues)
                if new_score > current_score:
                    current_score = new_score
                    if new_score > best_score:
                        best_score = new_score
                        best_residues = residues.copy()
                else:
                    delta = new_score - current_score
                    if delta < 0:
                        try:
                            prob = np.exp(delta / T)
                        except OverflowError:
                            prob = 0.0
                        if random.random() < prob:
                            current_score = new_score
                        else:
                            residues[i] = old_r
                    else:
                        current_score = new_score

        T *= cooling_rate

    return best_residues
