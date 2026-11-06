# EVOLVE-BLOCK-START
import time
import random
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 112.0  # Safety margin (150s limit, leave buffer)

    # Generate primes up to 1000
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n ** 0.5) + 1):
            if is_prime[i]:
                for j in range(i * i, n + 1, i):
                    is_prime[j] = False
        return [i for i, prime in enumerate(is_prime) if prime]
    
    primes = sieve(1000)
    n_primes = len(primes)
    primes_list = primes

    # Evaluation bound
    EVAL_BOUND = 10000

    def compute_counts_fast(residues):
        """Compute coverage counts for positions 1..EVAL_BOUND using vectorized ops."""
        counts = np.zeros(EVAL_BOUND + 1, dtype=np.int32)
        for i in range(n_primes):
            p = primes_list[i]
            r = residues[i]
            if r == 0:
                first = p
            else:
                first = r
            if first >= 1 and first <= EVAL_BOUND:
                idx = np.arange(first, EVAL_BOUND + 1, p)
                counts[idx] += 1
        return counts

    def get_L_from_counts(counts):
        """Find largest L such that counts[1..L] >= 2."""
        below = np.where(counts[1:] < 2)[0]
        if len(below) == 0:
            return EVAL_BOUND
        return int(below[0])

    def evaluate(residues):
        counts = compute_counts_fast(residues)
        return get_L_from_counts(counts)

    # Initialization: Random + Quick Greedy Fill
    random.seed(42)
    current = [random.randint(0, p - 1) for p in primes_list]
    
    # Quick Greedy Fill for first 200 positions to get a decent start
    for pos in range(1, 201):
        cnt = 0
        for i in range(n_primes):
            if current[i] == pos % primes_list[i]:
                cnt += 1
        if cnt >= 2:
            continue
        # Find primes not covering pos
        candidates = [i for i in range(n_primes) if current[i] != pos % primes_list[i]]
        if candidates:
            idx = random.choice(candidates)
            current[idx] = pos % primes_list[idx]

    best_residues = current[:]
    best_L = evaluate(best_residues)
    current_L = best_L

    # Iterative Gap Repair
    no_improve_count = 0
    max_no_improve = 2000
    
    while time.time() < deadline - 2.0:
        # Check if we need to kick out of local minimum
        if no_improve_count > max_no_improve:
            # Perturb current solution
            for _ in range(10):
                idx = random.randint(0, n_primes - 1)
                current[idx] = random.randint(0, primes_list[idx] - 1)
            current_L = evaluate(current)
            if current_L > best_L:
                best_L = current_L
                best_residues = current[:]
            no_improve_count = 0
            # Reduce patience to try more kicks if stuck
            max_no_improve = max(500, int(max_no_improve * 0.9))
            continue
        
        # Target the gap at current_L + 1
        gap_pos = current_L + 1
        if gap_pos > EVAL_BOUND:
            break
            
        # Find primes that do NOT cover gap_pos
        candidates = []
        for i in range(n_primes):
            if current[i] != gap_pos % primes_list[i]:
                candidates.append(i)
                
        if not candidates:
            # All primes already cover gap_pos? Then count at gap_pos is 168.
            # This implies gap_pos is not the first gap < 2, contradiction.
            # Unless gap_pos > EVAL_BOUND.
            break
            
        # Sample up to 50 candidates to try
        random.shuffle(candidates)
        improved = False
        for i in candidates[:50]:
            p = primes_list[i]
            old_r = current[i]
            new_r = gap_pos % p
            
            current[i] = new_r
            new_L = evaluate(current)
            
            if new_L > current_L:
                current_L = new_L
                if current_L > best_L:
                    best_L = current_L
                    best_residues = current[:]
                no_improve_count = 0
                improved = True
                break
            else:
                current[i] = old_r
        
        if not improved:
            no_improve_count += 1
            
    return [int(r) for r in best_residues]
# EVOLVE-BLOCK-END
