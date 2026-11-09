"""Rewritten program using structured number-theoretic search for pairwise-coprime sumsets."""
import math
import random
import time
from math import gcd

# EVOLVE-BLOCK-START

def _is_pairwise_coprime(vals):
    """Check if all values in vals are pairwise coprime."""
    n = len(vals)
    for i in range(n):
        for j in range(i + 1, n):
            if gcd(vals[i], vals[j]) != 1:
                return False
    return True

def _score(A, B):
    """Return (distinct_count, is_valid). Higher distinct_count with valid is better."""
    sums = set()
    for a in A:
        for b in B:
            sums.add(a + b)
    distinct = list(sums)
    if not _is_pairwise_coprime(distinct):
        return (len(distinct), False)
    return (len(distinct), True)

def _try_improve(A, B, deadline, rng):
    """Hill-climbing with targeted perturbations."""
    best_A, best_B = list(A), list(B)
    best_count, best_valid = _score(best_A, best_B)
    
    # Precompute current sums for quick comparison
    current_sums = set()
    for a in best_A:
        for b in best_B:
            current_sums.add(a + b)
    
    for iteration in range(3000):
        if time.time() > deadline:
            break
        
        # Try perturbing one element
        if rng.random() < 0.5:
            idx = rng.randrange(3)
            old_val = best_A[idx]
            new_val = rng.randint(1, 128)
            if new_val == old_val:
                continue
            trial_A = list(best_A)
            trial_A[idx] = new_val
            if len(set(trial_A)) != 3:
                continue
            count, valid = _score(trial_A, best_B)
            if valid and count > best_count:
                best_A, best_B = trial_A, list(best_B)
                best_count, best_valid = count, True
            elif valid and count == best_count and best_valid and rng.random() < 0.2:
                best_A = trial_A
            elif not best_valid and count > best_count:
                best_A, best_B = trial_A, list(best_B)
                best_count, best_valid = count, False
        else:
            idx = rng.randrange(3)
            old_val = best_B[idx]
            new_val = rng.randint(1, 128)
            if new_val == old_val:
                continue
            trial_B = list(best_B)
            trial_B[idx] = new_val
            if len(set(trial_B)) != 3:
                continue
            count, valid = _score(best_A, trial_B)
            if valid and count > best_count:
                best_A, best_B = list(best_A), trial_B
                best_count, best_valid = count, True
            elif valid and count == best_count and best_valid and rng.random() < 0.2:
                best_B = trial_B
            elif not best_valid and count > best_count:
                best_A, best_B = list(best_A), trial_B
                best_count, best_valid = count, False
    
    return best_A, best_B, best_count, best_valid

def _generate_prime_based_start(rng):
    """Generate starting points biased toward primes and numbers with few small factors."""
    # Primes up to 128
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127]
    
    # Mix of primes and other numbers
    if rng.random() < 0.7:
        A = sorted(rng.sample(primes, 3))
        B = sorted(rng.sample(primes, 3))
    else:
        # Use numbers that are 1 or 2 mod small primes
        candidates = [x for x in range(1, 129) if x not in [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]]
        A = sorted(rng.sample(candidates, 3))
        B = sorted(rng.sample(candidates, 3))
    
    return A, B

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    rng = random.Random(42)
    
    # Start with a known valid solution
    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    best_count, best_valid = _score(best_A, best_B)
    
    # If we already have 9 (maximum), return immediately
    if best_count == 9 and best_valid:
        return [best_A, best_B]
    
    # Try multiple restarts with different strategies
    for seed in range(150):
        if time.time() > deadline:
            break
        
        # Use different initialization strategies
        if seed % 3 == 0:
            A, B = _generate_prime_based_start(rng)
        elif seed % 3 == 1:
            A = sorted(rng.sample(range(1, 129), 3))
            B = sorted(rng.sample(range(1, 129), 3))
        else:
            # Bias toward small numbers to keep sums in a reasonable range
            small_range = range(1, 65)
            A = sorted(rng.sample(small_range, 3))
            B = sorted(rng.sample(small_range, 3))
        
        # Local improvement
        A2, B2, cnt, valid = _try_improve(A, B, deadline, rng)
        
        if valid and cnt > best_count:
            best_A, best_B = A2, B2
            best_count, best_valid = cnt, True
            if best_count == 9:
                break
        elif not best_valid and cnt > best_count:
            best_A, best_B = A2, B2
            best_count, best_valid = cnt, False
    
    # Final refinement from the best found
    if time.time() < deadline - 2 and best_count < 9:
        A2, B2, cnt, valid = _try_improve(best_A, best_B, deadline, rng)
        if valid and cnt > best_count:
            best_A, best_B = A2, B2
            best_count, best_valid = cnt, True
        elif not best_valid and cnt > best_count:
            best_A, best_B = A2, B2
            best_count, best_valid = cnt, False
    
    return [best_A, best_B]
# EVOLVE-BLOCK-END
