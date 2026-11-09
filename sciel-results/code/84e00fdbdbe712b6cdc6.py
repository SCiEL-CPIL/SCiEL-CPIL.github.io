"""Rewritten program using structured number-theoretic search for pairwise-coprime sumsets."""
import math
import random
import time

# EVOLVE-BLOCK-START
def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def _is_pairwise_coprime(vals):
    n = len(vals)
    for i in range(n):
        for j in range(i + 1, n):
            if _gcd(vals[i], vals[j]) != 1:
                return False
    return True

def _score(A, B):
    """Return (distinct_count, is_valid). Higher distinct_count with valid is better."""
    sums = set()
    for a in A:
        for b in B:
            sums.add(a + b)
    distinct = sorted(sums)
    if not _is_pairwise_coprime(distinct):
        return (len(distinct), False)
    return (len(distinct), True)

def _try_improve(A, B, deadline, rng):
    """Hill-climbing: try swapping one element at a time."""
    current_count, current_valid = _score(A, B)
    best_A, best_B = list(A), list(B)
    best_count = current_count
    best_valid = current_valid
    
    for _ in range(5000):
        if time.time() > deadline:
            break
        
        # Pick a random element from A or B to perturb
        if rng.random() < 0.5:
            idx = rng.randrange(3)
            old_val = best_A[idx]
            new_val = rng.randint(1, 128)
            if new_val == old_val:
                continue
            trial_A = list(best_A)
            trial_A[idx] = new_val
            # Ensure distinct within A
            if len(set(trial_A)) != 3:
                continue
            count, valid = _score(trial_A, best_B)
            if valid and count > best_count:
                best_A, best_B = trial_A, list(best_B)
                best_count, best_valid = count, True
            elif valid and count == best_count and best_valid:
                # Accept lateral moves for diversity
                if rng.random() < 0.3:
                    best_A = trial_A
            elif not valid and best_valid:
                pass  # reject invalid if currently valid
            elif not valid and not best_valid:
                if count > best_count:
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
            elif valid and count == best_count and best_valid:
                if rng.random() < 0.3:
                    best_B = trial_B
            elif not valid and best_valid:
                pass
            elif not valid and not best_valid:
                if count > best_count:
                    best_A, best_B = list(best_A), trial_B
                    best_count, best_valid = count, False
    
    return best_A, best_B, best_count, best_valid

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    rng = random.Random(42)
    
    # Start with a known valid solution
    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    best_count, best_valid = _score(best_A, best_B)
    
    # Try multiple restarts with different seeds
    for seed in range(200):
        if time.time() > deadline:
            break
        
        # Generate a diverse starting point
        A = sorted(rng.sample(range(1, 129), 3))
        B = sorted(rng.sample(range(1, 129), 3))
        
        # Local improvement
        A2, B2, cnt, valid = _try_improve(A, B, deadline, rng)
        
        if valid and cnt > best_count:
            best_A, best_B = A2, B2
            best_count, best_valid = cnt, True
        elif not best_valid and cnt > best_count:
            best_A, best_B = A2, B2
            best_count, best_valid = cnt, False
    
    # Final refinement from the best found
    if time.time() < deadline - 2:
        A2, B2, cnt, valid = _try_improve(best_A, best_B, deadline, rng)
        if valid and cnt > best_count:
            best_A, best_B = A2, B2
            best_count, best_valid = cnt, True
        elif not best_valid and cnt > best_count:
            best_A, best_B = A2, B2
            best_count, best_valid = cnt, False
    
    return [best_A, best_B]
# EVOLVE-BLOCK-END
