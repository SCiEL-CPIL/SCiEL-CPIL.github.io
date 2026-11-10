"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from itertools import combinations

def _valid_pairwise_coprime(sums_set):
    """Check if all distinct values in sums_set are pairwise coprime."""
    vals = list(sums_set)
    for i in range(len(vals)):
        for j in range(i + 1, len(vals)):
            if gcd(vals[i], vals[j]) != 1:
                return False
    return True

def _score(A, B):
    """Return the number of distinct values in A+B if they are pairwise coprime, else 0."""
    sums = set()
    for a in A:
        for b in B:
            sums.add(a + b)
    if _valid_pairwise_coprime(sums):
        return len(sums)
    return 0

def _local_search(A, B, max_iters=5000):
    """Local search to improve the configuration."""
    A = list(A)
    B = list(B)
    best_score = _score(A, B)
    
    for _ in range(max_iters):
        # Randomly pick which set to modify and which element
        if random.random() < 0.5:
            idx = random.randint(0, 2)
            new_val = random.randint(1, 128)
            old_val = A[idx]
            A[idx] = new_val
            if len(set(A)) < 3:
                A[idx] = old_val
                continue
            new_score = _score(A, B)
            if new_score > best_score:
                best_score = new_score
            else:
                A[idx] = old_val
        else:
            idx = random.randint(0, 2)
            new_val = random.randint(1, 128)
            old_val = B[idx]
            B[idx] = new_val
            if len(set(B)) < 3:
                B[idx] = old_val
                continue
            new_score = _score(A, B)
            if new_score > best_score:
                best_score = new_score
            else:
                B[idx] = old_val
    
    return A, B, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s limit
    
    # Start from the seed
    best_A, best_B = [1, 7, 13], [4, 10, 16]
    best_score = _score(best_A, best_B)
    
    # Try several random restarts with local search
    random.seed(42)
    restarts = 0
    while time.time() < deadline:
        # Generate a random starting point
        A = random.sample(range(1, 129), 3)
        B = random.sample(range(1, 129), 3)
        
        # Also try starting from the current best with small perturbations
        if random.random() < 0.3:
            A = list(best_A)
            B = list(best_B)
            # Perturb one element
            if random.random() < 0.5:
                idx = random.randint(0, 2)
                A[idx] = random.randint(1, 128)
            else:
                idx = random.randint(0, 2)
                B[idx] = random.randint(1, 128)
        
        # Run local search
        A, B, score = _local_search(A, B, max_iters=2000)
        
        if score > best_score:
            best_score = score
            best_A, best_B = A, B
        
        restarts += 1
        # Check if we've hit the theoretical max (9 distinct sums)
        if best_score == 9:
            break
    
    return [best_A, best_B]
# EVOLVE-BLOCK-END
