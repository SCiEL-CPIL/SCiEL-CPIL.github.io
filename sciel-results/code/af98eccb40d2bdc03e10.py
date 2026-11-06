"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import math
import time

def _is_coprime_pairwise(vals):
    """Check if all distinct values in vals are pairwise coprime."""
    distinct = list(set(vals))
    n = len(distinct)
    for i in range(n):
        for j in range(i + 1, n):
            if math.gcd(distinct[i], distinct[j]) != 1:
                return False
    return True

def _evaluate(A, B):
    """Return the number of distinct values in A+B if they are pairwise coprime, else 0."""
    sums = [a + b for a in A for b in B]
    distinct = list(set(sums))
    if _is_coprime_pairwise(distinct):
        return len(distinct)
    return 0

def construct():
    """
    Find A, B (each 3 distinct ints in [1,128]) maximizing |A+B| distinct values
    that are pairwise coprime.
    """
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s limit
    
    # Precompute: for each pair (a, b) in [1,128], the sum a+b
    # We'll do a structured search
    
    best_score = 0
    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    
    # First, try the seed
    seed_A = [1, 7, 13]
    seed_B = [4, 10, 16]
    seed_score = _evaluate(seed_A, seed_B)
    best_score = max(best_score, seed_score)
    if seed_score > 0:
        best_A, best_B = seed_A[:], seed_B[:]
    
    # Strategy: search over A and B using primes and prime-related numbers
    # to maximize pairwise coprime sums
    
    # Generate candidate values: primes, 1, and small numbers
    candidates = list(range(1, 129))
    
    # Focus on values that tend to produce coprime sums
    # Primes > 64 are great since their sums with small numbers are likely prime or coprime
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127]
    
    # Try all combinations from a focused candidate set
    # Use a subset of candidates to keep search tractable
    # Focus on: 1, small primes, large primes
    focused = [1] + primes[:15] + primes[-10:]
    focused = sorted(set(focused))
    
    # Backtracking search
    def backtrack(A_partial, B_partial, A_idx, B_idx):
        nonlocal best_score, best_A, best_B
        if time.time() > deadline:
            return
        
        # Pruning: if we already have 3 in both, evaluate
        if len(A_partial) == 3 and len(B_partial) == 3:
            score = _evaluate(A_partial, B_partial)
            if score > best_score:
                best_score = score
                best_A = A_partial[:]
                best_B = B_partial[:]
            return
        
        if len(A_partial) == 3:
            # Complete B
            for b in focused:
                if b in B_partial:
                    continue
                B_partial.append(b)
                backtrack(A_partial, B_partial, A_idx, B_idx)
                B_partial.pop()
                if time.time() > deadline:
                    return
            return
        
        if len(B_partial) == 3:
            # Complete A
            for a in focused:
                if a in A_partial:
                    continue
                A_partial.append(a)
                backtrack(A_partial, B_partial, A_idx, B_idx)
                A_partial.pop()
                if time.time() > deadline:
                    return
            return
        
        # Choose next element
        if A_idx <= B_idx:
            for a in focused:
                if a in A_partial:
                    continue
                A_partial.append(a)
                backtrack(A_partial, B_partial, A_idx + 1, B_idx)
                A_partial.pop()
                if time.time() > deadline:
                    return
        else:
            for b in focused:
                if b in B_partial:
                    continue
                B_partial.append(b)
                backtrack(A_partial, B_partial, A_idx, B_idx + 1)
                B_partial.pop()
                if time.time() > deadline:
                    return
    
    # Try full backtracking on focused set
    backtrack([], [], 0, 0)
    
    # If we didn't find 9, try local search / random perturbation
    if best_score < 9:
        import random
        random.seed(42)
        for trial in range(5000):
            if time.time() > deadline:
                break
            A = random.sample(range(1, 129), 3)
            B = random.sample(range(1, 129), 3)
            score = _evaluate(A, B)
            if score > best_score:
                best_score = score
                best_A, best_B = A[:], B[:]
                if best_score == 9:
                    break
    
    # Local improvement
    if best_score < 9 and time.time() < deadline - 5:
        import random
        random.seed(123)
        current_A, current_B = best_A[:], best_B[:]
        current_score = best_score
        for step in range(20000):
            if time.time() > deadline - 2:
                break
            # Pick a random element from A or B and perturb
            if random.random() < 0.5:
                idx = random.randint(0, 2)
                old = current_A[idx]
                new = random.randint(1, 128)
                if new in current_A and new != old:
                    continue
                current_A[idx] = new
                score = _evaluate(current_A, current_B)
                if score > current_score:
                    current_score = score
                    best_score = score
                    best_A, best_B = current_A[:], current_B[:]
                    if best_score == 9:
                        break
                else:
                    current_A[idx] = old
            else:
                idx = random.randint(0, 2)
                old = current_B[idx]
                new = random.randint(1, 128)
                if new in current_B and new != old:
                    continue
                current_B[idx] = new
                score = _evaluate(current_A, current_B)
                if score > current_score:
                    current_score = score
                    best_score = score
                    best_A, best_B = current_A[:], current_B[:]
                    if best_score == 9:
                        break
                else:
                    current_B[idx] = old
    
    return [best_A, best_B]
# EVOLVE-BLOCK-END
