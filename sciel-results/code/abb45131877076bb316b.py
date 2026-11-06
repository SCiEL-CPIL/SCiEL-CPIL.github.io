import time
import random
from math import gcd

def _pairwise_coprime_count(sums):
    """Count distinct sums that are pairwise coprime."""
    distinct = sorted(set(sums))
    count = 0
    for i in range(len(distinct)):
        ok = True
        for j in range(i+1, len(distinct)):
            if gcd(distinct[i], distinct[j]) != 1:
                ok = False
                break
        if ok:
            # Check all pairs from i onward with i
            pass
    # Better: find max subset that is pairwise coprime
    # But the task says "ALL DISTINCT values in A+B must be pairwise coprime"
    # So we need ALL distinct sums to be pairwise coprime
    # The score is the cardinality of A+B (number of distinct sums)
    # but only if they are ALL pairwise coprime
    # Wait, re-reading: "All DISTINCT values in A+B must be pairwise coprime. Maximize the cardinality of A+B"
    # So the constraint is that all distinct sums must be pairwise coprime, and we maximize the count.
    # If not all are pairwise coprime, the result is invalid (score 0)
    
    # Let me re-read: "Return two lists A,B, each containing exactly 3 distinct integers in [1,128]. 
    # All DISTINCT values in A+B must be pairwise coprime. Maximize the cardinality of A+B"
    
    # So the constraint is a hard constraint: all distinct sums must be pairwise coprime.
    # The objective is to maximize the number of distinct sums.
    
    # For evaluation, I think the score is just the number of distinct sums, 
    # but only if the pairwise coprime constraint is satisfied.
    # Otherwise score is 0.
    
    return len(distinct)

def _is_valid_pairwise_coprime(sums):
    """Check if all distinct sums are pairwise coprime."""
    distinct = sorted(set(sums))
    for i in range(len(distinct)):
        for j in range(i+1, len(distinct)):
            if gcd(distinct[i], distinct[j]) != 1:
                return False
    return True

def _score(A, B):
    """Score: number of distinct sums if all pairwise coprime, else 0."""
    sums = [a + b for a in A for b in B]
    if _is_valid_pairwise_coprime(sums):
        return len(set(sums))
    return 0

def _evaluate(A, B):
    """Evaluate and return (score, sums_list)."""
    sums = [a + b for a in A for b in B]
    if _is_valid_pairwise_coprime(sums):
        return len(set(sums)), sums
    return 0, sums

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Leave 5s margin
    
    best_score = 5
    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    
    # Precompute primes up to 256
    def is_prime(n):
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        return True
    
    primes = [p for p in range(2, 257) if is_prime(p)]
    
    # Strategy 1: Systematic search over structured sets
    # Try sets where elements are small and spread out
    
    def try_candidate(A, B):
        nonlocal best_score, best_A, best_B
        score, _ = _evaluate(A, B)
        if score > best_score:
            best_score = score
            best_A = A[:]
            best_B = B[:]
    
    # Try some initial structured candidates
    initial_candidates = [
        ([1, 2, 3], [4, 6, 8]),
        ([1, 2, 4], [3, 5, 7]),
        ([1, 3, 5], [2, 4, 6]),
        ([1, 2, 4], [5, 7, 9]),
        ([1, 3, 7], [2, 4, 6]),
        ([1, 2, 5], [3, 6, 7]),
        ([1, 3, 4], [2, 5, 7]),
        ([1, 2, 3], [5, 7, 11]),
        ([1, 2, 4], [7, 11, 13]),
        ([1, 3, 7], [4, 8, 12]),
    ]
    
    for A, B in initial_candidates:
        try_candidate(A, B)
    
    # Strategy 2: Local search / hill climbing
    # Start from best candidate and try swapping elements
    
    def local_search(A, B, max_iters=100000):
        nonlocal best_score, best_A, best_B
        current_A = A[:]
        current_B = B[:]
        current_score, _ = _evaluate(current_A, current_B)
        
        for iteration in range(max_iters):
            if time.time() > deadline:
                break
            
            # Try swapping one element in A or B
            improved = False
            for set_idx in range(2):  # 0 for A, 1 for B
                for pos in range(3):
                    old_val = current_A[pos] if set_idx == 0 else current_B[pos]
                    for new_val in range(1, 129):
                        if new_val == old_val:
                            continue
                        # Check distinctness
                        if set_idx == 0:
                            if new_val in current_A and new_val != old_val:
                                continue
                            new_A = current_A[:]
                            new_A[pos] = new_val
                            new_B = current_B
                        else:
                            if new_val in current_B and new_val != old_val:
                                continue
                            new_A = current_A
                            new_B = current_B[:]
                            new_B[pos] = new_val
                        
                        score, _ = _evaluate(new_A, new_B)
                        if score > current_score:
                            current_A = new_A
                            current_B = new_B
                            current_score = score
                            if score > best_score:
                                best_score = score
                                best_A = current_A[:]
                                best_B = current_B[:]
                            improved = True
                            break
                        elif score == current_score and score > 0:
                            # Keep for diversity, but don't count as improvement
                            pass
                    
                    if improved:
                        break
                if improved:
                    break
            
            if not improved:
                break
        
        return current_score
    
    # Run local search from multiple starting points
    start_points = [
        [1, 7, 13],
        [1, 2, 4],
        [1, 3, 5],
        [2, 3, 5],
        [1, 2, 3],
        [3, 5, 7],
    ]
    
    start_points_B = [
        [4, 10, 16],
        [3, 5, 7],
        [2, 4, 6],
        [1, 3, 5],
        [4, 6, 8],
        [5, 7, 11],
    ]
    
    for A_start in start_points:
        if time.time() > deadline:
            break
        for B_start in start_points_B:
            if time.time() > deadline:
                break
            local_search(A_start, B_start, max_iters=50000)
    
    # Strategy 3: Random restarts with local search
    random.seed(42)
    for _ in range(50):
        if time.time() > deadline:
            break
        
        A = random.sample(range(1, 129), 3)
        B = random.sample(range(1, 129), 3)
        local_search(A, B, max_iters=20000)
    
    # Final verification
    final_score, _ = _evaluate(best_A, best_B)
    if final_score < 5:
        # Fallback to known valid solution
        return [[1, 7, 13], [4, 10, 16]]
    
    return [best_A, best_B]
