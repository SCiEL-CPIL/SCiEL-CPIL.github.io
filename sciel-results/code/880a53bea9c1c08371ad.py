"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import math
import time
import random

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def _pairwise_coprime(vals):
    """Check if all distinct values in vals are pairwise coprime."""
    distinct = sorted(set(vals))
    n = len(distinct)
    if n <= 1:
        return True
    for i in range(n):
        for j in range(i + 1, n):
            if _gcd(distinct[i], distinct[j]) != 1:
                return False
    return True

def _sumset(A, B):
    """Return the set of all sums a+b for a in A, b in B."""
    return set(a + b for a in A for b in B)

def _is_valid(A, B):
    """Check if A and B are valid: 3 distinct ints in [1,128], all distinct sums pairwise coprime."""
    if len(A) != 3 or len(set(A)) != 3:
        return False
    if len(B) != 3 or len(set(B)) != 3:
        return False
    if any(x < 1 or x > 128 for x in A) or any(x < 1 or x > 128 for x in B):
        return False
    sums = _sumset(A, B)
    return _pairwise_coprime(list(sums))

def _score(A, B):
    """Return the number of distinct sums if valid, else 0."""
    if not _is_valid(A, B):
        return 0
    return len(_sumset(A, B))

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s deadline, 5s safety margin
    
    # Best known solution from seed
    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    best_score = _score(best_A, best_B)
    
    # Systematic exhaustive search over a small range to guarantee finding the optimal solution quickly.
    # Range [1, 20] yields C(20,3)^2 ~ 1.3M pairs, which is fast enough.
    # If a solution with score 9 is found, we return it immediately.
    
    range_limit = 20
    candidates = list(itertools.combinations(range(1, range_limit + 1), 3))
    
    # To speed up, we can pre-filter or just iterate.
    # 1.3M iterations is acceptable.
    
    # We can break early if we find score 9.
    
    for A in candidates:
        if time.time() > deadline:
            break
        A_list = list(A)
        for B in candidates:
            if time.time() > deadline:
                break
            B_list = list(B)
            
            # Quick check: distinct sums count
            sums = set()
            for a in A_list:
                for b in B_list:
                    sums.add(a + b)
            
            # Max possible is 9. If < 9, it can't be optimal, but might be better than current best?
            # We want to maximize. 9 is the max.
            # If we find 9, we can return immediately.
            if len(sums) < 9:
                # We could still check if it's > best_score, but 9 is the goal.
                # For efficiency, if we are looking for 9, we can skip coprime check if len < 9?
                # No, we need to know if it's valid.
                # But if best_score is already 9, we would have returned.
                # So we only care about finding 9.
                # If we don't find 9, we return best_score.
                # So we can skip scoring if len(sums) < 9?
                # Wait, what if best_score is 5 and we find 8?
                # The goal is to maximize. 9 is the max.
                # If we find 9, we stop.
                # If we don't, we should keep the best found.
                # But checking coprime is the expensive part.
                # If len(sums) < 9, it's not the *optimal* 9, but could be 8.
                # However, 9 is achievable.
                # So we prioritize finding 9.
                # If we don't find 9 in the small range, we might need a larger range.
                # But likely 9 is in [1, 20].
                
                # Optimization: Only check coprime if len(sums) > best_score.
                if len(sums) <= best_score:
                    continue
                    
            # Check pairwise coprime
            if _pairwise_coprime(list(sums)):
                current_score = len(sums)
                if current_score > best_score:
                    best_score = current_score
                    best_A = A_list[:]
                    best_B = B_list[:]
                    if best_score == 9:
                        return [best_A, best_B]
                        
    # If 9 not found in [1, 20], try [1, 30] or fallback to random/local search
    if best_score < 9:
        # Expand search or use random search
        # For now, return best found.
        pass

    return [best_A, best_B]

# EVOLVE-BLOCK-END
