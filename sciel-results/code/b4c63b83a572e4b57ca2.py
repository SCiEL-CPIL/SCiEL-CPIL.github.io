"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import isqrt, gcd

N = 256
DEADLINE = 85.0  # safety margin under 90s wall-clock

def _valid_set(S):
    """Check that for all distinct a,b,c in S, a does not divide b*c."""
    lst = list(S)
    n = len(lst)
    for i in range(n):
        a = lst[i]
        for j in range(i + 1, n):
            b = lst[j]
            for k in range(j + 1, n):
                c = lst[k]
                if (b * c) % a == 0:
                    return False
    return True

def _can_add(S, x):
    """Check if x can be added to S without violating the property."""
    lst = list(S)
    for i in range(len(lst)):
        a = lst[i]
        for j in range(i + 1, len(lst)):
            b = lst[j]
            if (b * x) % a == 0 or (a * x) % b == 0:
                return False
    return True

def construct():
    start = time.time()
    
    # Strategy: greedy construction starting from upper half, then try to insert lower numbers
    # The key insight: numbers in (128, 256] are large, so it's harder for a small a to divide b*c
    # when a is large. Start with a good base set, then greedily add more.
    
    # Phase 1: Build initial set using greedy from top down
    # Start with numbers in (N/2, N] that are "safe"
    base = set()
    candidates_top = list(range(N, N // 2, -1))  # 256 down to 129
    
    for x in candidates_top:
        if time.time() - start > DEADLINE:
            break
        if _can_add(base, x):
            base.add(x)
    
    # Phase 2: Try to add numbers from the lower range (2 to N//2)
    # Sort by descending value (larger numbers are easier to add)
    candidates_low = list(range(N // 2, 1, -1))
    
    for x in candidates_low:
        if time.time() - start > DEADLINE:
            break
        if _can_add(base, x):
            base.add(x)
    
    current_best = list(base)
    best_size = len(current_best)
    
    # Phase 3: Local search - try swapping elements to improve
    # Try removing one element and adding another
    remaining_time = DEADLINE - (time.time() - start)
    if remaining_time > 2.0:
        all_nums = list(range(2, N + 1))
        in_set = set(current_best)
        not_in_set = [x for x in all_nums if x not in in_set]
        
        improved = True
        while improved and (time.time() - start) < DEADLINE:
            improved = False
            # Try all pairs (remove r, add a)
            for r in list(current_best):
                if (time.time() - start) > DEADLINE:
                    break
                temp_set = set(current_best)
                temp_set.discard(r)
                for a in not_in_set:
                    if (time.time() - start) > DEADLINE:
                        break
                    if a in temp_set:
                        continue
                    # Check if adding a to temp_set is valid
                    if _can_add(temp_set, a):
                        new_set = temp_set | {a}
                        # Verify full validity
                        if _valid_set(new_set):
                            current_best = list(new_set)
                            best_size = len(current_best)
                            in_set = new_set
                            not_in_set = [x for x in all_nums if x not in in_set]
                            improved = True
                            break
                if improved:
                    break
        
        # Phase 4: Try adding more elements after swaps
        for x in not_in_set:
            if (time.time() - start) > DEADLINE:
                break
            if _can_add(set(current_best), x):
                current_best.append(x)
                best_size = len(current_best)
    
    # Final verification
    if not _valid_set(set(current_best)):
        # Fallback: return just the primes (known valid)
        result = [a for a in range(2, N + 1) if all(a % d for d in range(2, isqrt(a) + 1))]
        return result
    
    return current_best
# EVOLVE-BLOCK-END
