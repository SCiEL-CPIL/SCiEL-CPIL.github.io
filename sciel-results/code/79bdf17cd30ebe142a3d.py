"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from itertools import combinations
from functools import lru_cache

def construct():
    start_time = time.time()
    timeout = 85.0  # Safety margin below 90s limit
    
    # Generate all 5-smooth numbers up to 128
    smooth_numbers = []
    for x in range(1, 129):
        y = x
        for p in (2, 3, 5):
            while y % p == 0:
                y //= p
        if y == 1:
            smooth_numbers.append(x)
    
    n_smooth = len(smooth_numbers)
    smooth_set = set(smooth_numbers)
    
    # Precompute conflict: two numbers conflict if one divides the other
    # For antichain: no pair in the subset can have divisibility relation
    
    # Precompute all valid antichain subsets of size up to 8
    # Store as: antichain_by_sum[sum_value] = list of (size, tuple_of_values)
    # We want minimum size for each sum
    
    # First, build a set of smooth numbers for quick lookup
    smooth_idx = {v: i for i, v in enumerate(smooth_numbers)}
    
    # For each subset size k from 1 to 8, enumerate all combinations
    # and check if they form an antichain, then record the sum
    
    # To be efficient, only consider subsets where all elements <= 128
    # and sum <= 128
    
    max_k = 8
    # antichain_solutions[sum] = (min_size, tuple)
    antichain_solutions = {}
    
    for k in range(1, max_k + 1):
        if time.time() - start_time > timeout - 5:
            break
        for combo in combinations(smooth_numbers, k):
            s = sum(combo)
            if s > 128:
                continue
            # Check if this is an antichain
            if _is_antichain(combo):
                # If we haven't seen this sum, or this size is smaller
                if s not in antichain_solutions or k < antichain_solutions[s][0]:
                    antichain_solutions[s] = (k, combo)
    
    # Build answers
    answers = []
    for n in range(1, 129):
        if time.time() - start_time > timeout:
            # Fallback
            answers.append([n] if n in smooth_set else [1, n-1])
            continue
        
        if n in antichain_solutions:
            k, combo = antichain_solutions[n]
            answers.append(list(combo))
        else:
            # Fallback: try to find any valid representation
            # Simple greedy: use n itself if smooth, else try pairs
            if n in smooth_set:
                answers.append([n])
            else:
                # Try to find a valid pair
                found = None
                for a in smooth_numbers:
                    if a >= n:
                        break
                    b = n - a
                    if b in smooth_set and b > a:
                        if b % a != 0:
                            found = [a, b]
                            break
                if found is None:
                    # Last resort: just use [n] even if not smooth (shouldn't happen for valid task)
                    found = [n]
                answers.append(found)
    
    return answers

def _is_antichain(combo):
    """Check if no element divides another in the combo."""
    n = len(combo)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = combo[i], combo[j]
            if a > b:
                a, b = b, a
            if b % a == 0:
                return False
    return True
# EVOLVE-BLOCK-END
