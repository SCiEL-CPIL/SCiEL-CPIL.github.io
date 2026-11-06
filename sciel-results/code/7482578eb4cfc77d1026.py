from fractions import Fraction
import random

def _check_valid(S):
    """Check if all reciprocal subset sums are distinct."""
    sums = {Fraction(0, 1)}
    for x in S:
        new_sums = set()
        for s in sums:
            new_sums.add(s + Fraction(1, x))
        if sums & new_sums:
            return False
        sums |= new_sums
    return True

def construct():
    # Fallback to seed if search fails
    seed = [1, 2, 3, 5, 7, 11, 13, 17, 19]
    best_set = seed
    best_size = len(seed)
    
    # 1. Greedy approach: Add largest elements first
    # This tends to keep subset sums small and distinct
    candidates = list(range(22, 0, -1))
    current_set = []
    for x in candidates:
        new_set = current_set + [x]
        if _check_valid(new_set):
            current_set = new_set
            if len(current_set) > best_size:
                best_set = current_set
                best_size = len(current_set)
                
    # 2. Randomized search for a potentially larger set
    # Try to find a set of size best_size + 1
    target_size = best_size + 1
    # Limit iterations to ensure speed
    for _ in range(200):
        S = random.sample(range(1, 23), target_size)
        if _check_valid(S):
            best_set = S
            best_size = target_size
            # Try to go even bigger?
            target_size += 1
            if target_size > 12: # Cap search depth
                break
                
    return best_set
