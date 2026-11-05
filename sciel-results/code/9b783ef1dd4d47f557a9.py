import time
from fractions import Fraction
import random

def _precompute_conflicts():
    """
    For each x in 1..24, find all subsets Y of [1,24]\{x} such that sum(1/y for y in Y) == 1/x.
    Returns a list of lists of bitmasks.
    """
    universe = list(range(1, 25))
    conflicts = [[] for _ in range(25)] # 1-indexed
    
    for x in range(1, 25):
        target = Fraction(1, x)
        candidates = [y for y in universe if y != x]
        
        # Sort candidates by value ascending (reciprocals descending)
        candidates_sorted = sorted(candidates)
        n = len(candidates_sorted)
        reciprocals = [Fraction(1, y) for y in candidates_sorted]
        
        # Precompute suffix sums of reciprocals for pruning
        suffix_sums = [Fraction(0, 1)] * (n + 1)
        for i in range(n - 1, -1, -1):
            suffix_sums[i] = suffix_sums[i + 1] + reciprocals[i]
            
        found = []
        
        def dfs_opt(start_idx, current_sum, current_subset):
            if current_sum == target:
                if current_subset:
                    mask = 0
                    for y in current_subset:
                        mask |= (1 << y)
                    found.append(mask)
                return
            if current_sum > target:
                return
            if start_idx >= n:
                return
            # Prune if even adding all remaining smallest reciprocals isn't enough
            if current_sum + suffix_sums[start_idx] < target:
                return
            
            for i in range(start_idx, n):
                y = candidates_sorted[i]
                next_sum = current_sum + reciprocals[i]
                # Since reciprocals are sorted descending, if next_sum > target, 
                # we cannot use this y. But we can use larger y (smaller reciprocal).
                # So we continue the loop.
                dfs_opt(i + 1, next_sum, current_subset + [y])
                
        dfs_opt(0, Fraction(0, 1), [])
        conflicts[x] = found
        
    return conflicts

def _is_valid_fast(S, conflicts):
    """
    S is a set of integers.
    conflicts[x] is a list of bitmasks M such that sum(1/y for y in M) == 1/x.
    """
    if not S:
        return True
        
    mask_S = 0
    for x in S:
        mask_S |= (1 << x)
        
    for x in S:
        for M in conflicts[x]:
            # Check if M is a subset of S
            if (M & mask_S) == M:
                return False
    return True

def construct():
    start = time.time()
    deadline = start + 85 # Safety margin
    
    # Precompute conflicts
    conflicts = _precompute_conflicts()
    
    # Initial solution: Greedy from largest to smallest
    best_set = set()
    for c in range(24, 0, -1):
        trial = best_set | {c}
        if _is_valid_fast(trial, conflicts):
            best_set = trial

    best_size = len(best_set)
    
    # Local Search
    elements = list(range(1, 25))
    
    while time.time() < deadline:
        # Try to improve current best_set
        improved = False
        random.shuffle(elements)
        current_set = set(best_set)
        
        # Pass 1: Try to add elements
        for x in elements:
            if x in current_set:
                continue
            trial = current_set | {x}
            if _is_valid_fast(trial, conflicts):
                current_set = trial
                improved = True
        
        if len(current_set) > best_size:
            best_set = current_set
            best_size = len(current_set)
        
        # If no improvement, try perturbation
        if not improved:
            # Remove 1 or 2 random elements and refill
            if best_set:
                num_removals = random.randint(1, 2)
                for _ in range(num_removals):
                    if current_set:
                        victim = random.choice(list(current_set))
                        current_set.remove(victim)
                
                # Refill greedily in random order
                random.shuffle(elements)
                for x in elements:
                    if x in current_set:
                        continue
                    trial = current_set | {x}
                    if _is_valid_fast(trial, conflicts):
                        current_set = trial
                
                if len(current_set) > best_size:
                    best_set = current_set
                    best_size = len(current_set)
            else:
                break
                
        # Check time
        if time.time() > deadline - 5:
            break

    return sorted(best_set)
