import time
import random
from fractions import Fraction

DEADLINE = 85  # seconds, safety margin before 90s wall-clock kill

def _reciprocal_subsum_free(S):
    """
    Check if set S is reciprocal-subsum-free.
    For every element k in S, 1/k must NOT equal the sum of reciprocals
    of a non-empty subset of S \\ {k}.
    
    S is a list/set of distinct positive integers.
    """
    S = list(set(S))
    n = len(S)
    if n <= 1:
        return True
    
    # Precompute reciprocals
    recips = [Fraction(1, x) for x in S]
    
    for i, target in enumerate(recips):
        # Check all non-empty subsets of S \ {i}
        # Use meet-in-the-middle for efficiency
        others = [recips[j] for j in range(n) if j != i]
        m = len(others)
        
        if m == 0:
            continue
        
        # Meet in the middle
        half = m // 2
        left = others[:half]
        right = others[half:]
        
        # Generate all subset sums for left
        left_sums = set()
        left_sums.add(Fraction(0, 1))
        for val in left:
            new_sums = set()
            for s in left_sums:
                ns = s + val
                if ns <= target:  # prune: sums exceeding target are useless
                    new_sums.add(ns)
            left_sums.update(new_sums)
        
        # Generate all subset sums for right
        right_sums = set()
        right_sums.add(Fraction(0, 1))
        for val in right:
            new_sums = set()
            for s in right_sums:
                ns = s + val
                if ns <= target:
                    new_sums.add(ns)
            right_sums.update(new_sums)
        
        # Check if any combination of left sum + right sum equals target
        # and at least one side is non-zero (to exclude empty subset)
        for ls in left_sums:
            need = target - ls
            if need in right_sums:
                # Ensure the subset is non-empty
                if ls != Fraction(0, 1) or need != Fraction(0, 1):
                    return False
    
    return True

def _greedy_descending(start_set=None):
    """
    Greedy: scan from 24 down to 1, include k if adding it keeps the set free.
    start_set: optional initial set of already-included elements.
    """
    if start_set is None:
        current = set()
    else:
        current = set(start_set)
    
    for k in range(24, 0, -1):
        if k in current:
            continue
        candidate = current | {k}
        if _reciprocal_subsum_free(candidate):
            current.add(k)
    
    return current

def _randomized_greedy(order):
    """
    Greedy with a given order of consideration.
    Try adding each element in the given order if it maintains freeness.
    """
    current = set()
    for k in order:
        candidate = current | {k}
        if _reciprocal_subsum_free(candidate):
            current.add(k)
    return current

def _local_search(S, max_steps=500, time_limit=None):
    """
    Try to improve S by:
    1. Removing an element and trying to add a different one.
    2. Swapping elements.
    Returns improved set or S if no improvement.
    """
    S = set(S)
    best = set(S)
    best_size = len(best)
    
    start_time = time.time()
    
    for _ in range(max_steps):
        if time_limit is not None and (time.time() - start_time) > time_limit:
            break
        
        improved = False
        S_list = list(S)
        random.shuffle(S_list)
        
        for remove_elem in S_list:
            if (time.time() - start_time) > (time_limit or 1e9):
                break
            
            S_without = S - {remove_elem}
            
            # Try adding each element not in S_without
            for add_elem in range(1, 25):
                if add_elem in S_without:
                    continue
                
                candidate = S_without | {add_elem}
                if len(candidate) <= best_size:
                    continue
                
                if _reciprocal_subsum_free(candidate):
                    S = candidate
                    best = set(candidate)
                    best_size = len(best)
                    improved = True
                    break
            
            if improved:
                break
        
        if not improved:
            break
    
    return best

def construct():
    start_time = time.time()
    
    # Phase 1: Descending greedy (deterministic)
    best = _greedy_descending()
    
    # Phase 2: Local search on the greedy result
    time_remaining = DEADLINE - (time.time() - start_time)
    if time_remaining > 5:
        best = _local_search(best, max_steps=200, time_limit=min(15, time_remaining - 3))
    
    # Phase 3: Randomized restarts with different orders
    time_remaining = DEADLINE - (time.time() - start_time)
    random.seed(42)
    
    while time_remaining > 10:
        # Try different random orderings
        order = list(range(1, 25))
        random.shuffle(order)
        
        candidate = _randomized_greedy(order)
        
        if len(candidate) > len(best):
            best = candidate
        
        # Local search on this candidate
        time_remaining = DEADLINE - (time.time() - start_time)
        if time_remaining > 15 and len(candidate) >= len(best) - 1:
            improved = _local_search(candidate, max_steps=100, time_limit=min(10, time_remaining - 5))
            if len(improved) > len(best):
                best = improved
        
        time_remaining = DEADLINE - (time.time() - start_time)
        if time_remaining <= 10:
            break
        
        # Try biasing toward smaller numbers (harder to include but may allow larger sets)
        order = list(range(1, 25))
        random.shuffle(order)
        # Bias: put smaller numbers earlier
        order.sort(key=lambda x: (random.random(), x))
        
        candidate = _randomized_greedy(order)
        if len(candidate) > len(best):
            best = candidate
        
        time_remaining = DEADLINE - (time.time() - start_time)
        if time_remaining > 15:
            improved = _local_search(candidate, max_steps=100, time_limit=min(10, time_remaining - 5))
            if len(improved) > len(best):
                best = improved
    
    # Final check
    if not best:
        best = {13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24}
    
    return sorted(best)
