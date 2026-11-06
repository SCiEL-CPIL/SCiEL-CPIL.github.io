"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
from fractions import Fraction

def _is_valid_subset(S):
    """
    Check if S is a reciprocal-subsum-free set.
    S is a set of integers.
    Condition: For all x in S, 1/x is not the sum of reciprocals of any subset of S\{x}.
    """
    if not S:
        return True
    
    S_list = list(S)
    for x in S_list:
        target = Fraction(1, x)
        others = [y for y in S_list if y != x]
        
        # We want to check if target can be formed by summing a subset of {1/y for y in others}
        # Use a set of reachable sums, pruned by the target value
        reachable = {Fraction(0, 1)}
        
        # Sort others by reciprocal descending (i.e., y ascending) to potentially hit target faster or prune sooner?
        # Actually, adding small reciprocals (large y) first might keep sums small longer?
        # Let's just iterate.
        for y in others:
            r = Fraction(1, y)
            new_reachable = set()
            for s in reachable:
                ns = s + r
                if ns <= target:
                    new_reachable.add(ns)
            
            reachable.update(new_reachable)
            
            if target in reachable:
                return False
                
    return True

def construct():
    deadline = time.time() + 50  # Safety margin for 60s limit
    best_set = set()
    best_size = 0
    universe = list(range(1, 25))
    
    # Initial greedy attempt with sorted order
    current = set()
    for x in universe:
        trial = current | {x}
        if _is_valid_subset(trial):
            current = trial
    
    if len(current) > best_size:
        best_size = len(current)
        best_set = current
        
    # Randomized Greedy Search
    # The number of trials depends on time remaining.
    while time.time() < deadline:
        random.shuffle(universe)
        current = set()
        for x in universe:
            trial = current | {x}
            if _is_valid_subset(trial):
                current = trial
        
        if len(current) > best_size:
            best_size = len(current)
            best_set = current
            
        # Optional: Local Search / Swap
        # Try to improve current set by swapping one element for another
        # This is complex to implement efficiently within the time limit without a robust heuristic.
        # Randomized greedy is often sufficient for N=24.
        
    return list(best_set)
# EVOLVE-BLOCK-END
