"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from itertools import combinations

def _lcm(a, b):
    return a * b // gcd(a, b)

def _is_valid_triple(a, b, c):
    """Check if {a, b, c} forms a bad triple (equal pairwise LCMs)."""
    return _lcm(a, b) == _lcm(a, c) == _lcm(b, c)

def _conflicts_with_set(x, S):
    """Check if adding x to S creates any bad triple."""
    # A bad triple is {x, a, b} where a, b in S.
    # So we check all pairs (a, b) in S.
    S_list = list(S)
    n = len(S_list)
    for i in range(n):
        a = S_list[i]
        for j in range(i + 1, n):
            b = S_list[j]
            if _is_valid_triple(x, a, b):
                return True
    return False

def _greedy_construct():
    """Fast greedy construction to get a valid starting set."""
    selected = []
    for x in range(1, 129):
        if not _conflicts_with_set(x, selected):
            selected.append(x)
    return set(selected)

def _try_swap(S, x_out, x_in):
    """
    Check if we can remove x_out from S and add x_in to S.
    Returns True if the new set S - {x_out} + {x_in} is valid.
    """
    S_new = S - {x_out}
    if x_in in S_new:
        return False
    # Check if x_in conflicts with any pair in S_new
    # Note: We don't need to check if x_in conflicts with x_out because x_out is removed.
    # We only need to check pairs (a, b) in S_new.
    S_new_list = list(S_new)
    n = len(S_new_list)
    for i in range(n):
        a = S_new_list[i]
        for j in range(i + 1, n):
            b = S_new_list[j]
            if _is_valid_triple(x_in, a, b):
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 60s limit
    
    # 1. Initial Greedy Solution
    best_set = _greedy_construct()
    best_size = len(best_set)
    
    # 2. Local Search: Hill Climbing with Random Restarts
    # We try to improve the size.
    # Since we can't increase size by just swapping (size stays same), 
    # we need a strategy to find a larger set.
    # Actually, simple swaps preserve size. 
    # To increase size, we need to find an element not in S that can be added.
    # But if S is maximal (no element can be added), we must swap.
    # However, swapping doesn't increase size. 
    # Wait, the greedy method might not produce a maximal set? 
    # Greedy usually produces a maximal set (no single element can be added).
    # If it's maximal, we can't just add. 
    # So we must use swaps to change the structure, potentially allowing a larger set to be found in a different basin?
    # No, if S is maximal, |S| is fixed for that specific structure. 
    # To get a LARGER set, we need to find a different maximal set with higher cardinality.
    # Random restarts of greedy are a good way to explore different maximal sets.
    
    # Let's combine Greedy Restarts with Local Search for optimization.
    
    full_set = set(range(1, 129))
    
    while time.time() < deadline:
        # Strategy A: Randomized Greedy Restart
        # Shuffle the order of [1, 128] and run greedy
        candidates = list(range(1, 129))
        random.shuffle(candidates)
        current_set = set()
        for x in candidates:
            if not _conflicts_with_set(x, current_set):
                current_set.add(x)
        
        if len(current_set) > best_size:
            best_set = current_set
            best_size = len(current_set)
            
        # If we found a very large set, we can stop early if we hit a theoretical upper bound?
        # Upper bound is 128. 
        # If best_size is close to 128, we might be done.
        
        # Strategy B: Local Search (Swaps) to escape local maxima of Greedy?
        # Actually, Greedy produces a maximal set. Swaps keep size constant.
        # So swaps don't help increase size directly unless we use them to navigate 
        # to a region where a different greedy run yields better results? 
        # That's indirect.
        
        # Better Strategy: Start from a random valid set and try to ADD elements?
        # If we start with a small random valid set, we can try to add elements.
        # This is essentially the same as Greedy but with random start?
        # No, Greedy is sequential. 
        # Let's try: Start with empty set, pick random elements to add if valid.
        # This is "Randomized Greedy".
        
        # Let's also try a "Repair" strategy:
        # Start with a random subset, remove conflicts, then try to add back?
        # That's complex.
        
        # The most effective simple improvement is usually just many randomized greedy runs.
        # The greedy algorithm is sensitive to order.
        
        # Let's add a simple local search that tries to swap elements to "free up" space?
        # No, if it's maximal, no single element can be added.
        # So we are stuck at the size of that maximal set.
        # Different orders yield different maximal sets with potentially different sizes.
        # So Randomized Greedy is the key.
        
        # To make it faster, we can break early if we find a set of size 100+? 
        # We don't know the max.
        
        # Let's run as many randomized greedy passes as possible.
        
        # Optimization: 
        # If best_size is very high, we can stop.
        # But we don't know the max.
        
        # Let's add a small local search step: 
        # Try to swap an element in current_set with one not in current_set 
        # to see if it allows adding a THIRD element? 
        # That's a 2-swap (remove 1, add 1, then try to add 1 more).
        # This is expensive.
        
        pass # The loop above already does randomized greedy.

    return list(best_set)
# EVOLVE-BLOCK-END
