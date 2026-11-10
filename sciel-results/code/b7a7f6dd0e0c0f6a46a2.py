"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _precompute_triple_sums():
    """Precompute all triple sums of distinct pairs."""
    triple_sums = {}
    for x in range(1, 193):
        others = [v for v in range(1, 193) if v != x]
        # For each x, we need to check if x divides sum of any 3 distinct others
        # We'll precompute for each x the set of pairs (a,b) such that a+b+... 
        # Actually, for a given x, we need to check: does there exist 3 distinct others a,b,c
        # such that (a+b+c) % x == 0?
        # This is expensive to precompute fully. Instead, let's think differently.
        pass
    return triple_sums

def _can_add_to_set(S, x):
    """Check if x can be added to set S without violating the constraint.
    
    When adding x to S, we need to verify:
    1. x does not divide the sum of any 3 distinct elements in S
    2. No element a in S divides the sum of any 3 distinct elements in S ∪ {x} that include x
    
    For condition 1: check all triples from S, see if x divides any
    For condition 2: for each a in S, check triples that include x and 2 others from S
    """
    S_list = list(S)
    
    # Condition 1: x must not divide sum of any 3 distinct elements in S
    for a, b, c in combinations(S_list, 3):
        if (a + b + c) % x == 0:
            return False
    
    # Condition 2: For each element a in S, check if a divides sum of any 3 distinct 
    # elements in S ∪ {x} that include x.
    # The triples that include x are: {x, p, q} where p, q are distinct elements from S
    # So we need: for each a in S, for each pair (p, q) from S with p != q,
    # check if (x + p + q) % a == 0
    for a in S_list:
        for p, q in combinations(S_list, 2):
            if (x + p + q) % a == 0:
                return False
    
    return True

def _greedy_construct(order):
    """Greedy construction with given order."""
    S = set()
    for x in order:
        if _can_add_to_set(S, x):
            S.add(x)
    return S

def _local_search(S, time_budget=5.0):
    """Local search: try swaps and additions to improve the set."""
    S = set(S)
    start_time = time.time()
    
    # First, try to add any elements that can be added
    changed = True
    while changed and (time.time() - start_time) < time_budget:
        changed = False
        for x in range(1, 193):
            if x not in S and _can_add_to_set(S, x):
                S.add(x)
                changed = True
    
    # Then, try swaps: remove one element, try to add one or more
    improved = True
    while improved and (time.time() - start_time) < time_budget:
        improved = False
        current_size = len(S)
        # Try removing each element and see if we can add more
        for x in list(S):
            if (time.time() - start_time) > time_budget:
                break
            S_remove = S - {x}
            new_size = len(S_remove)
            # Try to add elements
            for y in range(1, 193):
                if y not in S_remove and y != x and _can_add_to_set(S_remove, y):
                    S_remove.add(y)
            if len(S_remove) > current_size:
                S = S_remove
                improved = True
                break
    
    return S

def construct():
    """Construct a large set of distinct positive integers <= 192 such that
    no selected integer divides the sum of any 3 distinct other selected integers."""
    start_time = time.time()
    deadline = start_time + 85.0  # Leave 5s safety margin
    
    best_set = set()
    best_size = 0
    
    # Strategy 1: Greedy with different orderings
    orders = [
        list(range(192, 0, -1)),  # Descending
        list(range(1, 193)),      # Ascending
    ]
    
    for order in orders:
        if time.time() > deadline:
            break
        S = _greedy_construct(order)
        if len(S) > best_size:
            best_size = len(S)
            best_set = set(S)
    
    # Strategy 2: Random restarts with greedy
    while time.time() < deadline - 10.0:
        order = list(range(1, 193))
        random.shuffle(order)
        S = _greedy_construct(order)
        if len(S) > best_size:
            best_size = len(S)
            best_set = set(S)
    
    # Strategy 3: Local search on the best found so far
    if time.time() < deadline - 2.0:
        remaining = deadline - 2.0 - time.time()
        S = _local_search(best_set, time_budget=remaining)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S
    
    return list(best_set)
# EVOLVE-BLOCK-END
