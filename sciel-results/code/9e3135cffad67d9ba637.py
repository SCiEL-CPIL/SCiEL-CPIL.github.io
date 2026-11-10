import random
import time
from itertools import combinations

def construct():
    """
    Finds a set of distinct positive integers <= 192, max size 48,
    such that no element divides the sum of any 3 distinct other elements.
    Uses a greedy initialization followed by local search.
    """
    N = 192
    MAX_SIZE = 48
    
    # Helper to check if adding x to set S is valid
    def is_valid_addition(S, x):
        # S is a list of integers
        # 1. Check if x divides sum of any 3 elements in S
        if len(S) >= 3:
            for t in combinations(S, 3):
                if sum(t) % x == 0:
                    return False
        
        # 2. Check if any a in S divides sum of x and any 2 other elements in S
        # For each a in S, we need to check pairs (b, c) from S \ {a}
        # such that (b + c + x) % a == 0
        if len(S) >= 2:
            for a in S:
                # We need pairs from S excluding a
                # To speed up, we can precompute sums of pairs? 
                # Here N is small (~48), so direct check is fine.
                # But we must ensure b, c are distinct and != a.
                # Since S is a set of distinct ints, b != c is guaranteed by combinations.
                # We just need to exclude a from the pool.
                others = [v for v in S if v != a]
                if len(others) < 2:
                    continue
                for b, c in combinations(others, 2):
                    if (b + c + x) % a == 0:
                        return False
        return True

    # Helper to check if a set S is fully valid (for initialization or swaps)
    # Note: is_valid_addition is for adding one element to an existing valid set.
    # For swaps, we remove u and add v. The new set must be valid.
    # We can check validity of the new set directly or use incremental checks.
    # Given the small size, a full check of the new set is safer and simpler to code correctly.
    def is_set_valid(S):
        S_list = list(S)
        for a in S_list:
            others = [v for v in S_list if v != a]
            if len(others) >= 3:
                for t in combinations(others, 3):
                    if sum(t) % a == 0:
                        return False
        return True

    # 1. Greedy Initialization
    # Start with empty set, add largest numbers first
    current_set = []
    for x in range(N, 0, -1):
        if is_valid_addition(current_set, x):
            current_set.append(x)
        if len(current_set) >= MAX_SIZE:
            break
            
    best_set = list(current_set)
    best_size = len(best_set)
    
    # 2. Local Search
    # We try to improve the size or find a better configuration.
    # Since the objective is just cardinality, we primarily want to increase size.
    # If we can't increase size, we might swap to enable future increases.
    
    all_numbers = set(range(1, N + 1))
    start_time = time.time()
    time_limit = 58.0  # Leave 2s margin
    
    # Randomize the order of candidates for addition attempts
    candidates = list(all_numbers - set(current_set))
    
    while time.time() - start_time < time_limit:
        # Try to add a new element
        if len(current_set) < MAX_SIZE and candidates:
            random.shuffle(candidates)
            added = False
            for x in candidates:
                if is_valid_addition(current_set, x):
                    current_set.append(x)
                    candidates.remove(x)
                    added = True
                    if len(current_set) > best_size:
                        best_size = len(current_set)
                        best_set = list(current_set)
                    break  # Restart search with new state
            
        # Try swaps if we are stuck or to explore
        if len(current_set) > 0 and candidates:
            # Pick a random element to remove
            u = random.choice(current_set)
            # Try to find a replacement
            random.shuffle(candidates)
            swapped = False
            for v in candidates:
                # Test set: current_set - {u} + {v}
                new_set = [x for x in current_set if x != u]
                if is_valid_addition(new_set, v):
                    # Perform swap
                    current_set = new_set + [v]
                    candidates.remove(v)
                    candidates.append(u)
                    swapped = True
                    # Note: size doesn't change, but configuration does.
                    # We don't update best_set here unless size increases, 
                    # but a swap might allow an addition later.
                    break
            
    return best_set

# EVOLVE-BLOCK-END
