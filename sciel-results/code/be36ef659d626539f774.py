# EVOLVE-BLOCK-START
import random
import time
from itertools import combinations

def _is_sidon(A):
    sums = set()
    for i in range(len(A)):
        for j in range(i, len(A)):
            s = A[i] + A[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _count_sidon_quadruples(S):
    count = 0
    for quad in combinations(S, 4):
        if _is_sidon(quad):
            count += 1
    return count

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Initial candidate
    seed_set = [0, 1, 2, 4, 5, 7, 8, 10, 11, 13, 14, 16]
    best_set = seed_set[:]
    best_cost = _count_sidon_quadruples(best_set)
    
    if best_cost == 0:
        return best_set

    current_set = best_set[:]
    current_cost = best_cost
    
    # Local Search
    # We will perform random swaps and check cost
    # To speed up, we can just do many iterations
    
    while time.time() < deadline:
        # Pick a random index to change
        i = random.randint(0, 11)
        old_val = current_set[i]
        
        # Pick a new value
        # To ensure distinctness, we can pick from a pool or retry
        while True:
            new_val = random.randint(0, 128)
            if new_val not in current_set:
                break
        
        current_set[i] = new_val
        new_cost = _count_sidon_quadruples(current_set)
        
        if new_cost < current_cost:
            current_cost = new_cost
            if new_cost < best_cost:
                best_cost = new_cost
                best_set = current_set[:]
                if best_cost == 0:
                    break
        else:
            # Revert
            current_set[i] = old_val

    return best_set
# EVOLVE-BLOCK-END
