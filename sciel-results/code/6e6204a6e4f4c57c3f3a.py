"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 18

import numpy as np
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a sorting network on 18 wires using a SAT-inspired heuristic approach.
    We use a layered structure with fixed depth and optimize comparator placement
    using a greedy + local search strategy guided by the 0-1 principle.
    
    The approach:
    1. Start with a known good network (odd-even transposition).
    2. Use local search to remove comparators where possible while maintaining correctness.
    3. Use the 0-1 principle: a network sorts all inputs iff it sorts all 2^N boolean inputs.
    
    For efficiency within 60s, we use a practical heuristic:
    - Start from a dense layered network
    - Iteratively try removing each comparator and check if sorting is still guaranteed
    - Use random restarts and simulated annealing to escape local minima
    """
    import random
    import time
    import math
    
    start_time = time.time()
    timeout = 55  # seconds, leave buffer
    
    def odd_even_transposition():
        comps = []
        for phase in range(N):
            for i in range(phase % 2, N - 1, 2):
                comps.append((i, i + 1))
        return comps

    def is_sorting_network_subset(comparators, subset_masks):
        """Check if the network sorts a subset of boolean inputs."""
        if not comparators:
            return True
            
        comp_list = list(comparators)
        n_comps = len(comp_list)
        
        for mask in subset_masks:
            state = [(mask >> i) & 1 for i in range(N)]
            for (i, j) in comp_list:
                if state[i] > state[j]:
                    state[i], state[j] = state[j], state[i]
            # Check sorted
            # Fast check: 0s then 1s
            seen_one = False
            for k in range(N):
                if state[k] == 1:
                    seen_one = True
                elif seen_one:
                    return False
        return True

    def is_sorting_network_full(comparators):
        """Check if the network sorts all boolean inputs."""
        if not comparators:
            return True
            
        comp_list = list(comparators)
        
        for mask in range(1 << N):
            state = [(mask >> i) & 1 for i in range(N)]
            for (i, j) in comp_list:
                if state[i] > state[j]:
                    state[i], state[j] = state[j], state[i]
            # Check sorted
            seen_one = False
            for k in range(N):
                if state[k] == 1:
                    seen_one = True
                elif seen_one:
                    return False
        return True

    # Generate a strong subset for fast checking
    # Include:
    # - All masks with 1 one
    # - All masks with N-1 ones
    # - All masks with N/2 ones (sample)
    # - Random masks
    # - Specific patterns
    subset_masks = set()
    for i in range(N):
        subset_masks.add(1 << i)
        subset_masks.add((1 << N) - 1 - (1 << i))
    
    # Add some random masks
    random.seed(42)
    for _ in range(2000):
        subset_masks.add(random.randint(0, (1 << N) - 1))
    
    # Add specific hard patterns
    for i in range(N):
        subset_masks.add(1 << i) # Already added
        subset_masks.add((1 << (N//2)) - 1) # First half 1s
        subset_masks.add(((1 << N) - 1) ^ ((1 << (N//2)) - 1)) # Second half 1s
        
    subset_masks = list(subset_masks)
    
    current = odd_even_transposition()
    best = list(current)
    best_len = len(current)
    
    # Simulated Annealing parameters
    T = 1.0
    cooling = 0.999
    min_T = 0.01
    
    iteration = 0
    while time.time() - start_time < timeout:
        iteration += 1
        if T < min_T:
            T = min_T
            
        # Generate a move
        move_type = random.random()
        
        new_current = list(current)
        
        if move_type < 0.5:
            # Remove a comparator
            if len(current) > 0:
                idx = random.randint(0, len(current) - 1)
                new_current.pop(idx)
            else:
                continue
        elif move_type < 0.8:
            # Add a comparator
            i, j = random.sample(range(N), 2)
            if i > j: i, j = j, i
            pos = random.randint(0, len(current))
            new_current.insert(pos, (i, j))
        else:
            # Move a comparator
            if len(current) > 1:
                idx = random.randint(0, len(current) - 1)
                comp = new_current.pop(idx)
                new_pos = random.randint(0, len(new_current))
                new_current.insert(new_pos, comp)
            else:
                continue
                
        # Check subset
        if is_sorting_network_subset(new_current, subset_masks):
            new_len = len(new_current)
            
            if new_len < best_len:
                # Candidate for new best
                if is_sorting_network_full(new_current):
                    best = new_current
                    best_len = new_len
                    current = new_current
                    # Restart temperature? Or just continue
                else:
                    # False positive, revert
                    # Do not update current, keep old
                    pass
            else:
                # Same or worse length
                # Accept with SA probability
                delta = new_len - len(current)
                if delta < 0 or random.random() < math.exp(-delta / T):
                    current = new_current
        else:
            # Subset failed, reject
            pass
            
        T *= cooling
        
    return best
# EVOLVE-BLOCK-END
