import time
import numpy as np
import random

# EVOLVE-BLOCK-START
def _is_dissociated_subset(S, recips=None):
    """
    Checks if a set of integers S is dissociated in the reciprocals.
    i.e., all subset sums of {1/s for s in S} are distinct.
    Uses vectorized numpy operations for speed.
    """
    if not S:
        return True
    
    n = len(S)
    
    if recips is None:
        reciprocals = np.array([1.0 / s for s in S])
    else:
        reciprocals = np.array([recips[s] for s in S])
        
    # Generate all subset sums using numpy
    sums = np.zeros(1, dtype=np.float64)
    for r in reciprocals:
        sums = np.concatenate([sums, sums + r])
        
    # Check for duplicates using sorting
    sorted_sums = np.sort(sums)
    # Check if any consecutive elements are equal
    if np.any(np.diff(sorted_sums) == 0):
        return False
        
    return True

def _greedy_build(order, recips):
    """Build a dissociated set greedily in the given order."""
    current_set = []
    used = set()
    
    for x in order:
        if x in used:
            continue
        new_set = current_set + [x]
        if _is_dissociated_subset(new_set, recips):
            current_set = new_set
            used.add(x)
            
    return current_set

def _local_search(current_set, recips, universe, deadline):
    """
    Perform local search to increase the size of the set.
    Moves:
    1. Add a random element.
    2. Swap a random element with a random new element (maintains size, explores landscape).
    3. Remove one element and add two new elements (increases size by 1).
    """
    current_set = list(current_set)
    current_set_set = set(current_set)
    best_set = list(current_set)
    best_size = len(current_set)
    
    while time.time() < deadline:
        # Try to Add
        remaining = [x for x in universe if x not in current_set_set]
        if not remaining:
            break
            
        # 1. Try to add a random element
        x = random.choice(remaining)
        new_set = current_set + [x]
        if _is_dissociated_subset(new_set, recips):
            current_set = new_set
            current_set_set.add(x)
            if len(current_set) > best_size:
                best_set = list(current_set)
                best_size = len(current_set)
            # If we found a larger set, we might want to restart or continue
            # Continuing is fine, we just added one.
            
        # 2. Try to Swap (maintain size, change composition)
        # This helps escape local minima where no single add works, but a different composition might allow adds later.
        if len(current_set) > 0:
            x_in = random.choice(remaining)
            x_out = random.choice(current_set)
            new_set = [x for x in current_set if x != x_out] + [x_in]
            if _is_dissociated_subset(new_set, recips):
                # Accept swap
                current_set = new_set
                current_set_set.remove(x_out)
                current_set_set.add(x_in)
                
        # 3. Try 2-for-1 (increase size by 1)
        # Remove one, add two.
        if len(current_set) > 0:
            x_out = random.choice(current_set)
            # Choose two distinct elements from remaining
            # To avoid bias, pick two random distinct from remaining
            if len(remaining) >= 2:
                x_in1, x_in2 = random.sample(remaining, 2)
                new_set = [x for x in current_set if x != x_out] + [x_in1, x_in2]
                if _is_dissociated_subset(new_set, recips):
                    current_set = new_set
                    current_set_set.remove(x_out)
                    current_set_set.add(x_in1)
                    current_set_set.add(x_in2)
                    if len(current_set) > best_size:
                        best_set = list(current_set)
                        best_size = len(current_set)

    return best_set

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 85s safety margin for 90s limit
    
    # Precompute reciprocals
    recips = {i: 1.0/i for i in range(1, 23)}
    universe = list(range(1, 23))
    
    # Known good seed (size 9)
    best_set = [1, 2, 3, 5, 7, 11, 13, 17, 19]
    best_size = len(best_set)
    
    # Initialize random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Strategy 1: Greedy with various orders
    # Fast and finds local optima quickly.
    
    orders = [
        list(range(1, 23)),
        list(range(22, 0, -1)),
    ]
    
    # Add some random permutations
    max_greedy_randoms = 200 # Increased from 100
    
    for _ in range(max_greedy_randoms):
        if time.time() > deadline - 30:  # Leave 30s for local search
            break
        cand_order = list(range(1, 23))
        random.shuffle(cand_order)
        orders.append(cand_order)
        
    for order in orders:
        if time.time() > deadline - 30:
            break
            
        current_set = []
        used = set()
        
        for x in order:
            if x in used:
                continue
            new_set = current_set + [x]
            if _is_dissociated_subset(new_set, recips):
                current_set = new_set
                used.add(x)
                
        if len(current_set) > best_size:
            best_set = current_set
            best_size = len(current_set)
            
        # Early exit if we reached a very high number
        if best_size >= 15:
            break

    # Strategy 2: Local Search to improve the best set found
    # Use the remaining time to try to grow the best_set further.
    if time.time() < deadline - 5:
        # Run local search on the current best set
        improved_set = _local_search(best_set, recips, universe, deadline - 5)
        if len(improved_set) > best_size:
            best_set = improved_set
            best_size = len(best_set)
            
    # Fallback
    if not best_set:
        return [1, 2, 3, 5, 7, 11, 13, 17, 19]
        
    return best_set
# EVOLVE-BLOCK-END
