"""Combined program for sorting networks on 17 wires.
Uses pre-computed global inputs for fast validation and a hybrid pruning/local search strategy.
"""
import time
import random
import numpy as np

N = 17
SEED_COMPARATORS = 136

# Precompute the input matrix and sorted reference once.
# Shape: (2^N, N)
_indices = np.arange(1 << N, dtype=np.uint32)
_bits = 1 << np.arange(N, dtype=np.uint32)
_INPUTS = (_indices[:, None] & _bits[None, :]) > 0
_SORTED_INPUTS = np.sort(_INPUTS, axis=1)

def is_sorted_network(comparators):
    """
    Check if the network sorts all 2^17 inputs using the zero-one principle.
    Uses pre-computed _INPUTS and _SORTED_INPUTS for speed.
    """
    if len(comparators) > 4096:
        return False
    
    # If no comparators, it's only valid if inputs are already sorted (false for N>1)
    if not comparators:
        return False

    # Copy the input matrix to avoid modifying the global constant
    state = _INPUTS.copy()
    
    for i, j in comparators:
        min_val = np.minimum(state[:, i], state[:, j])
        max_val = np.maximum(state[:, i], state[:, j])
        state[:, i] = min_val
        state[:, j] = max_val
        
    return np.array_equal(state, _SORTED_INPUTS)

def construct():
    start_time = time.time()
    # Safety margin: aim to finish well before 60s
    deadline = start_time + 55.0
    
    # 1. Generate Seed Network (Odd-Even Transposition)
    # Standard OET for N=17: 17 phases, ~8-9 comparators per phase -> ~136 total
    seed_network = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            seed_network.append((i, i + 1))
            
    # Verify seed (should be true)
    if not is_sorted_network(seed_network):
        # Fallback: return seed anyway, though it shouldn't happen
        return seed_network

    best_network = seed_network
    best_len = len(seed_network)
    current = list(seed_network)
    
    # 2. Aggressive Pruning Phase
    # Try to remove any comparator that is redundant.
    # We do multiple passes because removing one can make another redundant.
    # We iterate backwards to avoid index shifting issues during the pass, 
    # but since we break and restart on success, we can also just do a full scan.
    
    pruning_deadline = deadline - 15.0 # Reserve time for local search
    
    changed = True
    while changed and time.time() < pruning_deadline:
        changed = False
        # Iterate from end to start to handle deletions without index issues in a single pass
        # However, if we delete, we break and restart the while loop to catch new redundancies.
        for idx in range(len(current) - 1, -1, -1):
            if time.time() > pruning_deadline:
                break
            
            candidate = current.copy()
            candidate.pop(idx)
            
            if is_sorted_network(candidate):
                current = candidate
                best_network = current.copy()
                best_len = len(current)
                changed = True
                # Break the inner loop to restart the outer while loop
                # This ensures we check the new list from the beginning (or end) again
                break
        
    # 3. Local Search Phase
    # Try to improve further by swapping, moving, or replacing comparators.
    # We focus on small perturbations to the current best.
    
    local_search_deadline = deadline - 2.0
    current = best_network.copy()
    current_len = best_len
    
    # To speed up random generation, pre-generate a pool of random comparators?
    # Or just generate on the fly. Generating on the fly is fine.
    
    while time.time() < local_search_deadline:
        # Randomly select a move type
        move_type = random.randint(0, 2)
        neighbor = current.copy()
        
        if move_type == 0 and neighbor:
            # Strategy: Remove a random comparator
            idx = random.randint(0, len(neighbor) - 1)
            neighbor.pop(idx)
            
        elif move_type == 1 and len(neighbor) >= 2:
            # Strategy: Swap two comparators
            i1 = random.randint(0, len(neighbor) - 1)
            i2 = random.randint(0, len(neighbor) - 1)
            if i1 != i2:
                neighbor[i1], neighbor[i2] = neighbor[i2], neighbor[i1]
            else:
                continue # Skip if same index
                
        else:
            # Strategy: Replace a random comparator with a random new one
            if not neighbor:
                continue
            idx = random.randint(0, len(neighbor) - 1)
            i = random.randint(0, N - 2)
            j = random.randint(i + 1, N - 1)
            neighbor[idx] = (i, j)
            
        # Check validity
        if is_sorted_network(neighbor):
            neighbor_len = len(neighbor)
            if neighbor_len < current_len:
                current = neighbor
                current_len = neighbor_len
                if current_len < best_len:
                    best_network = current.copy()
                    best_len = current_len
            # If neighbor_len == current_len, we could accept it to explore the space, 
            # but since we only care about minimizing count, we stick to strict improvement 
            # unless we are stuck. For now, strict improvement is safer to avoid wandering.
            # However, accepting equal-length valid networks can help escape local minima 
            # in the "structure" space, which might lead to smaller networks later.
            # Let's accept equal length with some probability to allow structural changes.
            elif neighbor_len == current_len:
                if random.random() < 0.5:
                    current = neighbor

    # Final Check: Ensure we return a valid network
    if is_sorted_network(best_network):
        return best_network
    else:
        # Fallback to seed if something went wrong
        return seed_network
