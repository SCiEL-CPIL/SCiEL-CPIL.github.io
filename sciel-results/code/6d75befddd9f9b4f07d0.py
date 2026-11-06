"""
Optimized sorting network search for 17 wires.
Improves upon the seed by using aggressive pruning with local reordering
and a more effective local search strategy.
"""
import time
import random
import numpy as np

N = 17
SEED_COMPARATORS = 136

# Precompute inputs for validation
# 2^17 = 131,072 inputs, 17 bits each.
_indices = np.arange(1 << N, dtype=np.uint32)
_bits = 1 << np.arange(N, dtype=np.uint32)
_INPUTS = (_indices[:, None] & _bits[None, :]) > 0
_SORTED_INPUTS = np.sort(_INPUTS, axis=1)

# Use a subset for fast filtering in local search
# A larger sample reduces false positives (networks that pass sample but fail full)
_SAMPLE_SIZE = 4096
_SAMPLE_INDICES = np.random.choice(1 << N, size=_SAMPLE_SIZE, replace=False)
_SAMPLE_INPUTS = _INPUTS[_SAMPLE_INDICES]
_SAMPLE_SORTED = _SORTED_INPUTS[_SAMPLE_INDICES]

def is_sorted_network_full(comparators):
    """
    Full validation against all 2^N inputs.
    """
    if len(comparators) > 4096:
        return False
    if not comparators:
        return False

    state = _INPUTS.copy()
    
    for i, j in comparators:
        min_val = np.minimum(state[:, i], state[:, j])
        max_val = np.maximum(state[:, i], state[:, j])
        state[:, i] = min_val
        state[:, j] = max_val
        
    return np.array_equal(state, _SORTED_INPUTS)

def is_sorted_network_sample(comparators):
    """
    Fast validation against a sample of inputs.
    Used as a filter in local search.
    """
    if len(comparators) > 4096:
        return False
    if not comparators:
        return False

    state = _SAMPLE_INPUTS.copy()
    
    for i, j in comparators:
        min_val = np.minimum(state[:, i], state[:, j])
        max_val = np.maximum(state[:, i], state[:, j])
        state[:, i] = min_val
        state[:, j] = max_val
        
    return np.array_equal(state, _SAMPLE_SORTED)

def generate_seed_network():
    """
    Generate Odd-Even Transposition Network.
    """
    network = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            network.append((i, i + 1))
    return network

def construct():
    start_time = time.time()
    # Safety margin: aim to finish within 50s to be safe against 60s/90s limits
    deadline = start_time + 50.0
    
    # 1. Initialize with Seed Network
    current_network = generate_seed_network()
    if not is_sorted_network_full(current_network):
        # Fallback if seed generation is somehow flawed (should not happen)
        return current_network
    
    best_network = current_network.copy()
    best_len = len(current_network)
    
    # 2. Aggressive Pruning Phase
    # We iterate until no more comparators can be removed.
    # To make this more effective, we don't just try removal; 
    # we also try local swaps that might enable removal.
    
    pruning_deadline = deadline - 10.0 # Reserve 10s for final local search
    
    changed = True
    # Limit pruning passes to avoid infinite loops in degenerate cases, though time guard handles it
    max_pruning_passes = 100 
    
    while changed and time.time() < pruning_deadline and max_pruning_passes > 0:
        changed = False
        max_pruning_passes -= 1
        
        # Iterate backwards to handle index shifts
        for idx in range(len(current_network) - 1, -1, -1):
            if time.time() > pruning_deadline:
                break
            
            # Try removing comparator at idx
            candidate = current_network[:idx] + current_network[idx+1:]
            if is_sorted_network_full(candidate):
                current_network = candidate
                best_network = current_network.copy()
                best_len = len(current_network)
                changed = True
                break # Restart the while loop as the network structure changed significantly
        
        # If we didn't find a simple removal in the last pass, 
        # try a few local reorders to "shake" the network and potentially expose redundancies.
        if not changed:
            # Try swapping a few random pairs of comparators that are close to each other
            # This is a heuristic to escape local minima in the pruning landscape
            for _ in range(5):
                if time.time() > pruning_deadline:
                    break
                if len(current_network) < 2:
                    break
                i = random.randint(0, len(current_network) - 2)
                current_network[i], current_network[i+1] = current_network[i+1], current_network[i]
                # Check if this reorder allows a removal
                # We don't fully validate the swap itself for sorting (it's a valid network if original was),
                # but we check if any comparator is now redundant.
                for j in range(len(current_network)):
                    cand = current_network[:j] + current_network[j+1:]
                    if is_sorted_network_full(cand):
                        current_network = cand
                        best_network = current_network.copy()
                        best_len = len(current_network)
                        changed = True
                        break
                if changed:
                    break

    # 3. Local Search Phase
    # Now that we are at a local optimum for removals, we try to find smaller networks
    # by combining removals with replacements or moves.
    
    local_search_deadline = deadline - 2.0
    current = best_network.copy()
    current_len = best_len
    
    # Pre-generate random comparators for replacement moves
    random_comparators = set()
    for _ in range(100):
        i = random.randint(0, N - 2)
        j = random.randint(i + 1, N - 1)
        random_comparators.add((i, j))
    random_comparator_list = list(random_comparators)
    
    temperature = 1.0
    cooling_rate = 0.99
    
    while time.time() < local_search_deadline:
        # Decay temperature
        temperature *= cooling_rate
        
        neighbor = current.copy()
        
        # Choose a move
        r = random.random()
        if r < 0.3 and len(neighbor) > 0:
            # Remove a comparator
            idx = random.randint(0, len(neighbor) - 1)
            neighbor.pop(idx)
        elif r < 0.6 and len(neighbor) > 0:
            # Replace a comparator
            idx = random.randint(0, len(neighbor) - 1)
            new_comp = random_comparator_list[random.randint(0, len(random_comparator_list) - 1)]
            neighbor[idx] = new_comp
        elif r < 0.9 and len(neighbor) > 1:
            # Swap two comparators
            i1 = random.randint(0, len(neighbor) - 1)
            i2 = random.randint(0, len(neighbor) - 1)
            if i1 != i2:
                neighbor[i1], neighbor[i2] = neighbor[i2], neighbor[i1]
            else:
                continue
        else:
            # Move a comparator to a new position
            if len(neighbor) < 2:
                continue
            idx = random.randint(0, len(neighbor) - 1)
            new_idx = random.randint(0, len(neighbor) - 1)
            if idx != new_idx:
                comp = neighbor.pop(idx)
                neighbor.insert(new_idx, comp)
            else:
                continue
        
        # Fast check first
        if not is_sorted_network_sample(neighbor):
            continue
            
        # Full check
        if is_sorted_network_full(neighbor):
            neighbor_len = len(neighbor)
            
            if neighbor_len < current_len:
                current = neighbor
                current_len = neighbor_len
                if current_len < best_len:
                    best_network = current.copy()
                    best_len = current_len
                    # Reset temperature to explore more aggressively after improvement
                    temperature = 1.0
            else:
                # Accept worse or equal solutions with probability based on temperature
                # This helps escape local minima
                if neighbor_len == current_len:
                    if random.random() < 0.5:
                        current = neighbor
                elif random.random() < temperature:
                    current = neighbor

    # Final validation and return
    if is_sorted_network_full(best_network):
        return best_network
    else:
        return generate_seed_network()

# Helper for potential future use or testing
if __name__ == "__main__":
    net = construct()
    print(f"Found network with {len(net)} comparators")
