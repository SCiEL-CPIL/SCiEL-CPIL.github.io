import numpy as np
import time
import random

N = 17
M = 1 << N

# Precompute sorted states
# A state is sorted if all 0s are before all 1s.
# If a state has k ones, the ones are in positions N-k to N-1.
# Target state = ((1 << k) - 1) << (N - k)
states_all = np.arange(M, dtype=np.int32)

# Helper to get sorted target
def get_sorted_target(s):
    k = bin(s).count('1')
    if k == 0: return 0
    if k == N: return (1 << N) - 1
    return ((1 << k) - 1) << (N - k)

# Precompute sorted targets
sorted_targets = np.array([get_sorted_target(s) for s in range(M)], dtype=np.int32)

def is_valid(network):
    """
    Applies the network to the array of states and checks if all are sorted.
    Optimized for speed: uses in-place operations and early exit if possible.
    """
    # Allocate new array for current state
    curr = np.empty(M, dtype=np.int32)
    np.copyto(curr, states_all)
    
    for i, j in network:
        bi = (curr >> i) & 1
        bj = (curr >> j) & 1
        # Determine where swap is needed: bi=1, bj=0
        swap_mask = bi > bj
        if np.any(swap_mask):
            curr[swap_mask] ^= (1 << i) | (1 << j)
            
    # Check if all states match their sorted target
    # np.any on a boolean array is fast
    return not np.any(curr != sorted_targets)

def get_initial_network():
    # Odd-Even Transposition Network
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]

def construct():
    start_time = time.time()
    # Safety margin: evaluator kills at 90s, we want to be well within 60s.
    # Let's aim for 50s of computation.
    deadline = start_time + 50.0
    
    best_network = get_initial_network()
    best_size = len(best_network)
    
    all_comparators = [(i, j) for i in range(N) for j in range(i+1, N)]
    
    current_network = best_network[:]
    current_size = best_size
    
    # Phase 1: Greedy Deletion
    # Try to remove comparators one by one.
    # This is effective for reducing size from the seed.
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Iterate through a copy of indices to allow removal
        # We iterate backwards to avoid index shifting issues if we remove
        for idx in range(len(current_network) - 1, -1, -1):
            if time.time() > deadline:
                break
            test_network = current_network[:idx] + current_network[idx+1:]
            if is_valid(test_network):
                current_network = test_network
                current_size -= 1
                improved = True
                if current_size < best_size:
                    best_network = current_network[:]
                    best_size = current_size
    
    # Phase 2: Local Search (Swaps, Insertions, Deletions)
    # If time remains, try to improve further.
    while time.time() < deadline:
        # Random move selection
        move_type = random.random()
        
        if move_type < 0.4 and current_size > 0:
            # Deletion
            idx = random.randint(0, current_size - 1)
            test_network = current_network[:idx] + current_network[idx+1:]
            if is_valid(test_network):
                current_network = test_network
                current_size -= 1
                if current_size < best_size:
                    best_network = current_network[:]
                    best_size = current_size
        elif move_type < 0.7 and current_size > 1:
            # Swap
            i1 = random.randint(0, current_size - 2)
            i2 = random.randint(0, current_size - 2)
            if i1 != i2:
                test_network = current_network[:]
                test_network[i1], test_network[i2] = test_network[i2], test_network[i1]
                if is_valid(test_network):
                    current_network = test_network
                    # Size unchanged
        else:
            # Insertion
            # Insert a random comparator at a random position
            pos = random.randint(0, current_size)
            cmp = random.choice(all_comparators)
            test_network = current_network[:pos] + [cmp] + current_network[pos:]
            
            # Try to delete some comparators from this new network to keep size manageable
            # Greedy deletion on the new network
            # We don't want to spend too much time here, so limit deletions
            # If it's valid as is, we might accept it if it helps future deletions?
            # Usually, we want to delete. So we try to delete one.
            # For simplicity in this phase, just check if the inserted network is valid.
            # If valid, we might keep it if we think it helps structure, but it increases size.
            # Better: Insert, then try to delete a different one.
            # This is complex. Let's stick to: Insert, then if valid, accept as a "move" 
            # but only if we can immediately delete something else?
            # Actually, let's just do: Insert, then Greedy Delete one step.
            
            # To save time, let's just check if the new network is valid.
            # If it is, we accept it as a state change (size +1).
            # Then the next iteration might delete something.
            if is_valid(test_network):
                current_network = test_network
                current_size += 1
                # Note: We don't update best_size here because size increased.
                # But we update current_network to explore this region.
                
    return best_network
