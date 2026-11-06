import numpy as np
import time
import random

N = 17
NUM_INPUTS = 1 << N

def _is_valid(net):
    """
    Checks if the network sorts all 2^N binary inputs.
    Uses NumPy for vectorized simulation.
    """
    # Initialize all 2^N inputs
    # Using uint16 is sufficient for 17 bits
    inputs = np.arange(NUM_INPUTS, dtype=np.uint16)
    
    for i, j in net:
        # Extract bits i and j for all inputs
        bit_i = (inputs >> i) & 1
        bit_j = (inputs >> j) & 1
        
        # Compare-exchange: min to i, max to j
        # min(a, b) = a & b for bits
        # max(a, b) = a | b for bits
        new_bit_i = bit_i & bit_j
        new_bit_j = bit_i | bit_j
        
        # Update inputs
        # Clear bits i and j
        mask_i = ~(1 << i)
        mask_j = ~(1 << j)
        
        inputs = (inputs & mask_i) | (new_bit_i << i)
        inputs = (inputs & mask_j) | (new_bit_j << j)
        
    # Check if all outputs are sorted (non-decreasing)
    # Unpack bits into a matrix of shape (NUM_INPUTS, N)
    # This is the most expensive part, but vectorized.
    # mat[k, m] is bit m of input k
    mat = np.stack([(inputs >> i) & 1 for i in range(N)], axis=1)
    
    # Check if each row is non-decreasing
    # mat[:, :-1] <= mat[:, 1:]
    return np.all(mat[:, :-1] <= mat[:, 1:])

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Keep 5s safety margin
    
    # 1. Generate Seed: Odd-Even Transposition Network
    # This is a valid sorting network for N wires.
    # Count: N * (N-1) / 2 * 2 / 2? No.
    # Phases 0 to N-1.
    # Phase p has floor((N-1)/2) or similar comparators.
    # Total is roughly N^2/2. For 17, it's 136.
    seed_net = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            seed_net.append((i, i + 1))
            
    current_net = seed_net
    
    # 2. Greedy Deletion Pass
    # Attempt to remove each comparator if the network remains valid.
    # This quickly reduces the size from the over-dense seed.
    i = 0
    while i < len(current_net) and time.time() < deadline:
        # Try removing current_net[i]
        if len(current_net) > 1:
            test_net = current_net[:i] + current_net[i+1:]
            if _is_valid(test_net):
                current_net = test_net
                # Do not increment i, check the new element at i
            else:
                i += 1
        else:
            break
            
    # 3. Local Search
    random.seed(42)
    
    while time.time() < deadline:
        # Choose a random move
        # Options: Delete, Move, Swap
        # We bias towards Delete if possible, but Move/Swap help escape local minima.
        r = random.random()
        
        if r < 0.4 and len(current_net) > 1:
            # Delete
            idx = random.randint(0, len(current_net) - 1)
            new_net = current_net[:idx] + current_net[idx+1:]
        elif r < 0.8 and len(current_net) > 1:
            # Move
            idx = random.randint(0, len(current_net) - 1)
            comp = current_net[idx]
            new_net = current_net[:idx] + current_net[idx+1:]
            new_idx = random.randint(0, len(new_net))
            new_net.insert(new_idx, comp)
        else:
            # Swap / Change
            if not current_net:
                continue
            idx = random.randint(0, len(current_net) - 1)
            i, j = current_net[idx]
            # Generate a new random comparator
            i_new = random.randint(0, N - 1)
            j_new = random.randint(0, N - 1)
            if i_new == j_new:
                continue
            if i_new > j_new:
                i_new, j_new = j_new, i_new
            
            # Avoid trivial changes
            if (i_new, j_new) == (i, j):
                continue
                
            new_net = current_net[:]
            new_net[idx] = (i_new, j_new)
            
        if _is_valid(new_net):
            current_net = new_net

    return current_net
