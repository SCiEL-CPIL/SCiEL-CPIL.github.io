"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import numpy as np
import time

N = 16
NUM_INPUTS = 1 << N

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a size-efficient sorting network for 16 wires.
    Uses a NumPy-vectorized verification to rapidly check correctness of candidate networks.
    This allows for an extensive local search (deletion, swap, move) to minimize comparator count.
    """
    start_time = time.time()
    time_limit = 55.0

    # Pre-compute all inputs as a bit matrix (65536 x 16)
    # inputs[i, k] is the k-th bit of i
    inputs = np.zeros((NUM_INPUTS, N), dtype=np.uint8)
    for k in range(N):
        inputs[:, k] = (np.arange(NUM_INPUTS, dtype=np.uint32) >> k) & 1

    def verify(comparators):
        # Simulate the network on all inputs in parallel
        state = inputs.copy()
        for i, j in comparators:
            col_i = state[:, i]
            col_j = state[:, j]
            new_i = np.minimum(col_i, col_j)
            new_j = np.maximum(col_i, col_j)
            state[:, i] = new_i
            state[:, j] = new_j
        
        # Check if the output is sorted for all inputs.
        # Sorted binary vector: 0...01...1
        # This implies state[:, k] <= state[:, k+1] for all k.
        # Equivalently, no 1 is immediately followed by a 0.
        # We check if any row has a '1' at k and '0' at k+1.
        # diffs = state[:, 1:] - state[:, :-1]
        # if any diff is -1, it's invalid.
        # np.all(diffs >= 0) is efficient.
        return np.all(state[:, 1:] >= state[:, :-1])

    # Start with Odd-Even Transposition Network (120 comparators)
    network = [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
    
    # Verify initial network (should be True)
    if not verify(network):
        # Fallback: Return the initial network if verification fails (should not happen)
        return network

    no_improvement = 0
    max_no_improvement = 100
    
    while no_improvement < max_no_improvement:
        if time.time() - start_time > time_limit:
            break
            
        improved = False
        
        # 1. Try deletions
        for _ in range(50):
            if time.time() - start_time > time_limit:
                break
            if len(network) == 0:
                break
            idx = random.randint(0, len(network) - 1)
            candidate = network[:idx] + network[idx+1:]
            if verify(candidate):
                network = candidate
                improved = True
                no_improvement = 0
                break
        
        if improved:
            continue
            
        # 2. Try swaps of adjacent comparators
        if len(network) > 1:
            for _ in range(50):
                if time.time() - start_time > time_limit:
                    break
                i = random.randint(0, len(network) - 2)
                candidate = network[:]
                candidate[i], candidate[i+1] = candidate[i+1], candidate[i]
                if verify(candidate):
                    network = candidate
                    improved = True
                    no_improvement = 0
                    break
        
        if improved:
            continue
            
        # 3. Try moving a comparator to a different position
        if len(network) > 2:
            for _ in range(50):
                if time.time() - start_time > time_limit:
                    break
                src_idx = random.randint(0, len(network) - 1)
                dst_idx = random.randint(0, len(network) - 1)
                if src_idx == dst_idx or abs(src_idx - dst_idx) == 1:
                    continue
                candidate = network[:]
                comp = candidate.pop(src_idx)
                candidate.insert(dst_idx, comp)
                if verify(candidate):
                    network = candidate
                    improved = True
                    no_improvement = 0
                    break
                    
        if improved:
            continue
            
        no_improvement += 1
        
    return network
# EVOLVE-BLOCK-END
