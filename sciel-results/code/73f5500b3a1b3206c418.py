import numpy as np
import time

N = 18

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin
    
    # 1. Generate Initial Network (Odd-Even Transposition)
    network = [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
    
    # 2. Generate Subset of Inputs for Fast Evaluation
    # 4096 inputs for better accuracy in subset check
    subset_size = 4096
    inputs = np.random.randint(0, 2, size=(subset_size, N), dtype=np.int8)
    
    def is_sorted_subset(net, inp):
        state = inp.copy()
        for i, j in net:
            a = state[:, i]
            b = state[:, j]
            state[:, i] = np.minimum(a, b)
            state[:, j] = np.maximum(a, b)
        return np.all(state[:, :-1] <= state[:, 1:])
    
    candidates = []
    best_count = len(network)
    
    # 3. Greedy Removal Pass
    # Try to remove comparators one by one from the end to the start
    # Repeat until no more can be removed
    while time.time() < deadline - 10.0:
        removed_any = False
        # Iterate backwards to avoid index shifting issues if we just popped
        for idx in range(len(network) - 1, -1, -1):
            if time.time() > deadline - 10.0:
                break
            removed = network.pop(idx)
            if is_sorted_subset(network, inputs):
                removed_any = True
                # Keep it removed
                if len(network) < best_count:
                    best_count = len(network)
                    candidates.append(network.copy())
            else:
                network.insert(idx, removed)
        if not removed_any:
            break
            
    # 4. Random Local Search
    # Try random removals
    while time.time() < deadline - 5.0:
        if len(network) == 0:
            break
        idx = np.random.randint(0, len(network))
        removed = network.pop(idx)
        if is_sorted_subset(network, inputs):
            if len(network) < best_count:
                best_count = len(network)
                candidates.append(network.copy())
        else:
            network.insert(idx, removed)
            
    # 5. Final Verification on Full Set
    # We need to check the candidates against all 2^18 inputs.
    # Generate full inputs
    full_inputs = np.zeros((2**N, N), dtype=np.int8)
    for i in range(N):
        full_inputs[:, i] = (np.arange(2**N) >> (N-1-i)) & 1
        
    # Check candidates in reverse order (most recent/best first)
    for cand in reversed(candidates):
        if time.time() > deadline - 1.0:
            break
        if is_sorted_subset(cand, full_inputs):
            return cand
            
    # If no candidate passed full check, return the original OET network
    # which is guaranteed to be valid.
    return [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
# EVOLVE-BLOCK-END
