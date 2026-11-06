import numpy as np
import time

N = 17
MAX_COMPARATORS = 4096

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 60s limit
    
    # Generate all 2^N inputs
    # Shape: (2^N, N)
    # We can generate this efficiently using bit unpacking or recursion.
    # np.arange(2**N) gives 0 to 2^N-1.
    # We need to expand each into N bits.
    # A faster way:
    # inputs = np.zeros((2**N, N), dtype=np.uint8)
    # for i in range(N):
    #     inputs[:, i] = (np.arange(2**N) >> (N - 1 - i)) & 1
    # This is O(N * 2^N) which is 17 * 131072 ~ 2.2M ops. Very fast.
    
    # Actually, let's just generate it once outside if possible, but construct() is called once.
    # Inside is fine.
    
    inputs = np.zeros((2**N, N), dtype=np.uint8)
    for i in range(N):
        inputs[:, i] = (np.arange(2**N) >> (N - 1 - i)) & 1

    def is_valid(network):
        # network is a list of (i, j)
        if not network:
            # Empty network is only valid if N=1, which is not the case.
            # But for N>1, empty network fails unless inputs are already sorted? No.
            # We check if all inputs are sorted through it.
            pass
        
        # Copy inputs
        # To avoid copying the whole matrix every time if we can, but we modify it.
        # We must copy.
        state = inputs.copy()
        
        for i, j in network:
            # state[:, i] and state[:, j]
            # min -> i, max -> j
            min_val = np.minimum(state[:, i], state[:, j])
            max_val = np.maximum(state[:, i], state[:, j])
            state[:, i] = min_val
            state[:, j] = max_val
            
        # Check if sorted
        # sorted means state[:, k] <= state[:, k+1] for all k
        # diff = state[:, 1:] - state[:, :-1]
        # all diff >= 0
        # Since values are 0 or 1, diff is in {-1, 0, 1}
        # We can check if any diff is -1
        # Or simply: np.all(state[:, 1:] >= state[:, :-1])
        
        # Optimization: 
        # If we track that the network is "almost" sorted, we might skip? 
        # No, full check is fast enough.
        
        return np.all(state[:, 1:] >= state[:, :-1])

    # Initial Network: Odd-Even Transposition
    # Phases: 0 to N-1
    # Phase p: comparators (i, i+1) where i % 2 == p % 2
    network = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            network.append((i, i + 1))
    
    # Verify initial network
    if not is_valid(network):
        # Should not happen, but fallback
        return network

    best_network = network[:]
    
    # Greedy Deletion
    # Try to remove each comparator.
    # If successful, remove and restart the loop.
    # To be efficient, we can try to remove in a specific order.
    # Often, removing from the "middle" or "end" first works well.
    # Let's try removing from the end of the list backwards, then forwards.
    
    improved = True
    while improved:
        improved = False
        # Check time
        if time.time() > deadline:
            break
            
        # Try removing each comparator
        # We iterate backwards so that indices remain valid if we remove?
        # No, we break out and restart if we remove one.
        # So we can iterate forwards.
        
        for idx in range(len(network)):
            if time.time() > deadline:
                break
                
            # Try removing network[idx]
            candidate = network[:idx] + network[idx+1:]
            
            # Quick check: if candidate is empty and N>1, it's invalid.
            # But is_valid handles it.
            
            if is_valid(candidate):
                network = candidate
                best_network = network[:]
                improved = True
                break # Restart the for loop with the new network
    
    return best_network
