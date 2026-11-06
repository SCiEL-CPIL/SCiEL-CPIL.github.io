import numpy as np
import time

N = 17
MAX_COMPARATORS = 4096

def construct():
    start_time = time.time()
    # Safety margin: evaluator kills at 90s, we want to return well before.
    # Use 80s deadline to be safe.
    deadline = start_time + 80.0

    # Precompute all 2^N boolean inputs
    # Shape: (2^N, N)
    # We generate this once.
    inputs = np.zeros((2**N, N), dtype=np.uint8)
    for i in range(N):
        inputs[:, i] = (np.arange(2**N) >> (N - 1 - i)) & 1

    def is_valid(network):
        """
        Check if the network sorts all inputs.
        network: list of (i, j) pairs
        Returns: True if valid, False otherwise.
        """
        if not network:
            return N == 1
        
        state = inputs.copy()
        
        for i, j in network:
            # min to i, max to j
            # Since values are 0/1, we can use bitwise ops or np.minimum/maximum
            # np.minimum/maximum is vectorized and fast.
            min_val = np.minimum(state[:, i], state[:, j])
            max_val = np.maximum(state[:, i], state[:, j])
            state[:, i] = min_val
            state[:, j] = max_val
            
        # Check if sorted: state[:, k] <= state[:, k+1] for all k
        # Equivalent to: no inversions.
        # Since binary, sorted means all 0s then all 1s.
        # We can check if state[:, 1:] >= state[:, :-1] is all True.
        return np.all(state[:, 1:] >= state[:, :-1])

    # Generate initial Odd-Even Transposition Network (OETN)
    # This is a valid sorting network for N wires.
    # Phases: 0 to N-1
    # Phase p: comparators (i, i+1) where i % 2 == p % 2
    network = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            network.append((i, i + 1))
    
    # Verify initial network (should always be true for OETN)
    if not is_valid(network):
        # Fallback: return a known valid network or empty? 
        # OETN is always valid for N>=1.
        # If this fails, something is wrong with implementation.
        pass

    best_network = network[:]
    best_size = len(network)
    
    # Optimization Strategy:
    # 1. Systematic Deletion: Try to remove each comparator. If the network remains valid, keep it removed.
    #    Repeat passes until no more comparators can be removed.
    # 2. This is a form of local search (greedy).
    #    To speed up, we can try to be smart about the order or use parallel checks?
    #    With 131k inputs and ~100 comparators, a single check is ~131k * 100 * (few ops) ~ 10^7 ops.
    #    In Python/NumPy, this is fast (< 0.1s per check).
    #    We can afford ~1000 checks in 80s? 1000 * 0.1s = 100s. Maybe less.
    #    Let's optimize the check.
    
    # Optimization: Instead of copying the full inputs matrix (131k x 17) every time,
    # we can track the state more efficiently? 
    # Actually, copying 131k x 17 uint8 is ~2.2MB. memcpy is very fast.
    # The bottleneck is the loop over comparators.
    
    # Let's try a faster validity check by using a different representation or early exit?
    # Early exit: If we find an unsorted input, we can stop.
    # But we need to check ALL inputs.
    
    # Alternative: Use a "fingerprint" or subset of inputs?
    # No, correctness requires ALL inputs.
    
    # Let's stick to the full check but optimize the loop.
    
    def is_valid_fast(network):
        if not network:
            return N == 1
        state = inputs.copy()
        for i, j in network:
            # Vectorized min/max
            # We can potentially combine this? No, sequential dependency.
            min_val = np.minimum(state[:, i], state[:, j])
            max_val = np.maximum(state[:, i], state[:, j])
            state[:, i] = min_val
            state[:, j] = max_val
        
        # Check sorted
        # diff = state[:, 1:] - state[:, :-1]
        # return np.all(diff >= 0)
        # For binary, sorted means no '1' followed by '0'.
        # i.e., ~state[:, :-1] & state[:, 1:] == 0? No.
        # state[k] > state[k+1] is the violation.
        # 1 > 0 is violation.
        # So we want to ensure no (1, 0) adjacent pair.
        # mask = (state[:, :-1] == 1) & (state[:, 1:] == 0)
        # return not np.any(mask)
        # This might be faster than subtraction/comparison?
        # Let's use the comparison as it's clear.
        return np.all(state[:, 1:] >= state[:, :-1])

    # Deletion Loop
    improved = True
    while improved:
        improved = False
        if time.time() > deadline - 5.0:  # Leave 5s buffer
            break
            
        # Try to remove each comparator
        # Iterate through indices
        for idx in range(len(network)):
            if time.time() > deadline - 2.0:
                break
            
            # Create candidate network without idx
            # To avoid list slicing overhead in tight loop, we can just skip idx in a loop?
            # But is_valid takes a list.
            # Slicing is O(N). N is small (~100).
            candidate = network[:idx] + network[idx+1:]
            
            if is_valid_fast(candidate):
                network = candidate
                best_network = network[:]
                best_size = len(network)
                improved = True
                # Break to restart the pass with the new shorter list
                break
    
    return best_network
