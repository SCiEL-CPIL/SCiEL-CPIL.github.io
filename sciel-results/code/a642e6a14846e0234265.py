"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

N = 18

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    # Start with odd-even transposition network
    network = [(i, i + 1) for phase in range(N)
               for i in range(phase % 2, N - 1, 2)]
    
    # Precompute all 2^18 Boolean inputs as a matrix
    # Shape: (262144, 18)
    all_inputs = np.zeros((1 << N, N), dtype=np.int8)
    for bit in range(N):
        all_inputs[:, bit] = ((np.arange(1 << N) >> bit) & 1).astype(np.int8)
    
    def verify_network(net):
        """Verify network sorts all Boolean inputs using NumPy vectorization."""
        state = all_inputs.copy()
        for (i, j) in net:
            a = state[:, i]
            b = state[:, j]
            state[:, i] = np.minimum(a, b)
            state[:, j] = np.maximum(a, b)
        # Check if sorted: each row should be non-decreasing
        diffs = np.diff(state, axis=1)
        return np.all(diffs >= 0)
    
    # Verify initial network
    assert verify_network(network), "Initial network should be valid"
    
    best_network = network[:]
    best_count = len(best_network)
    
    # Phase 1: Greedy removal - try removing each comparator
    changed = True
    while changed and time.time() < deadline - 10.0:
        changed = False
        for i in range(len(best_network)):
            if time.time() > deadline - 10.0:
                break
            candidate = best_network[:i] + best_network[i+1:]
            if verify_network(candidate):
                best_network = candidate
                changed = True
                break  # Restart from beginning since network changed
    
    # Phase 2: Local search - try swapping adjacent comparators
    for iteration in range(20):
        if time.time() > deadline - 5.0:
            break
        improved = False
        for i in range(len(best_network) - 1):
            if time.time() > deadline - 5.0:
                break
            candidate = best_network[:]
            candidate[i], candidate[i+1] = candidate[i+1], candidate[i]
            if verify_network(candidate):
                best_network = candidate
                improved = True
                break
        if not improved:
            break
    
    # Phase 3: Final greedy removal pass
    changed = True
    while changed and time.time() < deadline - 2.0:
        changed = False
        for i in range(len(best_network)):
            if time.time() > deadline - 2.0:
                break
            candidate = best_network[:i] + best_network[i+1:]
            if verify_network(candidate):
                best_network = candidate
                changed = True
                break
    
    # Final verification
    if not verify_network(best_network):
        return network
    
    return best_network
# EVOLVE-BLOCK-END
