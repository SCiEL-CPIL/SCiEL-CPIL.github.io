import numpy as np
import time

N = 18

def generate_odd_even_transposition(n):
    """
    Generates the Odd-Even Transposition sorting network.
    """
    net = []
    for phase in range(n):
        for i in range(phase % 2, n - 1, 2):
            net.append((i, i + 1))
    return net

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s limit, leaving 10s margin for safety
    
    # 1. Generate initial network (Odd-Even Transposition is correct and simple)
    network = generate_odd_even_transposition(N)
    
    # 2. Prepare test vectors
    test_vectors = np.zeros((1 << N, N), dtype=np.uint8)
    for i in range(N):
        period = 1 << (N - i)
        half = period // 2
        pattern = np.array([0]*half + [1]*half, dtype=np.uint8)
        reps = (1 << N) // period
        test_vectors[:, i] = np.tile(pattern, reps)
        
    # 3. Optimization Loop
    def check_network(net):
        if not net:
            return False # Empty network is invalid for N>1
        current = test_vectors.copy()
        for i, j in net:
            min_val = np.minimum(current[:, i], current[:, j])
            max_val = np.maximum(current[:, i], current[:, j])
            current[:, i] = min_val
            current[:, j] = max_val
        diffs = np.diff(current, axis=1)
        return np.all(diffs >= 0)
    
    # Initial check (should be True)
    if not check_network(network):
        # Fallback: return the OET network as is
        return network
        
    # Greedy Removal
    changed = True
    passes = 0
    max_passes = 5 # Reduced passes to ensure time limit
    
    while changed and passes < max_passes and time.time() < deadline:
        changed = False
        passes += 1
        # Iterate in reverse
        for idx in range(len(network) - 1, -1, -1):
            if time.time() > deadline:
                break
            
            # Try removing network[idx]
            candidate = network[:idx] + network[idx+1:]
            
            # Check candidate
            if check_network(candidate):
                network = candidate
                changed = True
            # Else, keep it
        
        if time.time() > deadline:
            break

    # Ensure we return a valid, non-empty network
    if len(network) == 0:
        return generate_odd_even_transposition(N)
        
    return network
