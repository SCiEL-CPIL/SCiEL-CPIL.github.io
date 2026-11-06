import numpy as np
import time

N = 18

def construct():
    start_time = time.time()
    deadline = start_time + 80
    
    # Pre-compute all binary inputs
    # This is done once.
    inputs = np.arange(2**N)
    bits = (inputs[:, None] >> np.arange(N)) & 1
    
    def is_valid(net):
        state = bits.copy()
        for i, j in net:
            min_val = np.minimum(state[:, i], state[:, j])
            max_val = np.maximum(state[:, i], state[:, j])
            state[:, i] = min_val
            state[:, j] = max_val
            
        if not np.all(state[:, :-1] <= state[:, 1:]):
            return False
        return True

    # Seed: Odd-Even Transposition
    network = [(i, i + 1) for phase in range(N)
               for i in range(phase % 2, N - 1, 2)]
    
    # Greedy Reduction
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for idx in range(len(network) - 1, -1, -1):
            if time.time() > deadline:
                break
            
            c = network[idx]
            new_net = network[:idx] + network[idx+1:]
            
            if is_valid(new_net):
                network = new_net
                changed = True
                break # Restart scan
    
    # Optional: Local Search to try to find even smaller
    # Try to add a random comparator and reduce
    best_net = network
    best_len = len(network)
    
    # We can try a few random additions
    # But greedy reduction is the main thing.
    
    return best_net
