import numpy as np
import time

N = 18
NUM_MASKS = 1 << N

# Precompute expected sorted states
EXPECTED_STATES = np.empty(NUM_MASKS, dtype=np.int64)
for m in range(NUM_MASKS):
    k = bin(m).count('1')
    EXPECTED_STATES[m] = ((1 << k) - 1) << (N - k)

def is_valid(net):
    if not net:
        return False # For N=18, empty network is not a sorting network
        
    states = np.arange(NUM_MASKS, dtype=np.int64)
    
    for i, j in net:
        bits_i = (states >> i) & 1
        bits_j = (states >> j) & 1
        swap_mask = (bits_i > bits_j)
        if np.any(swap_mask):
            states[swap_mask] ^= (1 << i)
            states[swap_mask] ^= (1 << j)
            
    return np.all(states == EXPECTED_STATES)

def construct():
    # Seed network
    net = [(i, i + 1) for phase in range(N)
           for i in range(phase % 2, N - 1, 2)]
    
    start_time = time.time()
    deadline = 85.0
    
    best_net = net
    best_len = len(net)
    
    # Local Search
    while time.time() < deadline:
        # Try to remove a random comparator
        if len(net) > 0:
            idx = np.random.randint(len(net))
            test_net = net[:idx] + net[idx+1:]
            if is_valid(test_net):
                net = test_net
                if len(net) < best_len:
                    best_len = len(net)
                    best_net = net
            else:
                # Try to swap with a random other comparator
                if len(net) > 1:
                    idx2 = np.random.randint(len(net))
                    if idx2 != idx:
                        test_net = net.copy()
                        test_net[idx], test_net[idx2] = test_net[idx2], test_net[idx]
                        if is_valid(test_net):
                            net = test_net
                # Try to change wires of a random comparator
                if len(net) > 0:
                    idx = np.random.randint(len(net))
                    i, j = net[idx]
                    new_i = np.random.randint(N)
                    new_j = np.random.randint(N)
                    if new_i != new_j:
                        if new_i > new_j:
                            new_i, new_j = new_j, new_i
                        test_net = net.copy()
                        test_net[idx] = (new_i, new_j)
                        if is_valid(test_net):
                            net = test_net
        else:
            break
            
    return best_net
