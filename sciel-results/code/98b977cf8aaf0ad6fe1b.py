import numpy as np
import time

N = 18

def construct():
    # Generate Seed Network (Odd-Even Transposition)
    network = [(i, i + 1) for phase in range(N)
               for i in range(phase % 2, N - 1, 2)]
    
    # Prepare Verification Data
    num_inputs = 1 << N
    indices = np.arange(num_inputs, dtype=np.uint64)
    # bits[k] is the k-th bit of the index (wire k)
    bits = (indices[:, None] >> np.arange(N)) & 1
    
    def is_valid(net):
        state = bits.copy()
        for i, j in net:
            a = state[:, i]
            b = state[:, j]
            mask = a > b
            if np.any(mask):
                tmp = a[mask].copy()
                a[mask] = b[mask]
                b[mask] = tmp
        return np.all(state[:, :-1] <= state[:, 1:])

    # Verify seed
    if not is_valid(network):
        return network

    best_network = network[:]
    best_len = len(network)
    
    start_time = time.time()
    deadline = start_time + 85.0
    
    no_improvement = 0
    
    while time.time() < deadline:
        # 1. Try Removal
        if len(network) > 0:
            idx = np.random.randint(0, len(network))
            new_network = network[:idx] + network[idx+1:]
            if is_valid(new_network):
                network = new_network
                best_network = network[:]
                best_len = len(network)
                no_improvement = 0
                continue

        # 2. Try Swap/Move if stuck
        no_improvement += 1
        if no_improvement > 20:
            no_improvement = 0
            if len(network) > 1:
                # Move a random comparator to a random position
                idx1 = np.random.randint(0, len(network))
                idx2 = np.random.randint(0, len(network))
                if idx1 != idx2:
                    elem = network.pop(idx1)
                    network.insert(idx2, elem)
                    if is_valid(network):
                        # Keep
                        pass
                    else:
                        # Revert
                        network.pop(idx2)
                        network.insert(idx1, elem)
                        
    return best_network
