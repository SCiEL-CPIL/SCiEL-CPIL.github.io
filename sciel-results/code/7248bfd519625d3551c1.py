import time
import numpy as np

N = 18
MASKS = None

def _init_masks():
    global MASKS
    # Generate all 2^18 masks
    # Use bit manipulation to create a (2^18, 18) array
    # This is large: 2^18 * 18 * 1 byte (if uint8) = 4.7 MB. Fine.
    n_masks = 1 << N
    MASKS = np.zeros((n_masks, N), dtype=np.uint8)
    for i in range(N):
        MASKS[:, i] = (np.arange(n_masks) >> i) & 1

def _apply_network(network, x):
    # x is (2^18, 18)
    for i, j in network:
        # Get columns i and j
        # Compare
        # Swap if needed
        # Vectorized swap
        x_i = x[:, i]
        x_j = x[:, j]
        # min goes to i, max goes to j
        x[:, i] = np.minimum(x_i, x_j)
        x[:, j] = np.maximum(x_i, x_j)
    return x

def _is_sorted_network(network):
    x = MASKS.copy()
    _apply_network(network, x)
    # Check if sorted
    # A row is sorted if x[:, k] <= x[:, k+1] for all k
    # We can check np.all(x[:, :-1] <= x[:, 1:])
    return np.all(x[:, :-1] <= x[:, 1:])

def construct():
    start_time = time.time()
    deadline = start_time + 55
    
    _init_masks()
    
    # Seed: Odd-Even Transposition
    network = [(i, i + 1) for phase in range(N)
               for i in range(phase % 2, N - 1, 2)]
    
    # Verify
    assert _is_sorted_network(network)
    
    best_network = network
    best_size = len(network)
    
    improved = True
    while improved and (time.time() - start_time) < deadline:
        improved = False
        # Greedy Removal
        for idx in range(len(network) - 1, -1, -1):
            if (time.time() - start_time) > deadline - 5:
                break
            test_network = network[:idx] + network[idx+1:]
            if _is_sorted_network(test_network):
                network = test_network
                improved = True
                best_network = network
                best_size = len(network)
                break # Restart removal loop from end
        
        if not improved and (time.time() - start_time) < deadline - 10:
            # Try Add-Remove
            # Add a random comparator not in network
            existing = set(network)
            all_pairs = [(i, j) for i in range(N) for j in range(i+1, N)]
            candidates = [p for p in all_pairs if p not in existing]
            if candidates:
                # Try a few random additions
                for _ in range(5):
                    if (time.time() - start_time) > deadline - 5:
                        break
                    add_pair = candidates[np.random.randint(len(candidates))]
                    test_network = network + [add_pair]
                    # Try to remove one comparator
                    for idx in range(len(test_network) - 1, -1, -1):
                        if (time.time() - start_time) > deadline - 5:
                            break
                        if idx == len(test_network) - 1:
                            continue
                        removed = test_network[:idx] + test_network[idx+1:]
                        if len(removed) < len(network):
                            if _is_sorted_network(removed):
                                network = removed
                                improved = True
                                best_network = network
                                best_size = len(network)
                                break
                    if improved:
                        break
    
    return best_network
