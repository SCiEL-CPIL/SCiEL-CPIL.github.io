import time
import numpy as np
import random

N = 16

def _verify_network_fast(network, n=16):
    """Verify network sorts all 2^n inputs using zero-one principle with NumPy."""
    if not network:
        return n <= 1
    num_inputs = 1 << n
    inputs = np.arange(num_inputs, dtype=np.uint32)
    masks = (1 << np.arange(n, dtype=np.uint32))
    vals = ((inputs[:, None] & masks[None, :]) > 0).astype(np.uint8)
    
    for i, j in network:
        if i < 0 or i >= n or j < 0 or j >= n or i >= j:
            return False
        vi = vals[:, i]
        vj = vals[:, j]
        vi_min = vi & vj
        vj_max = vi | vj
        vals[:, i] = vi_min
        vals[:, j] = vj_max
        
    diffs = vals[:, 1:] - vals[:, :-1]
    return np.all(diffs >= 0)

def _bitonic_merge(n, direction):
    if n <= 1:
        return []
    comparators = []
    half = n // 2
    for i in range(half):
        if direction == 0:
            comparators.append((i, i + half))
        else:
            comparators.append((i + half, i))
    comparators.extend([(i, j) for i, j in _bitonic_merge(half, direction)])
    comparators.extend([(i + half, j + half) for i, j in _bitonic_merge(half, direction)])
    return comparators

def _bitonic_sort(n, direction=0):
    if n <= 1:
        return []
    comparators = []
    half = n // 2
    comparators.extend(_bitonic_sort(half, direction))
    comparators.extend([(i + half, j + half) for i, j in _bitonic_sort(half, 1 - direction)])
    comparators.extend(_bitonic_merge(n, direction))
    return comparators

def construct():
    start_time = time.time()
    deadline = start_time + 45
    
    # Start with Batcher's Bitonic Sort network
    best_net = _bitonic_sort(N, 0)
    
    # Try to reduce using greedy removal with random restarts
    for attempt in range(20): # More attempts
        if time.time() > deadline:
            break
        net = list(best_net)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            indices = list(range(len(net)))
            random.shuffle(indices)
            for idx in indices:
                if time.time() > deadline:
                    return best_net
                new_net = net[:idx] + net[idx+1:]
                if _verify_network_fast(new_net, N):
                    net = new_net
                    improved = True
                    break
            if len(net) < len(best_net):
                best_net = list(net)
                
    # Final verification
    if _verify_network_fast(best_net, N):
        return best_net
    else:
        # Fallback to original Batcher
        return _bitonic_sort(N, 0)
