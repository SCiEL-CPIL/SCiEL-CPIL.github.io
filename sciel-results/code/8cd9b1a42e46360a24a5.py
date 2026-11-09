import numpy as np
import time

N = 16

def construct():
    # Generate Bitonic Sort Network
    network = _bitonic_network(N)
    
    # Verify (optional, but good for safety)
    if _verify_network(network):
        return network
    else:
        # Fallback to OET if something went wrong
        return _oet_network(N)

def _bitonic_network(n):
    if n == 1:
        return []
    half = n // 2
    left = _bitonic_network(half)
    right = [(i + half, j + half) for i, j in _bitonic_network(half)]
    merge = _bitonic_merge(n)
    return left + right + merge

def _bitonic_merge(n):
    if n == 1:
        return []
    half = n // 2
    left_merge = _bitonic_merge(half)
    right_merge = [(i + half, j + half) for i, j in _bitonic_merge(half)]
    final = [(i, i + half) for i in range(half)]
    return left_merge + right_merge + final

def _oet_network(n):
    net = []
    for phase in range(n):
        for i in range(phase % 2, n - 1, 2):
            net.append((i, i + 1))
    return net

def _verify_network(network):
    if not network:
        return False
        
    idx = np.arange(2**16, dtype=np.uint32)
    inputs = ((idx[:, None] >> np.arange(16, dtype=np.uint32)) & 1).astype(np.uint8)
    
    for i, j in network:
        min_val = np.minimum(inputs[:, i], inputs[:, j])
        max_val = np.maximum(inputs[:, i], inputs[:, j])
        inputs[:, i] = min_val
        inputs[:, j] = max_val
        
    diffs = np.diff(inputs, axis=1)
    if np.any(diffs < 0):
        return False
    return True
