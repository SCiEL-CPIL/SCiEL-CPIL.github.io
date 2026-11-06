import time
import numpy as np

N = 16

def _batcher_oes(n):
    if n == 1:
        return []
    half = n // 2
    left = _batcher_oes(half)
    right = _batcher_oes(half)
    right = [(i + half, j + half) for i, j in right]
    merge = _batcher_merge(half)
    return left + right + merge

def _batcher_merge(k):
    comparators = []
    for r in range(int(np.log2(k))):
        step = 1 << r
        for i in range(0, 2*k, 2*step):
            for j in range(i, i + step):
                comparators.append((j, j + step))
    return comparators

def _is_sorted_network_fast(network):
    for mask in range(1 << N):
        vec = mask
        for i, j in network:
            vi = (vec >> i) & 1
            vj = (vec >> j) & 1
            if vi > vj:
                vec ^= (1 << i) | (1 << j)
        count_ones = bin(vec).count('1')
        expected = ((1 << count_ones) - 1) << (N - count_ones)
        if vec != expected:
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 55
    
    # Start with Batcher's OES (62 comparators)
    network = _batcher_oes(N)
    
    # Greedy removal
    while time.time() < deadline:
        removed = False
        for idx in range(len(network)):
            if time.time() > deadline:
                break
            temp_network = network[:idx] + network[idx+1:]
            if _is_sorted_network_fast(temp_network):
                network = temp_network
                removed = True
                break
        if not removed:
            break
            
    return network
