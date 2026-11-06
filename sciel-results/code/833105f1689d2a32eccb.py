import numpy as np
import time
import random

N = 16
SORTED_SET = frozenset((1 << k) - 1 for k in range(N + 1))

def _simulate_subset(network, masks):
    for i, j in network:
        bit_i = (masks >> i) & 1
        bit_j = (masks >> j) & 1
        swap_mask = bit_i & (~bit_j)
        if np.any(swap_mask):
            masks = np.where(swap_mask, (masks & ~(1 << i)) | (1 << j), masks)
    return masks

def _is_sorted(masks):
    return all(int(m) in SORTED_SET for m in masks)

def _full_check(network):
    all_masks = np.arange(1 << N, dtype=np.int32)
    results = _simulate_subset(network, all_masks)
    return _is_sorted(results)

def _bitonic_merge(n, offset=0, ascending=True):
    if n <= 1:
        return []
    comps = []
    half = n // 2
    for i in range(half):
        a = offset + i
        b = offset + i + half
        if ascending:
            comps.append((a, b))
        else:
            comps.append((b, a))
    left = _bitonic_merge(half, offset, ascending)
    right = _bitonic_merge(half, offset + half, ascending)
    return comps + left + right

def _bitonic_sort(n, offset=0, ascending=True):
    if n <= 1:
        return []
    half = n // 2
    left = _bitonic_sort(half, offset, True)
    right = _bitonic_sort(half, offset + half, False)
    merge = _bitonic_merge(n, offset, True)
    return left + right + merge

def construct():
    deadline = time.time() + 55
    rng = random.Random(42)
    
    network = _bitonic_sort(N, 0, True)
    
    if not _full_check(network):
        network = [(i, i + 1) for phase in range(N)
                   for i in range(phase % 2, N - 1, 2)]
    
    all_possible = list(range(1 << N))
    rng.shuffle(all_possible)
    fast_sample = np.array(all_possible[:1000], dtype=np.int32)
    med_sample = np.array(all_possible[:4000], dtype=np.int32)
    
    def fast_check(net):
        return _is_sorted(_simulate_subset(net, fast_sample))
    
    def med_check(net):
        return _is_sorted(_simulate_subset(net, med_sample))
    
    best_network = network[:]
    best_len = len(network)
    
    # Phase 1: Greedy removal
    improved = True
    while improved and time.time() < deadline:
        improved = False
        indices = list(range(len(network)))
        rng.shuffle(indices)
        for idx in indices:
            if time.time() > deadline:
                break
            new_network = network[:idx] + network[idx+1:]
            if fast_check(new_network) and med_check(new_network):
                if _full_check(new_network):
                    network = new_network
                    if len(network) < best_len:
                        best_network = network[:]
                        best_len = len(network)
                    improved = True
                    break
    
    # Phase 2: Try replacing comparators
    if time.time() < deadline:
        all_pairs = [(i, j) for i in range(N) for j in range(i+1, N)]
        for _ in range(200):
            if time.time() > deadline:
                break
            idx = rng.randint(0, len(network) - 1)
            for new_comp in all_pairs:
                if new_comp in network:
                    continue
                new_network = network[:]
                new_network[idx] = new_comp
                if fast_check(new_network) and med_check(new_network):
                    if _full_check(new_network):
                        network = new_network
                        if len(network) < best_len:
                            best_network = network[:]
                            best_len = len(network)
                        break
    
    return best_network
