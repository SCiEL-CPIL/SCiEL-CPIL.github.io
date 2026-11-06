import numpy as np
import time
import random

N = 16
SORTED_SET = set((1 << k) - 1 for k in range(N + 1))

def _simulate_subset(network, masks):
    masks = masks.copy()
    for i, j in network:
        bit_i = (masks >> i) & 1
        bit_j = (masks >> j) & 1
        swap = bit_i & (~bit_j)
        if np.any(swap):
            masks = np.where(swap, (masks & ~(1 << i)) | (1 << j), masks)
    return masks

def _fast_check(network, sample_masks):
    results = _simulate_subset(network, sample_masks)
    return all(int(r) in SORTED_SET for r in results)

def _full_check(network):
    all_masks = np.arange(1 << N, dtype=np.int32)
    results = _simulate_subset(network, all_masks)
    return all(int(r) in SORTED_SET for r in results)

def construct():
    deadline = time.time() + 55
    rng = random.Random(42)
    
    network = [(i, i + 1) for phase in range(N)
               for i in range(phase % 2, N - 1, 2)]
    
    # Create samples
    all_possible = list(range(1 << N))
    rng.shuffle(all_possible)
    fast_sample = np.array(all_possible[:500], dtype=np.int32)
    med_sample = np.array(all_possible[:2000], dtype=np.int32)
    
    changed = True
    while changed and time.time() < deadline:
        changed = False
        # Try removing each comparator
        indices = list(range(len(network)))
        rng.shuffle(indices)
        for idx in indices:
            if time.time() > deadline:
                break
            new_network = network[:idx] + network[idx+1:]
            if _fast_check(new_network, fast_sample):
                if _fast_check(new_network, med_sample):
                    if _full_check(new_network):
                        network = new_network
                        changed = True
                        break
    
    return network
