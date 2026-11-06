import numpy as np
import time
import random

N = 16
SORTED_SET = set((1 << k) - 1 for k in range(N + 1))

def _merge(k):
    """
    Generates comparators to merge two sorted lists of length k.
    The lists are assumed to be at indices 0..k-1 and k..2k-1.
    """
    if k == 1:
        return [(0, 1)]
    
    k_half = k // 2
    
    # 1. Merge even-indexed elements
    even_comps = []
    for i, j in _merge(k_half):
        mi = 2 * i if i < k_half else 2 * (i - k_half) + k
        mj = 2 * j if j < k_half else 2 * (j - k_half) + k
        even_comps.append((mi, mj))
        
    # 2. Merge odd-indexed elements
    odd_comps = []
    for i, j in _merge(k_half):
        mi = 2 * i + 1 if i < k_half else 2 * (i - k_half) + k + 1
        mj = 2 * j + 1 if j < k_half else 2 * (j - k_half) + k + 1
        odd_comps.append((mi, mj))
        
    # 3. Final comparisons between adjacent pairs (2i+1, 2i+2)
    final_comps = [(2 * i + 1, 2 * i + 2) for i in range(k - 1)]
    
    return even_comps + odd_comps + final_comps

def _batcher(n):
    """
    Generates Batcher's Odd-Even Merge Sort network for n wires.
    """
    if n == 1:
        return []
    
    k = n // 2
    left = _batcher(k)
    right = [(i + k, j + k) for i, j in _batcher(k)]
    merge = _merge(k)
    
    return left + right + merge

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
    
    # Start with Batcher's Odd-Even Merge Sort (63 comparators)
    # This is a strong starting point compared to the seed's 120 comparators
    network = _batcher(N)
    
    # Create samples for fast verification
    all_possible = list(range(1 << N))
    rng.shuffle(all_possible)
    fast_sample = np.array(all_possible[:500], dtype=np.int32)
    med_sample = np.array(all_possible[:2000], dtype=np.int32)
    
    # Aggressively remove redundant comparators
    # Phase 1: Greedy single-removal, scanning from middle outward
    # Middle comparators are more likely to be redundant in a merge network
    while time.time() < deadline:
        removed = False
        n_comps = len(network)
        
        if n_comps == 0:
            break
        
        # Build order: middle indices first, then edges
        mid = n_comps // 2
        order = list(range(mid, n_comps)) + list(range(mid - 1, -1, -1))
        
        for idx in order:
            if time.time() > deadline:
                break
            
            new_network = network[:idx] + network[idx+1:]
            if _fast_check(new_network, fast_sample):
                if _fast_check(new_network, med_sample):
                    if _full_check(new_network):
                        network = new_network
                        removed = True
                        break
        
        if not removed:
            break
    
    # Phase 2: Try removing pairs of adjacent comparators
    while time.time() < deadline:
        removed = False
        n_comps = len(network)
        
        if n_comps < 2:
            break
        
        for i in range(n_comps - 1):
            if time.time() > deadline:
                break
            
            new_network = network[:i] + network[i+2:]
            if _fast_check(new_network, fast_sample):
                if _fast_check(new_network, med_sample):
                    if _full_check(new_network):
                        network = new_network
                        removed = True
                        break
        
        if not removed:
            break
    
    # Phase 3: Final single-removal pass (in case phase 2 opened up new singles)
    while time.time() < deadline:
        removed = False
        n_comps = len(network)
        
        if n_comps == 0:
            break
        
        for idx in range(n_comps):
            if time.time() > deadline:
                break
            
            new_network = network[:idx] + network[idx+1:]
            if _fast_check(new_network, fast_sample):
                if _fast_check(new_network, med_sample):
                    if _full_check(new_network):
                        network = new_network
                        removed = True
                        break
        
        if not removed:
            break
    
    return network
