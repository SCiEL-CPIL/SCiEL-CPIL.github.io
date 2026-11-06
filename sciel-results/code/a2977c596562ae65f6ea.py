import numpy as np
import time
import random

N = 16
TOTAL_INPUTS = 1 << N

def _is_sorted_mask(mask):
    """Check if a single 16-bit mask is sorted (all 1s before 0s)."""
    # Sorted means mask is of form 000...0111...1 (k ones at the bottom)
    # i.e., mask == (1 << k) - 1 for some k
    if mask == 0 or mask == TOTAL_INPUTS - 1:
        return True
    # Count trailing ones
    inv = (~mask) & (TOTAL_INPUTS - 1)
    # If sorted, inv should be of form 111...1000...0
    # i.e., inv & (inv + 1) == 0
    return (inv & (inv + 1)) == 0

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

def _simulate_all(network):
    """
    Simulate the network on all 65536 inputs. Returns a boolean array
    of length 65536 where True means the output is sorted.
    """
    masks = np.arange(TOTAL_INPUTS, dtype=np.int32)
    for i, j in network:
        bit_i = (masks >> i) & 1
        bit_j = (masks >> j) & 1
        swap = bit_i & (~bit_j)
        if np.any(swap):
            masks = np.where(swap, (masks & ~(1 << i)) | (1 << j), masks)
    # Check sortedness: for each mask, check if it's of form 0...01...1
    inv = (~masks) & (TOTAL_INPUTS - 1)
    sorted_mask = (inv & (inv + 1)) == 0
    return sorted_mask

def _simulate_partial(network, start_idx, end_idx):
    """
    Simulate network[start_idx:end_idx] on all inputs, given initial state.
    Returns the final masks array.
    """
    masks = np.arange(TOTAL_INPUTS, dtype=np.int32)
    for i, j in network[start_idx:end_idx]:
        bit_i = (masks >> i) & 1
        bit_j = (masks >> j) & 1
        swap = bit_i & (~bit_j)
        if np.any(swap):
            masks = np.where(swap, (masks & ~(1 << i)) | (1 << j), masks)
    return masks

def _is_sorted_array(masks):
    """Check which masks in the array are sorted. Returns boolean array."""
    inv = (~masks) & (TOTAL_INPUTS - 1)
    return (inv & (inv + 1)) == 0

def _compute_unsorted_set(network):
    """
    Compute the set of inputs that are NOT sorted after running the network.
    Returns a boolean array of length TOTAL_INPUTS (True = unsorted).
    """
    sorted_mask = _simulate_all(network)
    return ~sorted_mask

def _try_remove_comparator(network, idx, unsorted_set):
    """
    Try removing comparator at index idx. Check if the network still sorts all inputs.
    Returns True if removal is valid, False otherwise.
    """
    n = len(network)
    if n == 0:
        return False
    
    # Simulate the full network with this comparator removed
    # For efficiency, simulate in two parts: before idx and after idx
    # Actually, just simulate the full network minus this one comparator
    
    masks = np.arange(TOTAL_INPUTS, dtype=np.int32)
    for k in range(n):
        if k == idx:
            continue
        i, j = network[k]
        bit_i = (masks >> i) & 1
        bit_j = (masks >> j) & 1
        swap = bit_i & (~bit_j)
        if np.any(swap):
            masks = np.where(swap, (masks & ~(1 << i)) | (1 << j), masks)
    
    sorted_mask = _is_sorted_array(masks)
    return np.all(sorted_mask)

def _try_remove_comparator_fast(network, idx, pre_masks, post_masks):
    """
    Try removing comparator at index idx using precomputed prefix/suffix states.
    pre_masks: state before applying network[idx]
    post_masks: state after applying network[idx]
    Returns True if removal is valid.
    """
    # If we remove comparator idx, the state after the prefix is pre_masks
    # Then we apply the suffix (network[idx+1:])
    # But we need to check if the final result is sorted
    
    # Actually, this approach requires simulating the suffix from pre_masks
    # which is expensive. Let's just do a full simulation.
    pass

def construct():
    deadline = time.time() + 55
    rng = random.Random(42)
    
    # Start with Batcher's Odd-Even Merge Sort (63 comparators)
    network = _batcher(N)
    
    # Phase 1: Greedy removal using unsorted-set tracking
    # For each comparator, check if removing it still sorts all inputs
    while time.time() < deadline:
        removed = False
        n_comps = len(network)
        
        if n_comps == 0:
            break
        
        # Try removing each comparator, starting from the middle
        mid = n_comps // 2
        order = list(range(mid, n_comps)) + list(range(mid - 1, -1, -1))
        
        for idx in order:
            if time.time() > deadline:
                break
            
            if _try_remove_comparator(network, idx, None):
                network = network[:idx] + network[idx+1:]
                removed = True
                break
        
        if not removed:
            break
    
    # Phase 2: Try removing pairs of comparators
    while time.time() < deadline:
        removed = False
        n_comps = len(network)
        
        if n_comps < 2:
            break
        
        for i in range(n_comps - 1):
            if time.time() > deadline:
                break
            
            new_network = network[:i] + network[i+2:]
            if _try_remove_comparator(new_network, -1, None):  # dummy, just check full
                # Actually, let's just check if new_network sorts everything
                sorted_mask = _simulate_all(new_network)
                if np.all(sorted_mask):
                    network = new_network
                    removed = True
                    break
        
        if not removed:
            break
    
    # Phase 3: Final single-removal pass
    while time.time() < deadline:
        removed = False
        n_comps = len(network)
        
        if n_comps == 0:
            break
        
        for idx in range(n_comps):
            if time.time() > deadline:
                break
            
            if _try_remove_comparator(network, idx, None):
                network = network[:idx] + network[idx+1:]
                removed = True
                break
        
        if not removed:
            break
    
    return network
