"""Large reciprocal-subsum-free sets: Greedy descending construction."""
import time
import math
import sys

_START = time.time()
_DEADLINE = 85.0  # Safety margin under 90 s wall-clock

N = 24

def _time_left():
    return _DEADLINE - (time.time() - _START)

# Precompute weights
_L, _W = None, None

def _init_weights():
    global _L, _W
    if _W is not None:
        return
    L = 1
    for k in range(1, 25):
        L = L * k // math.gcd(L, k)
    _L = L
    _W = [L // k for k in range(1, 25)]

_init_weights()

def _is_subset_sum(target, items):
    """Check if target can be formed by a subset sum of items.
    Uses meet-in-the-middle for efficiency."""
    if target < 0:
        return False
    if target == 0:
        return True # Empty subset sums to 0
    
    n = len(items)
    if n == 0:
        return target == 0
    
    # If target is larger than sum of all items, impossible
    if sum(items) < target:
        return False
    
    mid = n // 2
    left = items[:mid]
    right = items[mid:]
    
    left_sums = set()
    # Generate subset sums of left
    for mask in range(1 << len(left)):
        s = 0
        mm = mask
        while mm:
            lb = mm & (-mm)
            b = lb.bit_length() - 1
            s += left[b]
            mm ^= lb
        left_sums.add(s)
        
    # Check right subsets
    for mask in range(1 << len(right)):
        s = 0
        mm = mask
        while mm:
            lb = mm & (-mm)
            b = lb.bit_length() - 1
            s += right[b]
            mm ^= lb
        if s > target:
            continue
        if (target - s) in left_sums:
            return True
            
    return False

def _is_valid_addition(k_idx, selected_indices):
    """
    Check if adding element k_idx (0-based) to selected_indices (list of 0-based indices)
    results in a valid set.
    selected_indices does NOT contain k_idx.
    """
    S = selected_indices
    # 1. Check if k_idx is subsumed by S
    # w[k_idx] should NOT be a subset sum of weights in S
    if S:
        weights_S = [_W[i] for i in S]
        if _is_subset_sum(_W[k_idx], weights_S):
            return False
    else:
        # If S is empty, w[k_idx] cannot be sum of empty set (unless w[k_idx]==0, which is false)
        pass

    # 2. Check if any y in S becomes subsumed by S U {k_idx}
    for y in S:
        # We need to check if w[y] is a subset sum of (S U {k_idx}) \ {y}
        # This set is (S \ {y}) U {k_idx}
        # w[y] = sum(subset of S\{y}) + (0 or 1)*w[k_idx]
        # Case A: w[k_idx] not in subset. Then w[y] is subset sum of S\{y}. 
        #          This was already false when y was added (assuming we maintain validity).
        #          But wait, did adding other elements later make it true?
        #          Yes, so we must have checked it when those elements were added.
        #          So if the set was valid before adding k, w[y] was NOT a subset sum of S\{y}.
        # Case B: w[k_idx] IS in subset. Then w[y] - w[k_idx] must be a subset sum of S\{y}.
        diff = _W[y] - _W[k_idx]
        if diff >= 0:
            others = [i for i in S if i != y]
            if others:
                weights_others = [_W[i] for i in others]
                if _is_subset_sum(diff, weights_others):
                    return False
            else:
                if diff == 0:
                    return False
        # If diff < 0, impossible since weights are positive.
        
    return True

def construct():
    _init_weights()
    selected = []
    
    # Greedy descending: try 24, 23, ..., 1
    for k in range(24, 0, -1):
        k_idx = k - 1
        if _time_left() < 1.0:
            break
            
        if _is_valid_addition(k_idx, selected):
            selected.append(k_idx)
            
    if not selected:
        return [24]
        
    return [i + 1 for i in selected]
