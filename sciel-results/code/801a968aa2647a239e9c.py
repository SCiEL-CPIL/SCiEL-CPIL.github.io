import time
import random
import numpy as np

N = 18
NUM_BOOLS = 1 << N
SEED = 153.0
DEADLINE = 85.0  # wall-clock safety margin (evaluator kills at 90s)

# ---------------------------------------------------------------------------
# Fast verification using bitmasks over all 2^N Boolean inputs.
# Wire i's mask is a NUM_BOOLS-bit integer where bit x is 1 iff wire i holds
# the value 1 for Boolean input x.
# ---------------------------------------------------------------------------
_ALL_ONES = (1 << NUM_BOOLS) - 1
_TOP_BIT = 1 << (NUM_BOOLS - 1)
_ANY_BIT = 1  # bit 0 (all-zeros input)

def _initial_masks():
    """Wire i's mask = bit (1<<i) set on every input where bit i of the input is 1."""
    masks = [0] * N
    for i in range(N):
        m = 0
        for x in range(1 << i):
            m |= 1 << ((x << 1) | (1 << i))
        masks[i] = m
    return masks

def _verify(masks):
    """Check if the network sorts all inputs: top wire must be 1 on every non-zero input."""
    return (masks[0] & _ANY_BIT) == _ANY_BIT

def _apply(masks, comps):
    for i, j in comps:
        a, b = masks[i], masks[j]
        masks[i] = a & b
        masks[j] = a | b
    return masks

def _verify_comps(comps):
    return _verify(_apply(_initial_masks(), comps))

def _odd_even_transposition_network(n):
    """Standard odd-even transposition sorting network."""
    comps = []
    for phase in range(n):
        for i in range(phase % 2, n - 1, 2):
            comps.append((i, i + 1))
    return comps

def _batcher_oe_merge_sort(n, offset=0):
    """Batcher's odd-even merge sort network for n wires."""
    if n <= 1:
        return []
    comps = []
    half = n // 2
    comps.extend(_batcher_oe_merge_sort(half, offset))
    comps.extend(_batcher_oe_merge_sort(n - half, offset + half))
    comps.extend(_merge_batcher(half, n - half, offset))
    return comps

def _merge_batcher(k, m, offset):
    """Merge two sorted blocks of length k and m."""
    if k == 0 or m == 0:
        return []
    if k == 1 and m == 1:
        return [(offset, offset + 1)]
    comps = []
    k2 = (k + 1) // 2
    m2 = (m + 1) // 2
    comps.extend(_merge_batcher(k2, m2, offset))
    if k >= 2:
        comps.extend(_merge_batcher(k - k2, m - m2, offset + 1))
    for p in range(offset, offset + k + m, 2):
        if p + 1 < offset + k + m:
            comps.append((p, p + 1))
    return comps

def _greedy_remove(comps, budget):
    """Greedy removal: try to remove each comparator if the network remains valid."""
    masks = _apply(_initial_masks(), comps)
    kept = []
    for idx, (i, j) in enumerate(comps):
        if time.time() > budget:
            # If we time out, return kept + remaining to ensure validity
            return kept + comps[idx:]
        
        a, b = masks[i], masks[j]
        # Temporarily remove comparator
        masks[i] = a | b
        masks[j] = a
        if _verify(masks):
            # Comparator was redundant: leave it removed
            continue
        else:
            # Restore
            masks[i] = a & b
            masks[j] = a | b
            kept.append((i, j))
    return kept

def _simulated_annealing(comps, budget):
    """
    Simulated Annealing to escape local minima of greedy removal.
    Moves:
      1. Remove a comparator (if valid) -> accept
      2. Swap two comparators -> accept with probability based on "energy" (validity)
      3. Move a comparator to a different position -> accept with probability
    """
    current_comps = list(comps)
    current_masks = _apply(_initial_masks(), current_comps)
    current_valid = _verify(current_masks)
    best_comps = list(current_comps)
    best_len = len(current_comps)
    
    T = 1.0
    T_min = 0.01
    cooling = 0.995
    
    rng = random.Random(42)
    
    while time.time() < budget and T > T_min:
        # Pick a random move
        move_type = rng.random()
        n_comps = len(current_comps)
        
        if move_type < 0.5 and n_comps > 1:
            # Try to remove a random comparator
            idx = rng.randint(0, n_comps - 1)
            i, j = current_comps[idx]
            a, b = current_masks[i], current_masks[j]
            
            # Temporarily remove
            current_masks[i] = a | b
            current_masks[j] = a
            valid = _verify(current_masks)
            
            if valid:
                # Accept removal
                current_comps.pop(idx)
                current_valid = True
                if len(current_comps) < best_len:
                    best_comps = list(current_comps)
                    best_len = len(current_comps)
            else:
                # Restore
                current_masks[i] = a & b
                current_masks[j] = a | b
                # Reject with some probability to allow exploration
                if rng.random() > T:
                    current_valid = False
        elif move_type < 0.8 and n_comps > 1:
            # Swap two comparators
            idx1 = rng.randint(0, n_comps - 1)
            idx2 = rng.randint(0, n_comps - 1)
            if idx1 != idx2:
                # Swap
                current_comps[idx1], current_comps[idx2] = current_comps[idx2], current_comps[idx1]
                # Re-verify
                new_masks = _apply(_initial_masks(), current_comps)
                valid = _verify(new_masks)
                
                if valid:
                    current_masks = new_masks
                    current_valid = True
                    if len(current_comps) < best_len:
                        best_comps = list(current_comps)
                        best_len = len(current_comps)
                else:
                    # Revert
                    current_comps[idx1], current_comps[idx2] = current_comps[idx2], current_comps[idx1]
                    current_valid = False
        else:
            # Move a comparator to a different position
            if n_comps > 1:
                idx1 = rng.randint(0, n_comps - 1)
                idx2 = rng.randint(0, n_comps - 1)
                if idx1 != idx2:
                    # Remove from idx1 and insert at idx2
                    comp = current_comps.pop(idx1)
                    if idx2 > idx1:
                        idx2 -= 1
                    current_comps.insert(idx2, comp)
                    
                    # Re-verify
                    new_masks = _apply(_initial_masks(), current_comps)
                    valid = _verify(new_masks)
                    
                    if valid:
                        current_masks = new_masks
                        current_valid = True
                        if len(current_comps) < best_len:
                            best_comps = list(current_comps)
                            best_len = len(current_comps)
                    else:
                        # Revert
                        current_comps.pop(idx2)
                        current_comps.insert(idx1, comp)
                        current_valid = False
        
        T *= cooling
    
    return best_comps

def construct():
    start = time.time()
    budget = start + DEADLINE
    
    # Seed: Batcher's odd-even merge sort (better initial size than OETN)
    comps = _batcher_oe_merge_sort(N)
    if not _verify_comps(comps):
        comps = _odd_even_transposition_network(N)
    
    if not _verify_comps(comps):
        return comps
    
    best_comps = list(comps)
    best_len = len(comps)
    
    # Phase 1: Deterministic Greedy Removal
    if time.time() < budget - 20.0:
        new = _greedy_remove(comps, budget - 20.0)
        if len(new) < best_len:
            comps = new
            best_comps = list(comps)
            best_len = len(comps)
    
    # Phase 2: Simulated Annealing to escape local minima
    if time.time() < budget - 5.0:
        new = _simulated_annealing(comps, budget - 5.0)
        if len(new) < best_len:
            best_comps = list(new)
            best_len = len(new)
    
    # Final greedy pass on the best found
    if time.time() < budget - 1.0:
        new = _greedy_remove(best_comps, budget - 1.0)
        if len(new) < best_len:
            best_comps = list(new)
            best_len = len(new)
    
    # Guarantee a valid network is returned
    if not _verify_comps(best_comps):
        best_comps = _batcher_oe_merge_sort(N)
    
    return best_comps
