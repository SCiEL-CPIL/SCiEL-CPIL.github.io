"""Optimized solver for large reciprocal weight under lcm separation."""
# EVOLVE-BLOCK-START
import time
from math import gcd, lcm
from fractions import Fraction
from itertools import combinations
import random

# Precompute conflict masks: for each x in [2,256], bitmask of y in [2,256] where lcm(x,y) <= 256
# We index numbers 2..256 as 0..254
N = 255  # numbers 2..256 inclusive
VALS = list(range(2, 257))

# conflict_mask[i] is a Python int bitmask where bit j is set if VALS[i] and VALS[j] conflict (lcm <= 256)
conflict_mask = [0] * N

for i in range(N):
    a = VALS[i]
    mask = 0
    for j in range(N):
        if i == j:
            continue
        b = VALS[j]
        if lcm(a, b) <= 256:
            mask |= (1 << j)
    conflict_mask[i] = mask

# Precompute reciprocal values as Fractions for exact comparison
recip = [Fraction(1, VALS[i]) for i in range(N)]

def can_add(selected_mask, idx):
    """Check if we can add index idx to the set represented by selected_mask."""
    return (selected_mask & conflict_mask[idx]) == 0

def total_recip(selected_mask):
    """Compute total reciprocal sum for a bitmask."""
    s = Fraction(0)
    m = selected_mask
    while m:
        lsb = m & (-m)
        idx = lsb.bit_length() - 1
        s += recip[idx]
        m ^= lsb
    return s

def greedy_construct(order=None):
    """Greedy construction with optional custom ordering."""
    if order is None:
        # Sort by 1/x descending (i.e., x ascending) - standard greedy
        order = list(range(N))
        # Actually for maximizing sum, we want smaller numbers first
        order = sorted(range(N), key=lambda i: VALS[i])
    
    selected_mask = 0
    for idx in order:
        if can_add(selected_mask, idx):
            selected_mask |= (1 << idx)
    return selected_mask

def local_search(initial_mask, time_limit=85.0):
    """Local search with add/swap/delete moves to maximize reciprocal sum."""
    start_time = time.time()
    best_mask = initial_mask
    best_val = total_recip(initial_mask)
    
    # Precompute list of indices for faster iteration
    all_indices = list(range(N))
    
    # For efficiency, maintain a set of selected indices
    def mask_to_set(m):
        s = set()
        while m:
            lsb = m & (-m)
            idx = lsb.bit_length() - 1
            s.add(idx)
            m ^= lsb
        return s
    
    selected = mask_to_set(best_mask)
    
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        
        # Try adding elements
        best_add_idx = -1
        best_add_gain = Fraction(0)
        for idx in all_indices:
            if idx in selected:
                continue
            if can_add(best_mask, idx):
                gain = recip[idx]
                if gain > best_add_gain:
                    best_add_gain = gain
                    best_add_idx = idx
        
        if best_add_idx >= 0:
            best_mask |= (1 << best_add_idx)
            selected.add(best_add_idx)
            best_val += best_add_gain
            improved = True
            continue
        
        # Try swaps: remove one, add another (or add multiple)
        # Single swap: remove i, add j
        best_swap = None  # (i, j, gain)
        for i in list(selected):
            for j in all_indices:
                if j in selected:
                    continue
                # Check if j can be added after removing i
                new_mask = best_mask & ~(1 << i)
                if can_add(new_mask, j):
                    gain = recip[j] - recip[i]
                    if best_swap is None or gain > best_swap[2]:
                        best_swap = (i, j, gain)
        
        if best_swap and best_swap[2] > 0:
            i, j, gain = best_swap
            best_mask = (best_mask & ~(1 << i)) | (1 << j)
            selected.remove(i)
            selected.add(j)
            best_val += gain
            improved = True
            continue
        
        # Try 2-for-1 swaps: remove two, add one
        # This is more expensive, so only do it if we have time
        if (time.time() - start_time) < time_limit - 5:
            sel_list = list(selected)
            best_21_swap = None  # (i, j, k, gain)
            for idx_i in range(len(sel_list)):
                i = sel_list[idx_i]
                for idx_j in range(idx_i + 1, len(sel_list)):
                    j = sel_list[idx_j]
                    new_mask = best_mask & ~(1 << i) & ~(1 << j)
                    for k in all_indices:
                        if k in selected:
                            continue
                        if can_add(new_mask, k):
                            gain = recip[k] - recip[i] - recip[j]
                            if best_21_swap is None or gain > best_21_swap[3]:
                                best_21_swap = (i, j, k, gain)
            
            if best_21_swap and best_21_swap[3] > 0:
                i, j, k, gain = best_21_swap
                best_mask = (best_mask & ~(1 << i) & ~(1 << j)) | (1 << k)
                selected.remove(i)
                selected.remove(j)
                selected.add(k)
                best_val += gain
                improved = True
                continue
        
        # Try 1-for-2 swaps: remove one, add two
        if (time.time() - start_time) < time_limit - 3:
            for i in list(selected):
                new_mask = best_mask & ~(1 << i)
                # Find best pair to add
                best_pair = None  # (j, k, gain)
                for j in all_indices:
                    if j in selected:
                        continue
                    if not can_add(new_mask, j):
                        continue
                    mask_after_j = new_mask | (1 << j)
                    for k in all_indices:
                        if k in selected or k == j:
                            continue
                        if can_add(mask_after_j, k):
                            gain = recip[j] + recip[k] - recip[i]
                            if best_pair is None or gain > best_pair[2]:
                                best_pair = (j, k, gain)
                
                if best_pair and best_pair[2] > 0:
                    j, k, gain = best_pair
                    best_mask = (best_mask & ~(1 << i)) | (1 << j) | (1 << k)
                    selected.remove(i)
                    selected.add(j)
                    selected.add(k)
                    best_val += gain
                    improved = True
                    break
        
        if (time.time() - start_time) > time_limit:
            break
    
    return best_mask, best_val

def multi_start_greedy(num_starts=50, time_limit=80.0):
    """Try multiple greedy orderings and pick the best."""
    start_time = time.time()
    best_mask = 0
    best_val = Fraction(0)
    
    # Standard greedy first
    mask = greedy_construct()
    val = total_recip(mask)
    if val > best_val:
        best_val = val
        best_mask = mask
    
    # Try random perturbations
    remaining_time = time_limit - (time.time() - start_time)
    attempts = 0
    while (time.time() - start_time) < time_limit - 10:
        # Create a random ordering with bias toward small numbers
        order = list(range(N))
        # Shuffle but keep small numbers more likely to appear early
        random.shuffle(order)
        # Sort by a mix of value and random
        order.sort(key=lambda i: VALS[i] + random.random() * 50)
        
        mask = greedy_construct(order)
        val = total_recip(mask)
        if val > best_val:
            best_val = val
            best_mask = mask
        attempts += 1
    
    return best_mask, best_val, attempts

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Leave 5s safety margin
    
    # Phase 1: Multi-start greedy
    best_mask, best_val, _ = multi_start_greedy(num_starts=100, time_limit=min(20.0, deadline - start_time - 10))
    
    # Phase 2: Local search
    if time.time() < deadline - 5:
        remaining = deadline - time.time() - 2
        best_mask, best_val = local_search(best_mask, time_limit=remaining)
    
    # Convert mask back to list of values
    result = []
    m = best_mask
    while m:
        lsb = m & (-m)
        idx = lsb.bit_length() - 1
        result.append(VALS[idx])
        m ^= lsb
    
    # Ensure non-empty
    if not result:
        result = [2]
    
    return sorted(result)
# EVOLVE-BLOCK-END
