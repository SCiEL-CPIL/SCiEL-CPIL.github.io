import time
import numpy as np
import random

N = 18

# EVOLVE-BLOCK-START
_DEADLINE = 85.0  # Leave 5s margin for cleanup/return
_start_time = None

def _check_time():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return time.time() - _start_time

def _odd_even_transposition(n=N):
    """Odd-even transposition sort network."""
    comps = []
    for phase in range(n):
        for i in range(phase % 2, n - 1, 2):
            comps.append((i, i + 1))
    return comps

def _is_valid_network_numpy(comparators, n=N):
    """
    Check if a comparator list sorts all 2^n Boolean inputs using NumPy vectorization.
    This is significantly faster than a pure Python loop.
    """
    if not comparators:
        return n <= 1
    
    # Quick structural checks
    for i, j in comparators:
        if not (0 <= i < j < n):
            return False

    # Generate all 2^n inputs
    # Shape: (2^n, n)
    # We can use np.unpackbits or just create the range and bit-shift
    # For n=18, 2^18 = 262144.
    # Creating an array of 262144 x 18 integers is ~4.7 MB, which is fine.
    
    # Efficient way to create the matrix:
    # Create an array of indices 0..2^n - 1
    # Then for each bit position k, extract the k-th bit.
    
    # Method:
    # inputs = np.arange(1 << n, dtype=np.uint8).reshape(-1, 1)
    # bits = (inputs >> np.arange(n, dtype=np.uint8)) & 1
    # This creates a (2^n, n) array of 0s and 1s.
    
    try:
        # Use uint8 for efficiency
        indices = np.arange(1 << n, dtype=np.uint8).reshape(-1, 1)
        shifts = np.arange(n, dtype=np.uint8)
        wires = (indices >> shifts) & 1
        
        # Apply comparators
        for i, j in comparators:
            wi = wires[:, i]
            wj = wires[:, j]
            # min to i, max to j
            wires[:, i] = np.minimum(wi, wj)
            wires[:, j] = np.maximum(wi, wj)
        
        # Check if sorted
        # A sequence is sorted if wires[:, k] <= wires[:, k+1] for all k
        # We can check if any wire[k] > wire[k+1]
        diffs = wires[:, 1:] - wires[:, :-1]
        # If any diff is negative, it's not sorted.
        # Since values are 0 or 1, diff is -1, 0, or 1.
        return not np.any(diffs < 0)
    except Exception:
        return False

def _greedy_prune(comparators, time_limit=80.0):
    """
    Greedily remove comparators while maintaining validity.
    Uses NumPy-based validity check.
    """
    comps = list(comparators)
    deadline = _start_time + time_limit if _start_time else time.time() + time_limit
    
    # Initial check
    if not _is_valid_network_numpy(comps):
        return comps

    # Strategy: 
    # 1. Try removing from the end first (often less critical in transposition sorts)
    # 2. Then random removals to escape local minima
    
    # Pass 1: Reverse order removal
    i = len(comps) - 1
    while i >= 0 and time.time() < deadline:
        if _check_time() > deadline:
            break
        # Try removing
        new_comps = comps[:i] + comps[i+1:]
        if _is_valid_network_numpy(new_comps):
            comps = new_comps
        else:
            i -= 1
            
    # Pass 2: Forward order removal
    pass_count = 0
    while time.time() < deadline and pass_count < 5:
        changed = False
        i = 0
        while i < len(comps) and time.time() < deadline:
            new_comps = comps[:i] + comps[i+1:]
            if _is_valid_network_numpy(new_comps):
                comps = new_comps
                changed = True
                # Don't increment i, check the new element at position i
            else:
                i += 1
        pass_count += 1
        if not changed:
            break
            
    # Pass 3: Random removal attempts to break symmetry/stuck states
    while time.time() < deadline - 5:
        if len(comps) == 0:
            break
        idx = random.randint(0, len(comps) - 1)
        new_comps = comps[:idx] + comps[idx+1:]
        if _is_valid_network_numpy(new_comps):
            comps = new_comps

    return comps

def _swap_comparators(comparators, i, j):
    """Try swapping two comparators."""
    new_comps = list(comparators)
    new_comps[i], new_comps[j] = new_comps[j], new_comps[i]
    return _is_valid_network_numpy(new_comps)

def _local_search(comparators, time_limit=80.0):
    """
    Local search: try removing and swapping comparators.
    """
    comps = list(comparators)
    deadline = _start_time + time_limit if _start_time else time.time() + time_limit
    
    best = comps
    best_len = len(comps)
    
    # First, greedy removal
    i = len(comps) - 1
    while i >= 0 and time.time() < deadline:
        if _check_time() > deadline:
            break
        new_comps = comps[:i] + comps[i+1:]
        if _is_valid_network_numpy(new_comps):
            comps = new_comps
            best = comps
            best_len = len(comps)
        else:
            i -= 1
            
    # Then, try more removals in forward direction
    for pass_num in range(3):
        if time.time() >= deadline:
            break
        i = 0
        while i < len(comps) and time.time() < deadline:
            new_comps = comps[:i] + comps[i+1:]
            if _is_valid_network_numpy(new_comps):
                comps = new_comps
                best = comps
                best_len = len(comps)
            else:
                i += 1
                
    # Local search: try swapping adjacent comparators to enable further removal
    # This is a heuristic to reorder the network to make some comparators redundant
    if time.time() < deadline - 10:
        # Simple random swaps
        for _ in range(100): # Limit swaps to avoid infinite loop
            if time.time() > deadline - 5:
                break
            if len(comps) < 2:
                break
            i = random.randint(0, len(comps) - 2)
            j = i + 1
            new_comps = list(comps)
            new_comps[i], new_comps[j] = new_comps[j], new_comps[i]
            if _is_valid_network_numpy(new_comps):
                # If valid, try to prune again
                pruned = _greedy_prune(new_comps, time_limit=0.5)
                if len(pruned) < best_len:
                    best = pruned
                    best_len = len(pruned)
                    comps = pruned

    return best

def construct():
    """
    Construct a sorting network for 18 wires.
    Returns an ordered list of integer pairs (i, j) where 0 <= i < j < 18.
    """
    global _start_time
    _start_time = time.time()
    
    # Generate initial candidate: Odd-even transposition
    # This is known to be valid and has 153 comparators
    initial_comps = _odd_even_transposition(N)
    
    # Verify initial candidate (should be valid)
    if not _is_valid_network_numpy(initial_comps):
        # Fallback: return the seed as is
        return initial_comps

    # Perform local search / pruning
    # Allocate most of the time to this
    time_left = _DEADLINE - _check_time() - 5
    if time_left > 1:
        best_comps = _local_search(initial_comps, time_limit=time_left)
    else:
        best_comps = initial_comps

    # Final validity check
    if not _is_valid_network_numpy(best_comps):
        return initial_comps

    return best_comps
# EVOLVE-BLOCK-END
