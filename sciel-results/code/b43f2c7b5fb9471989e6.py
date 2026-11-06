"""Sorting network search for 17 wires with aggressive optimization."""
import numpy as np
import time
import random

N = 17
NUM_BOOL_INPUTS = 1 << N  # 131072
DEADLINE = 85.0  # safety margin before 90s kill

_start_time = time.time()

def _time_left():
    return DEADLINE - (time.time() - _start_time)

def _validate_network(comparators):
    """Check if network sorts all 2^N boolean inputs using numpy bit-packing."""
    if not comparators:
        return N == 1
    
    # Generate all 2^N boolean inputs as uint8 array
    # Each column is one input, each row is one wire
    # Use bit manipulation: input i has wire w set if bit w of i is 1
    inputs = np.arange(NUM_BOOL_INPUTS, dtype=np.uint32)
    # Expand to (N, NUM_BOOL_INPUTS) uint8 array
    wires = np.zeros((N, NUM_BOOL_INPUTS), dtype=np.uint8)
    for w in range(N):
        wires[w] = ((inputs >> w) & 1).astype(np.uint8)
    
    # Apply each comparator
    for i, j in comparators:
        # min goes to wire i, max goes to wire j
        a = wires[i]
        b = wires[j]
        wires[i] = np.minimum(a, b)
        wires[j] = np.maximum(a, b)
    
    # Check: for each input, wires should be sorted (non-decreasing)
    # wires[w] <= wires[w+1] for all w
    for w in range(N - 1):
        if np.any(wires[w] > wires[w + 1]):
            return False
    return True

def _validate_network_fast(comparators):
    """Faster validation using precomputed input matrix."""
    if not comparators:
        return N == 1
    
    # Precompute all inputs as a (NUM_BOOL_INPUTS, N) uint8 array
    # Then transpose to (N, NUM_BOOL_INPUTS) for efficient column ops
    inputs = np.arange(NUM_BOOL_INPUTS, dtype=np.uint32)
    wires = np.zeros((N, NUM_BOOL_INPUTS), dtype=np.uint8)
    for w in range(N):
        wires[w] = ((inputs >> w) & 1).astype(np.uint8)
    
    for i, j in comparators:
        a = wires[i]
        b = wires[j]
        # Use arithmetic: min = a*b (for 0/1), max = a+b-a*b
        # Actually for 0/1: min(a,b) = a AND b = a*b, max(a,b) = a OR b = a + b - a*b
        # But np.minimum/maximum should be fine and clear
        wires[i] = np.minimum(a, b)
        wires[j] = np.maximum(a, b)
    
    # Check sorted: for each column, values should be non-decreasing
    # wires[w] <= wires[w+1] for all w
    for w in range(N - 1):
        if np.any(wires[w] > wires[w + 1]):
            return False
    return True

def _make_bitonic_network(n):
    """Generate bitonic sort network for n wires (pad to power of 2 if needed)."""
    # For non-power-of-2, use the next power of 2 and ignore extra wires
    # Actually, let's just do bitonic for the largest power of 2 <= n, 
    # then handle remaining wires separately
    # Better: use bitonic merge approach
    
    comparators = []
    
    def bitonic_merge(lo, hi, direction):
        """Merge two bitonic sequences in wires [lo, hi)."""
        length = hi - lo
        if length <= 1:
            return
        half = length // 2
        # Compare-exchange elements i half apart
        for i in range(half):
            a = lo + i
            b = lo + i + half
            if direction == 0:  # ascending
                comparators.append((a, b))
            else:
                comparators.append((b, a))
        # Recursively merge
        bitonic_merge(lo, lo + half, direction)
        bitonic_merge(lo + half, hi, direction)
    
    def bitonic_sort(lo, hi, direction):
        length = hi - lo
        if length <= 1:
            return
        half = length // 2
        bitonic_sort(lo, lo + half, 0)       # ascending
        bitonic_sort(lo + half, hi, 1)        # descending
        bitonic_merge(lo, hi, direction)
    
    # For N=17, use bitonic sort on 16 wires + handle wire 16
    # Actually, let's just do a standard bitonic sort that works for any n
    # by padding to next power of 2
    
    power = 1
    while power < n:
        power *= 2
    
    # Generate for 'power' wires, but only use first n
    comparators = []
    
    def bitonic_merge_pad(lo, hi, direction):
        length = hi - lo
        if length <= 1:
            return
        half = length // 2
        for i in range(half):
            a = lo + i
            b = lo + i + half
            if a < n and b < n:
                if direction == 0:
                    comparators.append((a, b))
                else:
                    comparators.append((b, a))
        bitonic_merge_pad(lo, lo + half, direction)
        bitonic_merge_pad(lo + half, hi, direction)
    
    def bitonic_sort_pad(lo, hi, direction):
        length = hi - lo
        if length <= 1:
            return
        half = length // 2
        bitonic_sort_pad(lo, lo + half, 0)
        bitonic_sort_pad(lo + half, hi, 1)
        bitonic_merge_pad(lo, hi, direction)
    
    bitonic_sort_pad(0, power, 0)
    
    # Filter out comparators that reference wires >= n
    comparators = [(i, j) for i, j in comparators if i < n and j < n]
    return comparators

def _make_odd_even_transposition(n):
    """Odd-even transposition sorting network."""
    comparators = []
    for phase in range(n):
        for i in range(phase % 2, n - 1, 2):
            comparators.append((i, i + 1))
    return comparators

def _make_odd_even_merge(n):
    """Odd-even merge sort network."""
    comparators = []
    
    def odd_even_merge(lo, hi):
        length = hi - lo
        if length <= 1:
            return
        half = length // 2
        # Compare-exchange i and i+half for i in range(half)
        for i in range(half):
            a = lo + i
            b = lo + i + half
            if a < b:
                comparators.append((a, b))
            else:
                comparators.append((b, a))
        # Recursively merge
        odd_even_merge(lo, lo + half)
        odd_even_merge(lo + half, hi)
    
    def odd_even_sort(lo, hi):
        length = hi - lo
        if length <= 1:
            return
        half = length // 2
        odd_even_sort(lo, lo + half)
        odd_even_sort(lo + half, hi)
        odd_even_merge(lo, hi)
    
    odd_even_sort(0, n)
    return comparators

def _try_remove_comparators(comparators, num_passes=50):
    """Try to remove comparators one at a time, repeatedly."""
    current = list(comparators)
    
    for _ in range(num_passes):
        if _time_left() < 10:
            break
        removed = False
        for idx in range(len(current) - 1, -1, -1):
            if _time_left() < 10:
                break
            # Try removing comparator at idx
            candidate = current[:idx] + current[idx+1:]
            if _validate_network_fast(candidate):
                current = candidate
                removed = True
                break
        if not removed:
            break
    
    return current

def _local_search(comparators, max_steps=2000):
    """Local search: try moving or swapping comparators."""
    current = list(comparators)
    best = current
    best_len = len(current)
    
    step = 0
    no_improve_count = 0
    max_no_improve = 50
    
    while step < max_steps and _time_left() > 15:
        step += 1
        if no_improve_count > max_no_improve:
            break
        
        # Randomly pick a move type
        move_type = random.random()
        
        if move_type < 0.4:
            # Move a comparator to a different position
            if len(current) < 2:
                continue
            idx = random.randint(0, len(current) - 1)
            new_idx = random.randint(0, len(current) - 1)
            if idx == new_idx:
                continue
            comp = current.pop(idx)
            current.insert(new_idx, comp)
            
            if _validate_network_fast(current):
                # Keep the move
                no_improve_count = 0
            else:
                # Revert
                current.pop(new_idx)
                current.insert(idx, comp)
                no_improve_count += 1
        
        elif move_type < 0.7:
            # Swap two comparators
            if len(current) < 2:
                continue
            idx1, idx2 = random.sample(range(len(current)), 2)
            current[idx1], current[idx2] = current[idx2], current[idx1]
            
            if _validate_network_fast(current):
                no_improve_count = 0
            else:
                current[idx1], current[idx2] = current[idx2], current[idx1]
                no_improve_count += 1
        
        else:
            # Try to remove a random comparator
            if len(current) < 2:
                continue
            idx = random.randint(0, len(current) - 1)
            comp = current.pop(idx)
            
            if _validate_network_fast(current):
                no_improve_count = 0
                if len(current) < best_len:
                    best = list(current)
                    best_len = len(current)
            else:
                current.insert(idx, comp)
                no_improve_count += 1
    
    # Final: try removal again
    current = _try_remove_comparators(current, num_passes=30)
    if len(current) < best_len:
        best = current
        best_len = len(current)
    
    return best

def construct():
    """Main entry point: find smallest sorting network for N=17 wires."""
    global _start_time
    _start_time = time.time()
    
    # Generate candidate seed networks
    seeds = []
    
    # 1. Odd-even transposition (baseline, 136 comparators)
    seeds.append(_make_odd_even_transposition(N))
    
    # 2. Bitonic sort (should be ~80-90 comparators for 17 wires)
    seeds.append(_make_bitonic_network(N))
    
    # 3. Odd-even merge sort
    seeds.append(_make_odd_even_merge(N))
    
    # Validate and pick the best seed
    best_network = None
    best_count = float('inf')
    
    for seed in seeds:
        if _time_left() < 20:
            break
        if _validate_network_fast(seed):
            if len(seed) < best_count:
                best_count = len(seed)
                best_network = list(seed)
    
    if best_network is None:
        # Fallback to OETN
        best_network = _make_odd_even_transposition(N)
        best_count = len(best_network)
    
    # Aggressive deletion
    best_network = _try_remove_comparators(best_network, num_passes=100)
    best_count = len(best_network)
    
    # Local search
    if _time_left() > 30:
        result = _local_search(best_network, max_steps=3000)
        if len(result) < best_count:
            best_network = result
            best_count = len(result)
    
    # Final deletion pass
    if _time_left() > 10:
        best_network = _try_remove_comparators(best_network, num_passes=50)
    
    return best_network
