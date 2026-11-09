"""Size-efficient sorting network synthesis for N=18 wires."""
import time
import random
import numpy as np

N = 18
N_BOOL = 1 << N  # 262144
DEADLINE = 80.0  # safety margin under 90s wall-clock

# Precompute the sorted target for each Boolean input
# For a Boolean input (bitmask), sorted output has the top k bits set where k = popcount(mask)
# We represent a wire vector as an integer where wire i is bit i.
# After sorting, the k ones should be in the top k positions (wires N-k..N-1).

def _build_sorted_targets():
    """Precompute the sorted output bitmask for every input bitmask."""
    targets = np.zeros(N_BOOL, dtype=np.uint32)
    for mask in range(N_BOOL):
        k = bin(mask).count('1')
        # Top k bits set: bits (N-k) through (N-1)
        if k == 0:
            targets[mask] = 0
        else:
            targets[mask] = ((1 << k) - 1) << (N - k)
    return targets

def _apply_comparator(state, i, j):
    """Apply one compare-exchange to a state vector (numpy array of uint8).
    state[i] = min, state[j] = max."""
    a = state[i]
    b = state[j]
    if a > b:
        state[i] = b
        state[j] = a
    return state

def _verify_network(comparators, targets):
    """Verify that the network sorts all 2^N Boolean inputs.
    Uses bit-parallel: track all 2^N states simultaneously using numpy arrays.
    Each wire holds a uint32 array of length N_BOOL where the value is 0 or 1.
    """
    # Initialize: for each wire i, the value across all inputs is bit i of the input mask
    # wires[i][mask] = (mask >> i) & 1
    wires = np.zeros((N, N_BOOL), dtype=np.uint8)
    for i in range(N):
        # Bit i of each mask
        bit_val = ((np.arange(N_BOOL, dtype=np.uint32) >> i) & 1).astype(np.uint8)
        wires[i] = bit_val

    for (i, j) in comparators:
        a = wires[i]
        b = wires[j]
        # min: a if a < b else b  ->  for bits: min(a,b) = a & b (since values are 0/1)
        # max: a if a > b else b  ->  for bits: max(a,b) = a | b
        wires[i] = a & b
        wires[j] = a | b

    # Check all inputs
    # For each input mask, the sorted output should match targets[mask]
    # The current state on wire i is wires[i][mask]
    # We need: for each mask, the vector [wires[0][mask], ..., wires[N-1][mask]] == targets[mask]
    # Build the output bitmask for each input
    out_masks = np.zeros(N_BOOL, dtype=np.uint32)
    for i in range(N):
        out_masks |= wires[i].astype(np.uint32) << i

    return np.array_equal(out_masks, targets)

def _verify_network_fast(comparators, targets):
    """Faster verification using the fact that we only need to check a subset
    or use a more efficient representation."""
    # Actually for 18 wires, 2^18 = 262144 states, each wire is a uint8 array of 262144.
    # Total memory: 18 * 262144 * 1 byte = ~4.7 MB. This is fine.
    # The comparator operations are vectorized numpy ops.
    wires = np.zeros((N, N_BOOL), dtype=np.uint8)
    for i in range(N):
        bit_val = ((np.arange(N_BOOL, dtype=np.uint32) >> i) & 1).astype(np.uint8)
        wires[i] = bit_val

    for (i, j) in comparators:
        a = wires[i]
        b = wires[j]
        wires[i] = a & b
        wires[j] = a | b

    out_masks = np.zeros(N_BOOL, dtype=np.uint32)
    for i in range(N):
        out_masks |= wires[i].astype(np.uint32) << i

    return np.array_equal(out_masks, targets)

def _odd_even_transposition():
    """Standard odd-even transposition sorting network."""
    comps = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            comps.append((i, i + 1))
    return comps

def _bitonic_sort_network():
    """Bitonic sorting network for N=18 (not power of 2, so use a general approach).
    Actually, let's just use the odd-even transposition as a baseline and try to optimize."""
    return _odd_even_transposition()

def _greedy_construct(targets):
    """Greedy construction: start with empty network, at each step add the comparator
    that fixes the most remaining incorrect inputs. Use bit-parallel tracking."""
    # Track which inputs are not yet sorted
    # Initialize wires
    wires = np.zeros((N, N_BOOL), dtype=np.uint8)
    for i in range(N):
        bit_val = ((np.arange(N_BOOL, dtype=np.uint32) >> i) & 1).astype(np.uint8)
        wires[i] = bit_val

    # Compute which inputs are currently unsorted
    out_masks = np.zeros(N_BOOL, dtype=np.uint32)
    for i in range(N):
        out_masks |= wires[i].astype(np.uint32) << i
    unsorted = (out_masks != targets)
    
    comps = []
    max_comps = 200  # cap for greedy
    
    while unsorted.any() and len(comps) < max_comps:
        # Try all possible comparators and see which fixes the most unsorted inputs
        best_gain = -1
        best_comp = None
        best_new_unsorted = None
        best_new_wires = None
        
        for i in range(N):
            for j in range(i + 1, N):
                # Simulate applying comparator (i, j)
                a = wires[i]
                b = wires[j]
                new_ai = a & b
                new_bj = a | b
                
                # Check if this changes anything
                if np.array_equal(new_ai, a) and np.array_equal(new_bj, b):
                    continue
                
                # Compute new output masks only for the wires that changed
                # Actually we need to recompute the full out_mask for unsorted inputs
                # This is expensive. Let's be smarter.
                
                # For each unsorted input, check if applying (i,j) makes it sorted
                # We need the full output after applying (i,j)
                # Let's just compute the delta
                # new_out for wire i: new_ai, wire j: new_bj, others unchanged
                # out_mask_new = out_mask_old - (old_ai << i) - (old_bj << j) + (new_ai << i) + (new_bj << j)
                
                old_ai_bits = a.astype(np.uint32)
                old_bj_bits = b.astype(np.uint32)
                new_ai_bits = new_ai.astype(np.uint32)
                new_bj_bits = new_bj.astype(np.uint32)
                
                # Only consider unsorted inputs
                unsorted_idx = np.where(unsorted)[0]
                if len(unsorted_idx) == 0:
                    break
                    
                # For unsorted inputs, compute new out_mask
                # This is still expensive for large N_BOOL. Let's limit the search.
                # For now, just pick a random comparator and see if it helps.
                pass
        
        # Fallback: pick a random comparator that changes the state
        changed = False
        for _ in range(100):
            i = random.randint(0, N - 2)
            j = random.randint(i + 1, N - 1)
            a = wires[i]
            b = wires[j]
            new_ai = a & b
            new_bj = a | b
            if np.array_equal(new_ai, a) and np.array_equal(new_bj, b):
                continue
            wires[i] = new_ai
            wires[j] = new_bj
            comps.append((i, j))
            changed = True
            break
        if not changed:
            break
        
        # Recompute unsorted
        out_masks = np.zeros(N_BOOL, dtype=np.uint32)
        for i in range(N):
            out_masks |= wires[i].astype(np.uint32) << i
        unsorted = (out_masks != targets)
    
    return comps

def _local_search(comps, targets, time_limit=5.0):
    """Local search: try to remove or swap comparators to reduce count while maintaining correctness."""
    if not comps:
        return comps
    
    start_time = time.time()
    best = list(comps)
    best_count = len(best)
    
    while time.time() - start_time < time_limit:
        # Try removing each comparator
        improved = False
        for idx in range(len(best)):
            candidate = best[:idx] + best[idx+1:]
            if len(candidate) >= best_count:
                continue
            if _verify_network_fast(candidate, targets):
                best = candidate
                best_count = len(best)
                improved = True
                break
        if improved:
            continue
        
        # Try swapping adjacent comparators
        improved = False
        for idx in range(len(best) - 1):
            candidate = list(best)
            candidate[idx], candidate[idx+1] = candidate[idx+1], candidate[idx]
            if _verify_network_fast(candidate, targets):
                best = candidate
                improved = True
                break
        if improved:
            continue
        
        # Try inserting a comparator at a random position
        if len(best) < 150:
            i = random.randint(0, N - 2)
            j = random.randint(i + 1, N - 1)
            pos = random.randint(0, len(best))
            candidate = best[:pos] + [(i, j)] + best[pos:]
            if _verify_network_fast(candidate, targets):
                best = candidate
                best_count = len(best)
                improved = True
        
        if not improved:
            # Stuck, break
            break
    
    return best

def construct():
    """Construct a sorting network for N=18 wires."""
    start_time = time.time()
    
    # Precompute targets
    targets = _build_sorted_targets()
    
    # Start with odd-even transposition
    comps = _odd_even_transposition()
    
    # Verify
    if not _verify_network_fast(comps, targets):
        # Should not happen, but just in case
        return comps
    
    # Local search to reduce comparator count
    remaining_time = DEADLINE - (time.time() - start_time)
    if remaining_time > 5:
        comps = _local_search(comps, targets, time_limit=min(remaining_time - 2, 60))
    
    # Final verification
    if time.time() - start_time < DEADLINE:
        if not _verify_network_fast(comps, targets):
            # Fallback to known good
            comps = _odd_even_transposition()
    
    return comps
