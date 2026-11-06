"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 17
NUM_INPUTS = 1 << N  # 131072
DEADLINE = 85  # seconds, safety margin before 90s kill

# EVOLVE-BLOCK-START

def _make_all_inputs_packed():
    """Create array of all 2^N inputs as uint64 array."""
    return np.arange(NUM_INPUTS, dtype=np.uint64)

def _apply_comparator_packed(inputs, i, j):
    """Apply comparator (i, j) to packed inputs."""
    mask_i = np.uint64(1) << i
    mask_j = np.uint64(1) << j
    bit_i = (inputs & mask_i) != 0
    bit_j = (inputs & mask_j) != 0
    new_i = bit_i & bit_j
    new_j = bit_i | bit_j
    inputs = inputs & ~(mask_i | mask_j)
    inputs = np.where(new_i, inputs | mask_i, inputs)
    inputs = np.where(new_j, inputs | mask_j, inputs)
    return inputs

def _is_sorted_packed(inputs):
    """Check if all packed inputs are sorted (non-decreasing)."""
    for k in range(N - 1):
        mask_k = np.uint64(1) << k
        mask_k1 = np.uint64(1) << (k + 1)
        bit_k = (inputs & mask_k) != 0
        bit_k1 = (inputs & mask_k1) != 0
        if np.any(bit_k & ~bit_k1):
            return False
    return True

def _simulate_network_packed(comparators, inputs):
    """Simulate full network, return final packed state."""
    for i, j in comparators:
        inputs = _apply_comparator_packed(inputs, i, j)
    return inputs

def _verify_network_packed(comparators, inputs):
    """Verify network sorts all inputs."""
    final = _simulate_network_packed(comparators, inputs)
    return _is_sorted_packed(final)

def _generate_odd_even_transposition():
    """Odd-even transposition sorting network."""
    comps = []
    for phase in range(N):
        start = phase % 2
        for i in range(start, N - 1, 2):
            comps.append((i, i + 1))
    return comps

def _generate_bitonic_network():
    """Bitonic sorting network for N wires (power of 2 padded)."""
    # Pad to next power of 2
    size = 1
    while size < N:
        size *= 2
    # Generate bitonic network for size wires
    comps = []
    def _bitonic_stage(start, length, ascending=True):
        if length == 1:
            return
        half = length // 2
        # First half ascending, second half descending (or vice versa)
        _bitonic_stage(start, half, ascending)
        _bitonic_stage(start + half, half, not ascending)
        # Merge
        for i in range(half):
            if ascending:
                comps.append((start + i, start + half + i))
            else:
                comps.append((start + half + i, start + i))
    
    # Build bitonic network for 'size' wires
    for k in range(2, size + 1, 2):
        for start in range(0, size, k):
            _bitonic_stage(start, k, True)
    
    # Filter to only comparators within N wires
    filtered = [(i, j) for i, j in comps if i < N and j < N]
    return filtered

def _generate_odd_even_merge_sort_network():
    """Odd-even merge sort based network."""
    # Use a simpler approach: build from known small networks
    # For 17 wires, use a combination of smaller sorting networks
    comps = []
    # Build using recursive odd-even merge
    def _odd_even_merge(n, offset, comps_out):
        if n <= 1:
            return
        half = n // 2
        # Sort first half and second half (ascending)
        _odd_even_merge(half, offset, comps_out)
        _odd_even_merge(n - half, offset + half, comps_out)
        # Odd-even merge
        for i in range(half):
            left = offset + i
            right = offset + half + i
            if i % 2 == 0:
                comps_out.append((left, right))
            else:
                comps_out.append((left + 1, right + 1) if left + 1 < right + 1 < offset + n else (left, right))
    
    # Simpler: just use odd-even transposition as base, it's well-known
    return _generate_odd_even_transposition()

def _remove_redundant_batched(comparators, inputs, time_budget):
    """Remove redundant comparators using batched verification."""
    start_time = time.time()
    comps = list(comparators)
    
    # Precompute prefix states for efficient batch testing
    # For each position, we want to know the state BEFORE that comparator
    # and the state AFTER all subsequent comparators
    
    changed = True
    while changed and (time.time() - start_time) < time_budget:
        changed = False
        n_comps = len(comps)
        
        # Compute state after each prefix
        # prefix_states[k] = state after applying comps[0..k-1]
        prefix_states = [None] * (n_comps + 1)
        prefix_states[0] = inputs.copy()
        for k in range(n_comps):
            prefix_states[k + 1] = _apply_comparator_packed(prefix_states[k], *comps[k])
        
        # For each comparator k, test if removing it still sorts
        # This means: state after prefix k-1, skip comp k, apply comps k+1..end
        # We can precompute suffix effects
        
        # Try removals from end to start (later comparators more likely redundant)
        for k in range(n_comps - 1, -1, -1):
            if (time.time() - start_time) > time_budget:
                break
            
            # Test removal of comparator k
            # State before k: prefix_states[k]
            # Skip comp k, apply comps[k+1:]
            state = prefix_states[k]
            valid = True
            for m in range(k + 1, n_comps):
                state = _apply_comparator_packed(state, *comps[m])
                if (time.time() - start_time) > time_budget:
                    valid = False
                    break
            
            if valid and _is_sorted_packed(state):
                comps = comps[:k] + comps[k+1:]
                changed = True
                break  # Restart since list changed
    
    return comps

def construct():
    start_time = time.time()
    inputs = _make_all_inputs_packed()
    
    # Generate multiple candidate networks
    candidates = []
    
    # 1. Odd-even transposition
    oe = _generate_odd_even_transposition()
    candidates.append(oe)
    
    # 2. Bitonic network
    try:
        bitonic = _generate_bitonic_network()
        if len(bitonic) < len(oe):
            candidates.append(bitonic)
    except:
        pass
    
    # Verify and reduce each candidate
    best_comps = None
    best_count = float('inf')
    
    remaining_time = DEADLINE - (time.time() - start_time) - 5
    
    for idx, cand in enumerate(candidates):
        if (time.time() - start_time) > DEADLINE - 5:
            break
        
        # Verify base network
        if not _verify_network_packed(cand, inputs):
            continue
        
        # Allocate time for reduction
        time_for_this = remaining_time / (len(candidates) - idx)
        
        reduced = _remove_redundant_batched(cand, inputs, time_for_this)
        
        if len(reduced) < best_count:
            best_count = len(reduced)
            best_comps = reduced
    
    # If no valid reduced network found, fall back to base
    if best_comps is None:
        best_comps = oe
    
    # Final verification
    if not _verify_network_packed(best_comps, inputs):
        # Fallback to original
        best_comps = oe
    
    return best_comps

# EVOLVE-BLOCK-END
