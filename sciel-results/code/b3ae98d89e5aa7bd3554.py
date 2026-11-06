"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

N = 16

# EVOLVE-BLOCK-START
def _generate_batcher_network():
    """
    Generates Batcher's Odd-Even Merge Sort network for N wires.
    Returns a list of (i, j) comparators.
    """
    def merge(n, offset):
        if n == 1:
            return [(offset, offset + 1)]
        
        h = n // 2
        # Step 1: compare corresponding elements
        step1 = [(offset + i, offset + i + n) for i in range(n)]
        
        def strided_merge(n, offset, stride):
            if n == 1:
                return [(offset, offset + stride)]
            
            h = n // 2
            step1 = [(offset + i * stride, offset + (n + i) * stride) for i in range(n)]
            even = strided_merge(h, offset, 2 * stride)
            odd = strided_merge(h, offset + stride, 2 * stride)
            return step1 + even + odd
        
        even_merge = strided_merge(h, offset, 2)
        odd_merge = strided_merge(h, offset + 1, 2)
        
        return step1 + even_merge + odd_merge

    def sort(n, offset):
        if n <= 1:
            return []
        h = n // 2
        left = sort(h, offset)
        right = sort(h, offset + h)
        m = merge(h, offset)
        return left + right + m

    return sort(N, 0)

def _apply_network_vectorized(comps, inputs):
    """
    Apply the network to a batch of inputs (numpy array of shape (batch_size, N)).
    """
    vec = inputs.copy()
    for i, j in comps:
        min_val = np.minimum(vec[:, i], vec[:, j])
        max_val = np.maximum(vec[:, i], vec[:, j])
        vec[:, i] = min_val
        vec[:, j] = max_val
    return vec

def _is_sorted_network_vectorized(comps, bits):
    """
    Check if the network sorts all boolean inputs in 'bits'.
    'bits' is a numpy array of shape (2^N, N).
    """
    out = _apply_network_vectorized(comps, bits)
    # Check if each row is sorted (non-decreasing)
    diffs = np.diff(out, axis=1)
    return np.all(diffs >= 0)

def _verify_network(comparators, n=N):
    """Verify network sorts all 2^n Boolean inputs using zero-one principle."""
    for mask in range(1 << n):
        wires = [(mask >> i) & 1 for i in range(n)]
        for i, j in comparators:
            if wires[i] > wires[j]:
                wires[i], wires[j] = wires[j], wires[i]
        if wires != sorted(wires):
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Leave 5s safety margin for 60s limit
    
    # Pre-compute all boolean inputs for vectorized checking
    n_inputs = 1 << N
    indices = np.arange(n_inputs)
    bits = (indices[:, None] >> np.arange(N)[::-1]) & 1
    
    # Generate initial network (Batcher's, 63 comparators for N=16)
    current_comps = _generate_batcher_network()
    
    # Verify initial network (sanity check)
    if not _is_sorted_network_vectorized(current_comps, bits):
        # Fallback to a simple valid network if Batcher generation failed (unlikely)
        # Odd-even transposition network
        current_comps = []
        for i in range(N):
            if i % 2 == 0:
                for j in range(0, N - 1, 2):
                    current_comps.append((j, j + 1))
            else:
                for j in range(1, N - 1, 2):
                    current_comps.append((j, j + 1))
        if not _is_sorted_network_vectorized(current_comps, bits):
            return current_comps # Return whatever we have, though it should be valid

    best_comps = list(current_comps)
    
    # Local search: try to remove redundant comparators
    # We iterate through the list and try to remove each one.
    # If removal is valid, we keep it removed.
    # We repeat until no more removals are possible or time runs out.
    
    while time.time() < deadline:
        removed_any = False
        # Iterate backwards to avoid index shifting issues during removal
        for idx in range(len(current_comps) - 1, -1, -1):
            if time.time() > deadline:
                break
            
            # Try removing comparator at idx
            candidate = current_comps[:idx] + current_comps[idx+1:]
            
            # Check validity
            if _is_sorted_network_vectorized(candidate, bits):
                current_comps = candidate
                removed_any = True
        
        if not removed_any:
            break
            
    return current_comps
# EVOLVE-BLOCK-END
