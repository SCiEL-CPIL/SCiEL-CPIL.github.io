"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from itertools import product

N = 16

# EVOLVE-BLOCK-START

def _simulate_network(comparators, inputs):
    """
    Simulate a compare-exchange network on multiple boolean inputs.
    inputs: shape (M, N) boolean array
    comparators: list of (i, j)
    Returns sorted outputs (M, N)
    """
    state = inputs.copy()
    for i, j in comparators:
        # min goes to i, max goes to j
        new_i = np.minimum(state[:, i], state[:, j])
        new_j = np.maximum(state[:, i], state[:, j])
        state[:, i] = new_i
        state[:, j] = new_j
    return state

def _is_sorted_network(comparators):
    """
    Check if the network sorts all 2^16 boolean inputs using the zero-one principle.
    Uses numpy vectorization for speed.
    """
    # Generate all 2^16 boolean inputs
    # This is 65536 x 16
    # We can generate this efficiently
    # Actually, for 16 wires, 2^16 = 65536 inputs.
    # We can process in chunks to save memory if needed, but 65536*16 is small.
    
    # Generate all combinations
    # np.arange(2**N) then unpack bits
    # But 2**16 is 65536, which is fine.
    
    # Efficient generation of all boolean vectors
    # x = np.arange(2**N, dtype=np.uint16).reshape(-1, 1)
    # bits = (x >> np.arange(N, 0, -1)) & 1
    # This gives shape (65536, 16)
    
    # Let's use a more memory efficient approach if possible, but 65536*16 bytes is ~1MB, very small.
    
    x = np.arange(2**N, dtype=np.uint16).reshape(-1, 1)
    bits = (x >> np.arange(N-1, -1, -1)) & 1  # shape (65536, 16)
    # Ensure bits are boolean
    bits = bits.astype(bool)
    
    # Simulate
    output = _simulate_network(comparators, bits)
    
    # Check if each row is sorted (non-decreasing)
    # sorted means output[:, i] <= output[:, i+1] for all i
    # Since they are boolean, this means no 1 followed by 0
    # i.e., for each row, once we see a 0, all subsequent must be 0? No.
    # Non-decreasing: 0,0,0,1,1,1
    # So for each row, if output[i] is 0, all previous must be 0? No.
    # If output[i] is 0, then output[j] for j>i must be 0? No, non-decreasing means 0...0 1...1
    # So if output[i] is 1, then output[j] for j>i must be 1.
    # Equivalently, if output[i] is 0, then output[j] for j<i must be 0.
    # Simplest: check that output[:, :-1] <= output[:, 1:]
    
    diffs = output[:, 1:] - output[:, :-1]
    # If any diff is negative, it's not sorted.
    # Since values are 0 or 1, diff is -1, 0, or 1.
    # Not sorted if any diff < 0.
    if np.any(diffs < 0):
        return False
    return True

def _odd_even_transposition():
    """Generate the odd-even transposition sorting network."""
    comparators = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            comparators.append((i, i + 1))
    return comparators

def _try_improve(comparators):
    """
    Try to remove redundant comparators from the network.
    Returns the improved list of comparators.
    """
    # Start with the given comparators
    current = list(comparators)
    
    # Try removing each comparator one by one
    improved = True
    while improved:
        improved = False
        for idx in range(len(current)):
            # Try removing comparator at idx
            candidate = current[:idx] + current[idx+1:]
            if _is_sorted_network(candidate):
                current = candidate
                improved = True
                break  # Restart the loop since the list changed
    
    return current

def construct():
    # Start with the odd-even transposition network
    comparators = _odd_even_transposition()
    
    # Try to improve by removing redundant comparators
    # This might be slow if done naively, but let's try.
    # With 120 comparators, checking 120 * 65536 * 16 operations per removal attempt...
    # 120 * 1M ops = 120M ops, which might be slow in Python but numpy helps.
    # Let's see if we can do better.
    
    # Actually, the zero-one principle check is vectorized, so it should be fast.
    # Let's try the improvement.
    
    # To speed up, we can try removing from the end first, or in some order.
    # Let's just do the iterative removal.
    
    # We need to be careful about time. 60s limit.
    # Let's add a timeout check or just hope it's fast enough.
    
    # Alternative: use a greedy approach where we check if a comparator is redundant
    # by seeing if removing it still sorts all inputs.
    
    # Let's try a more efficient approach: 
    # For each comparator, check if it's redundant.
    # We can do this in parallel? No, numpy is single-threaded by default.
    
    # Let's just do the iterative removal with early stopping.
    
    # To make it faster, we can precompute some things or use a smarter order.
    # But for now, let's try the simple approach.
    
    # Actually, let's try a different strategy: 
    # Start from the bottom (last comparators) and work backwards, 
    # since later comparators are more likely to be redundant.
    
    current = list(comparators)
    improved = True
    while improved:
        improved = False
        # Try removing from the end first
        for idx in range(len(current) - 1, -1, -1):
            candidate = current[:idx] + current[idx+1:]
            if _is_sorted_network(candidate):
                current = candidate
                improved = True
                break
    
    return current

# EVOLVE-BLOCK-END
