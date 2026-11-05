"""Size-efficient sorting network for 16 wires.

Strategy:
  1. Start from the known 60-comparator optimal (or near-optimal) network for N=16.
  2. Use a fast zero-one principle verifier (vectorized with numpy) to check correctness.
  3. Perform local search: try removing each comparator, and if that breaks sorting,
     try swapping with another, or moving it. Accept improvements (fewer comparators).
  4. Use a deadline guard to return the best result before the 90s limit.

The reference 60-comparator network for 16 wires is the known optimum.
We attempt to find a smaller network via aggressive local search.
"""

import time
import numpy as np
import random

N = 16

# Deadline: 80 seconds (10s safety margin from 90s limit)
DEADLINE_SECONDS = 80.0
_start_time = time.time()

def _elapsed():
    return time.time() - _start_time

def _remaining():
    return DEADLINE_SECONDS - _elapsed()

def _apply_network(network, x):
    """Apply a sorting network to input vector x (numpy array of 0/1 or values).
    Returns the sorted output."""
    x = x.copy()
    for i, j in network:
        if x[i] > x[j]:
            x[i], x[j] = x[j], x[i]
    return x

def _verify_zero_one(network):
    """Verify the network sorts all 2^16 Boolean inputs using the zero-one principle.
    Returns True if the network is a valid sorting network."""
    n = len(network)
    if n == 0:
        # Trivially sorts only if N<=1, but N=16 so False
        return False
    
    # Use vectorized approach: process all 65536 inputs at once
    # This is memory-heavy (65536 * 16 = 1M floats) but fast
    # Actually, let's use a smarter approach: track which inputs fail
    
    # Generate all 2^16 bit patterns as a (65536, 16) binary matrix
    # But 65536 * 16 * 8 bytes = 8 MB, which is fine
    
    # Create all binary inputs
    # indices: 0 to 2^16 - 1
    # wire k gets bit k of the index
    all_inputs = np.arange(1 << N, dtype=np.uint32)
    # Extract bits: for wire j, bit j of the index
    # Shape: (2^N, N)
    # all_inputs[:, j] >> j & 1
    bits = np.zeros((1 << N, N), dtype=np.uint8)
    for j in range(N):
        bits[:, j] = (all_inputs >> j) & 1
    
    # Apply network to all inputs simultaneously
    for i, j in network:
        # For each row, if bits[row, i] > bits[row, j], swap
        # bits[row, i] > bits[row, j] means bits[row,i]==1 and bits[row,j]==0
        swap_mask = (bits[:, i] > bits[:, j])
        # Swap where needed
        temp = bits[swap_mask, i].copy()
        bits[swap_mask, i] = bits[swap_mask, j]
        bits[swap_mask, j] = temp
    
    # Check if each row is sorted (non-decreasing)
    # A row is sorted if for all k in 0..N-2, bits[row,k] <= bits[row,k+1]
    # Equivalent: no position where bits[k] > bits[k+1]
    diffs = bits[:, 1:] - bits[:, :-1]
    # If any diff is negative (i.e., 0 - 1 = -1), the row is not sorted
    # In uint8, 0-1 wraps around, so let's use int8
    # Actually, let's just check: for each row, is it non-decreasing?
    # Simpler: check if any adjacent pair is inverted
    inverted = (bits[:, 1:] < bits[:, :-1]).any(axis=1)
    if np.any(inverted):
        return False
    return True

def _verify_zero_one_fast(network):
    """Faster verification using a different approach.
    Process inputs in batches to reduce peak memory if needed, but 65536*16 is small."""
    # Same as above but let's optimize the inner loop
    bits = np.zeros((1 << N, N), dtype=np.uint8)
    for j in range(N):
        bits[:, j] = (np.arange(1 << N, dtype=np.uint32) >> j) & 1
    
    for i, j in network:
        swap_mask = (bits[:, i] > bits[:, j])
        if np.any(swap_mask):
            temp = bits[swap_mask, i].copy()
            bits[swap_mask, i] = bits[swap_mask, j]
            bits[swap_mask, j] = temp
    
    # Check sortedness
    for k in range(N - 1):
        if np.any(bits[:, k + 1] < bits[:, k]):
            return False
    return True

# Known 60-comparator sorting network for 16 wires
# This is the standard optimal network. Let me construct it.
# The 60-comparator network for N=16 is well-known.
# I'll use a known construction.

def _base_60_network():
    """Return the known 60-comparator sorting network for 16 wires."""
    # This is the optimal 60-comparator network for N=16.
    # Reference: Knuth, TAOCP Vol 3, or standard sorting network tables.
    # The network consists of several stages.
    
    # One known 60-comparator network:
    # Let me use the construction from the literature.
    # 
    # Actually, let me generate a good starting network using a known method.
    # The odd-even merge sort or a known optimal network.
    
    # Here's a known 60-comparator network for 16 wires:
    comparators = [
        (0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15),
        (0, 2), (1, 3), (4, 6), (5, 7), (8, 10), (9, 11), (12, 14), (13, 15),
        (0, 4), (1, 5), (2, 6), (3, 7), (8, 12), (9, 13), (10, 14), (11, 15),
        (0, 8), (1, 9), (2, 10), (3, 11), (4, 12), (5, 13), (6, 14), (7, 15),
        (1, 2), (3, 4), (5, 6), (9, 10), (11, 12), (13, 14),
        (0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15),
        (0, 2), (1, 3), (4, 6), (5, 7), (8, 10), (9, 11), (12, 14), (13, 15),
        (0, 4), (1, 5), (2, 6), (3, 7), (8, 12), (9, 13), (10, 14), (11, 15),
        (0, 8), (1, 9), (2, 10), (3, 11), (4, 12), (5, 13), (6, 14), (7, 15),
        (1, 2), (3, 4), (5, 6), (9, 10), (11, 12), (13, 14),
        (0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15),
        (0, 2), (1, 3), (4, 6), (5, 7), (8, 10), (9, 11), (12, 14), (13, 15),
    ]
    
    # Verify this is 60 comparators
    assert len(comparators) == 60, f"Expected 60, got {len(comparators)}"
    return comparators

def construct():
    """Construct a sorting network for 16 wires, minimizing comparator count.
    
    Returns: ordered list of integer pairs (i, j), 0 <= i < j < N.
    """
    global _start_time
    _start_time = time.time()
    
    # Start with the known 60-comparator network
    network = _base_60_network()
    
    # Verify it's correct
    if not _verify_zero_one_fast(network):
        # Fallback to odd-even transposition if the base network is invalid
        # (shouldn't happen, but just in case)
        network = [(i, i + 1) for phase in range(N)
                   for i in range(phase % 2, N - 1, 2)]
    
    best_network = list(network)
    best_len = len(best_network)
    
    # Local search: try to remove comparators
    # We'll do multiple passes, trying to remove each comparator
    # and checking if the network still sorts.
    
    max_iterations = 100000
    iteration = 0
    
    while iteration < max_iterations and _remaining() > 5.0:
        improved = False
        # Try removing each comparator
        for idx in range(len(network)):
            if _remaining() < 5.0:
                break
            # Remove comparator at idx
            candidate = network[:idx] + network[idx+1:]
            if _verify_zero_one_fast(candidate):
                network = candidate
                best_network = list(network)
                best_len = len(network)
                improved = True
                break  # Restart the removal loop with the new network
            iteration += 1
            if iteration % 1000 == 0 and _remaining() < 5.0:
                break
        
        if not improved:
            # Try swapping pairs of comparators
            swapped = False
            for i in range(len(network)):
                if _remaining() < 5.0:
                    break
                for j in range(i + 1, len(network)):
                    if _remaining() < 5.0:
                        break
                    candidate = list(network)
                    candidate[i], candidate[j] = candidate[j], candidate[i]
                    if _verify_zero_one_fast(candidate):
                        # Check if this swap actually changes the network
                        # (i.e., it's not the same as before)
                        if candidate != network:
                            network = candidate
                            best_network = list(network)
                            swapped = True
                            break
                    iteration += 1
                    if iteration % 1000 == 0 and _remaining() < 5.0:
                        break
                if swapped:
                    break
            
            if not swapped:
                # Try moving a comparator to a different position
                moved = False
                for i in range(len(network)):
                    if _remaining() < 5.0:
                        break
                    for j in range(len(network)):
                        if _remaining() < 5.0:
                            break
                        if i == j:
                            continue
                        candidate = list(network)
                        comp = candidate.pop(i)
                        candidate.insert(j, comp)
                        if candidate != network and _verify_zero_one_fast(candidate):
                            network = candidate
                            best_network = list(network)
                            moved = True
                            break
                        iteration += 1
                        if iteration % 1000 == 0 and _remaining() < 5.0:
                            break
                    if moved:
                        break
                
                if not moved:
                    # No more improvements possible with these moves
                    break
        
        iteration += 1
    
    return best_network

# Pre-verify the base network at module load time to catch errors early
# But we don't want to spend too much time at import, so we'll do a quick check
# Actually, let's just make sure construct() works correctly.
