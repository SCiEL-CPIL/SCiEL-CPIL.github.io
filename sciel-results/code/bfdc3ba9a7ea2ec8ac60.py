"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

N = 16
MAX_COMPARATORS = 4096
SEED_DEADLINE = 45.0  # safety margin before 60s kill

# Pre-compute the input matrix for verification to avoid regeneration overhead
_N_INPUTS = 1 << N
_INDICES = np.arange(_N_INPUTS, dtype=np.uint32)
_INPUT_MATRIX = np.zeros((_N_INPUTS, N), dtype=np.uint8)
for _i in range(N):
    _INPUT_MATRIX[:, _i] = (_INDICES >> _i) & 1

def _apply_network(network, x):
    """Apply the compare-exchange network to input vector x (numpy array)."""
    for i, j in network:
        if x[i] > x[j]:
            x[i], x[j] = x[j], x[i]
    return x

def _verify_network_fastest(network):
    """
    Fast verification using pre-computed input matrix.
    """
    # Copy the input matrix
    x = _INPUT_MATRIX.copy()
    
    # Apply network
    for i, j in network:
        mask = x[:, i] > x[:, j]
        if np.any(mask):
            x[mask, i] = 0
            x[mask, j] = 1
    
    # Verify sorted
    for i in range(N - 1):
        if np.any(x[:, i] > x[:, i + 1]):
            return False
    return True

def _count_comparators(network):
    return len(network)

def _compute_depth(network):
    """Compute the parallel depth of the network."""
    if not network:
        return 0
    
    n = len(network)
    wire_to_last_comp = {}  # wire -> last comparator index that touched it
    depths = [0] * n
    
    for idx, (i, j) in enumerate(network):
        max_dep = 0
        for wire in (i, j):
            if wire in wire_to_last_comp:
                prev_idx = wire_to_last_comp[wire]
                if depths[prev_idx] + 1 > max_dep:
                    max_dep = depths[prev_idx] + 1
        depths[idx] = max_dep
        wire_to_last_comp[i] = idx
        wire_to_last_comp[j] = idx
    
    return max(depths) if depths else 0

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a sorting network for 16 wires using Batcher's Odd-Even Merge Sort.
    """
    start_time = time.time()
    deadline = start_time + SEED_DEADLINE
    
    # Hardcoded optimal networks for small N
    def optimal_sort_1():
        return []
    
    def optimal_sort_2():
        return [(0, 1)]
    
    def optimal_sort_3():
        return [(0, 1), (1, 2), (0, 1)]
    
    def optimal_sort_4():
        # 5 comparators
        return [(0, 1), (2, 3), (0, 2), (1, 3), (1, 2)]
        
    def optimal_sort_5():
        # 9 comparators
        return [(0, 1), (2, 3), (3, 4), (0, 2), (1, 3), (0, 3), (1, 4), (2, 4), (1, 2)]

    def batcher_merge(k, offset, step, network):
        """
        Recursively merge two sorted lists of size k.
        The lists are interleaved in the array starting at 'offset' with 'step'.
        Logical indices 0..k-1 map to physical offset + i*step.
        Logical indices k..2k-1 map to physical offset + (i+k)*step.
        """
        if k == 1:
            # Compare the two elements
            i1 = offset
            i2 = offset + step
            network.append((i1, i2))
            return
        
        # First pass: compare (i, i+k)
        for i in range(k):
            i1 = offset + i * step
            i2 = offset + (i + k) * step
            network.append((i1, i2))
        
        # Recursive merges on even and odd logical indices
        # Even logical indices: 0, 2, 4...
        # Physical indices: offset, offset + 2*step, ...
        batcher_merge(k // 2, offset, 2 * step, network)
        
        # Odd logical indices: 1, 3, 5...
        # Physical indices: offset + step, offset + 3*step, ...
        batcher_merge(k // 2, offset + step, 2 * step, network)

    def batcher_sort(n, offset, step, network):
        """
        Recursively sort n elements.
        """
        if n == 1:
            return
        if n == 2:
            network.append((offset, offset + step))
            return
        if n == 3:
            network.append((offset, offset + step))
            network.append((offset + step, offset + 2 * step))
            network.append((offset, offset + step))
            return
        if n == 4:
            # Use optimal 4-wire network
            # Map logical 0,1,2,3 to physical offset, offset+step, ...
            comps = optimal_sort_4()
            for i, j in comps:
                network.append((offset + i * step, offset + j * step))
            return
        
        # Split in half
        half = n // 2
        batcher_sort(half, offset, step, network)
        batcher_sort(half, offset + half * step, step, network)
        
        # Merge the two halves
        batcher_merge(half, offset, step, network)

    # Generate the network for N=16
    network = []
    batcher_sort(N, 0, 1, network)
    
    best_network = network
    best_count = len(best_network)
    
    # Verify the generated network
    if not _verify_network_fastest(best_network):
        # Fallback to odd-even transposition if Batcher fails (should not happen)
        best_network = [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
        best_count = len(best_network)
    
    # Attempt greedy removal to reduce count further
    current = list(best_network)
    changed = True
    while changed:
        changed = False
        for idx in range(len(current) - 1, -1, -1):
            if time.time() > deadline - 5:
                break
            candidate = current[:idx] + current[idx+1:]
            if _verify_network_fastest(candidate):
                current = candidate
                changed = True
        if time.time() > deadline - 5:
            break
    
    best_network = current
    best_count = len(best_network)
    
    return best_network
# EVOLVE-BLOCK-END
