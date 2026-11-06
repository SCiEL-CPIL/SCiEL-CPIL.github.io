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
    Construct a sorting network for 16 wires.
    Uses Batcher's bitonic sort construction (72 comparators) as a starting point,
    then attempts to remove redundant comparators to reach the optimal 60.
    """
    start_time = time.time()
    deadline = start_time + SEED_DEADLINE
    
    def bitonic_sort_network(n):
        """
        Generate Batcher's bitonic sort network for n wires (n must be power of 2).
        """
        def bitonic_merge(n, offset=0, ascending=True):
            if n == 1:
                return []
            half = n // 2
            comparators = []
            for i in range(half):
                a = i + offset
                b = i + half + offset
                if not ascending:
                    a, b = b, a
                comparators.append((a, b))
            comparators.extend(bitonic_merge(half, offset, ascending))
            comparators.extend(bitonic_merge(half, half + offset, not ascending))
            return comparators
        
        def bitonic_sort(n, offset=0, ascending=True):
            if n == 1:
                return []
            half = n // 2
            comparators = []
            comparators.extend(bitonic_sort(half, offset, True))
            comparators.extend(bitonic_sort(half, half + offset, False))
            comparators.extend(bitonic_merge(n, offset, ascending))
            return comparators
        
        return bitonic_sort(n)
    
    # Start with Batcher's Bitonic Sort (72 comparators)
    current = bitonic_sort_network(N)
    
    # Verify initial network
    if not _verify_network_fastest(current):
        # Fallback to odd-even transposition if bitonic fails (shouldn't happen)
        current = [(i, i + 1) for phase in range(N)
                   for i in range(phase % 2, N - 1, 2)]
    
    # Greedy removal of redundant comparators
    changed = True
    while changed and time.time() < deadline - 2:
        changed = False
        # Iterate backwards to handle index shifting
        for idx in range(len(current) - 1, -1, -1):
            if time.time() > deadline - 2:
                break
            candidate = current[:idx] + current[idx+1:]
            if _verify_network_fastest(candidate):
                current = candidate
                changed = True
                # If we reach 60, we can stop early as it's known optimal
                if len(current) <= 60:
                    return current
    
    return current
# EVOLVE-BLOCK-END
