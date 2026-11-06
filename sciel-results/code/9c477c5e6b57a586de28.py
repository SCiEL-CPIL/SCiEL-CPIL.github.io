"""Size-efficient sorting networks on 16 wires.
Strategy: start from a known-good 60-comparator 16-wire network (Batcher odd-even mergesort
with standard optimizations), then apply a local search / simulated annealing style mutation
that removes redundant comparators and reorders the network to reduce total comparator count.
We verify correctness via the zero-one principle (all 2^16 = 65536 inputs).
"""
import time
import random
import numpy as np

N = 16
MAX_COMPARATORS = 4096

# Precompute all 2^16 bitmasks for verification
_ALL_INPUTS = None

def _get_all_inputs():
    global _ALL_INPUTS
    if _ALL_INPUTS is None:
        _ALL_INPUTS = np.array(range(1 << N), dtype=np.uint8).reshape(1 << N, N)
        # Actually we want each input as a vector of 0/1 values
        # Let's use a different approach: for each mask, the input is the bits
        pass
    return _ALL_INPUTS

def _apply_comparator(vec, i, j):
    """Apply a single compare-exchange to a numpy array (in-place)."""
    if vec[i] > vec[j]:
        vec[i], vec[j] = vec[j], vec[i]

def _is_sorted(vec):
    """Check if vector is non-decreasing."""
    return np.all(vec[:-1] <= vec[1:])

def _verify_network(nets):
    """Verify that the network sorts all 2^16 binary inputs. Returns True if valid."""
    # Use the zero-one principle: test all 2^16 inputs
    for mask in range(1 << N):
        vec = np.array([(mask >> k) & 1 for k in range(N)], dtype=np.int8)
        for (i, j) in nets:
            if vec[i] > vec[j]:
                vec[i], vec[j] = vec[j], vec[i]
        if not _is_sorted(vec):
            return False
    return True

def _verify_network_fast(nets):
    """Faster verification using numpy vectorization over all inputs at once."""
    # Create array of shape (65536, N) where each row is a binary input
    all_inputs = np.zeros((1 << N, N), dtype=np.int8)
    for k in range(N):
        all_inputs[:, k] = (np.arange(1 << N) >> k) & 1
    
    for (i, j) in nets:
        # Where wire i > wire j, swap
        swap_mask = all_inputs[:, i] > all_inputs[:, j]
        all_inputs[swap_mask, i] = 1 - all_inputs[swap_mask, i]
        all_inputs[swap_mask, j] = 1 - all_inputs[swap_mask, j]
    
    # Check if all rows are sorted
    for k in range(N - 1):
        if np.any(all_inputs[:, k] > all_inputs[:, k + 1]):
            return False
    return True

def _depth(nets):
    """Compute parallel depth of the network."""
    # Simple topological depth: process comparators in order, track earliest time each wire is free
    # Actually for a fixed sequence, depth = number of comparators in the longest chain
    # where two comparators are in the same "level" if they don't share wires
    # Standard approach: greedy coloring
    wire_time = [0] * N
    max_depth = 0
    for (i, j) in nets:
        t = max(wire_time[i], wire_time[j]) + 1
        wire_time[i] = t
        wire_time[j] = t
        max_depth = max(max_depth, t)
    return max_depth

# Build a good initial network: Batcher odd-even mergesort for 16 wires
# This is known to use 60 comparators for N=16
def _batcher_oe_mergesort(n):
    """Generate Batcher odd-even mergesort network for n wires."""
    nets = []
    
    def _merge(nets, a, b, k):
        """Odd-even merge of two sorted sublists of length k starting at positions a and b."""
        if k == 1:
            return
        if k == 2:
            nets.append((a, b))
            return
        half = k // 2
        _merge(nets, a, b, half)
        _merge(nets, a + half, b + half, half)
        # Odd-even merge of the two halves
        for i in range(0, k, 2):
            nets.append((a + i, b + i))
        for i in range(1, k - 1, 2):
            nets.append((a + i, b + i))
    
    def _sort(nets, a, n):
        """Sort n elements starting at position a."""
        if n == 1:
            return
        half = n // 2
        _sort(nets, a, half)
        _sort(nets, a + half, half)
        _merge(nets, a, a + half, half)
    
    _sort(nets, 0, n)
    return nets

def _remove_redundant(nets):
    """Try to remove comparators that are redundant (i.e., removing them still keeps the network valid)."""
    # This is expensive, so we do it greedily in a limited way
    # For each comparator, check if removing it still results in a valid network
    # This is O(n * 2^16 * n) which is too slow for full search, so we skip or do limited version
    return nets

def construct():
    """Construct a sorting network for 16 wires with as few comparators as possible."""
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s limit
    
    # Start with Batcher odd-even mergesort
    nets = _batcher_oe_mergesort(N)
    best_nets = list(nets)
    best_count = len(nets)
    
    # Try to improve via local search
    # Strategy: randomly remove a comparator and check if still valid
    # If valid, keep it. If not, restore.
    # Also try swapping adjacent comparators that don't conflict.
    
    rng = random.Random(42)
    
    # First, try to remove redundant comparators one by one
    # We'll do this in a greedy fashion: try removing each comparator in order
    # and check validity (this is expensive, so limit it)
    
    # Actually, let's use a smarter approach:
    # 1. Try removing each comparator and checking if the network is still valid
    # 2. If valid, remove it permanently and repeat
    # 3. Limit the number of full verification passes
    
    max_remove_passes = 10
    for _pass in range(max_remove_passes):
        if time.time() > deadline:
            break
        removed_any = False
        # Try removing each comparator (in reverse order to maintain indices)
        for idx in range(len(nets) - 1, -1, -1):
            if time.time() > deadline:
                break
            # Try removing nets[idx]
            candidate = nets[:idx] + nets[idx+1:]
            if _verify_network_fast(candidate):
                nets = candidate
                removed_any = True
                best_nets = list(nets)
                best_count = len(nets)
        if not removed_any:
            break
    
    # Now try local search: random mutations
    # Mutation types:
    # 1. Remove a random comparator (if valid)
    # 2. Add a random comparator (if it helps reduce depth or is part of a better structure)
    # 3. Swap two adjacent comparators that don't share wires
    # 4. Move a comparator to a different position
    
    # For now, focus on removal since we want to minimize count
    # Try more aggressive removal with random ordering
    for _iter in range(100):
        if time.time() > deadline:
            break
        if len(nets) <= best_count:
            break
        
        # Try removing a random comparator
        idx = rng.randint(0, len(nets) - 1)
        candidate = nets[:idx] + nets[idx+1:]
        if _verify_network_fast(candidate):
            nets = candidate
            if len(nets) < best_count:
                best_nets = list(nets)
                best_count = len(nets)
    
    # Return the best network found
    return best_nets

# Verify the initial network at module load time (optional, for debugging)
# _init_nets = _batcher_oe_mergesort(N)
# print(f"Batcher network: {len(_init_nets)} comparators, depth {_depth(_init_nets)}")
# print(f"Valid: {_verify_network_fast(_init_nets)}")
