"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

N = 16

# EVOLVE-BLOCK-START

N = 16


def _bitonic_sort_network(n):
    """Generate Batcher's bitonic sort network for n elements (n must be power of 2).
    
    Returns a list of (i, j) comparators where i < j, meaning min goes to wire i, max to wire j.
    """
    comparators = []
    
    def _bitonic_merge(start, size, direction):
        """Merge a bitonic sequence of given size starting at 'start'.
        direction: True for ascending (min at lower index), False for descending."""
        if size <= 1:
            return
        half = size // 2
        # Compare elements half positions apart
        for i in range(half):
            a = start + i
            b = start + i + half
            if direction:
                # Ascending: min at lower index (a < b)
                comparators.append((a, b))
            else:
                # Descending: max at lower index, so we need min at b, max at a
                # Since we store (i,j) with i<j and min->i, max->j,
                # for descending we want max at a, min at b
                # So we swap: use (b, a) but that violates i<j constraint
                # Instead, we represent it as comparing a and b but putting max at a
                # This is equivalent to: if a < b, swap them (so b goes to a, a goes to b)
                # In our representation (i,j) with i<j means min->i, max->j
                # For descending, we want max->a, min->b where a<b
                # So we need to put the result such that wire a gets max and wire b gets min
                # This is achieved by: if wires[a] < wires[b], swap them
                # Which is the opposite of our standard comparator
                # We can handle this by noting that a "descending" comparator (a,b) 
                # is equivalent to: the value at a after should be >= value at b after
                # Standard comparator (a,b): after, wire[a] <= wire[b]
                # Descending comparator: after, wire[a] >= wire[b]
                # We can implement descending by swapping the roles: 
                # Actually, let's just track direction and handle it in verification
                # For now, let me use a different approach: 
                # A descending comparator between a and b (a<b) means:
                #   new_a = max(old_a, old_b), new_b = min(old_a, old_b)
                # A standard comparator (a,b) means:
                #   new_a = min(old_a, old_b), new_b = max(old_a, old_b)
                # So descending is the "inverse" operation.
                # We can represent this by swapping a and b in the tuple, but we need i<j.
                # So for descending comparator between positions a and b (a<b),
                # we store it as (a, b) but with a flag, OR we just note that 
                # in the network generation, we need to be careful.
                # 
                # Let me use a cleaner approach: generate comparators with direction info
                # and then convert to the required format.
                # Actually, the simplest way: for descending order between a and b (a<b),
                # we want the larger value at position a. This is NOT achievable with 
                # a single (i,j) comparator where i<j and min->i, max->j.
                # 
                # Wait - I think I'm overcomplicating this. Let me reconsider.
                # In a bitonic sort network, we have two types of comparators:
                # 1. "Up" comparators: min goes to lower index, max to higher index
                # 2. "Down" comparators: max goes to lower index, min to higher index
                # 
                # The problem states: "Each compare-exchange writes min to wire i and max to j"
                # where i < j. So we can ONLY do "up" comparators.
                # 
                # This means we CANNOT directly implement a standard bitonic sort network
                # which requires both up and down comparators!
                # 
                # We need to find a different approach. Let me use a network that only
                # uses "up" comparators (min to lower index, max to higher index).
                # 
                # One option: use a merge sort network or odd-even merge network,
                # which can be implemented with only "up" comparators.
                # 
                # Actually wait - let me re-read the problem. "compare-exchange writes min to wire i and max to j"
                # This is a fixed operation. So yes, we can only sort in ascending order using
                # these comparators. A bitonic sort network with descending comparators won't work.
                # 
                # Let me use Batcher's odd-even merge sort instead, which only needs
                # ascending comparators.
                pass
        
        # Recursively merge the two halves
        _bitonic_merge(start, half, True)
        _bitonic_merge(start + half, half, False)
    
    # This approach has issues with descending comparators. Let me use a different method.
    # I'll implement Batcher's odd-even merge sort network which uses only ascending comparators.
    return comparators


def _odd_even_merge_network(n):
    """Generate Batcher's odd-even merge network for merging two sorted sequences of length n.
    Both sequences must be sorted in ascending order.
    Returns comparators for the 2n wires, where first n wires contain the first sequence
    and last n wires contain the second sequence.
    Only uses ascending comparators (min to lower index, max to higher index).
    """
    if n <= 1:
        return []
    
    if n == 2:
        # Merge two sorted pairs: positions 0,1 (sorted) and 2,3 (sorted)
        # Compare (0,2) and (1,3), then fix with (1,2)
        return [(0, 2), (1, 3), (1, 2)]
    
    half = n // 2
    comparators = []
    
    # Step 1: Odd-even merge the odd and even indexed elements separately
    # Odd indexed elements: positions 1, 3, 5, ..., 2*half-1 in first half
    #                      and positions n+1, n+3, ..., n+2*half-1 in second half
    # Even indexed elements: positions 0, 2, 4, ..., 2*half-2 in first half
    #                       and positions n, n+2, ..., n+2*half-2 in second half
    
    # Merge odd-indexed elements (they form two sorted sequences of length half)
    # First sequence: positions 1, 3, 5, ..., 2*half-1
    # Second sequence: positions n+1, n+3, ..., n+2*half-1
    # We need to map these to a contiguous range for the recursive merge
    odd_comps = _odd_even_merge_network(half)
    for a, b in odd_comps:
        # Map index a in the "odd subsequence" to actual position
        # Odd positions in first half: 1, 3, 5, ... -> index a maps to 2*a+1
        # Odd positions in second half: n+1, n+3, ... -> index a+half maps to n+2*(a+half)+1... 
        # Wait, let me think about this more carefully.
        # The odd subsequence from first half: positions 1, 3, 5, ..., 2*half-1 (half elements)
        # The odd subsequence from second half: positions n+1, n+3, ..., n+2*half-1 (half elements)
        # In the recursive merge, we treat these as two sequences of length half
        # Position a in the combined odd sequence (0-indexed):
        #   if a < half: actual position = 2*a + 1
        #   if a >= half: actual position = n + 2*(a - half) + 1
        actual_a = 2 * a + 1 if a < half else n + 2 * (a - half) + 1
        actual_b = 2 * b + 1 if b < half else n + 2 * (b - half) + 1
        if actual_a < actual_b:
            comparators.append((actual_a, actual_b))
        else:
            comparators.append((actual_b, actual_a))
    
    # Merge even-indexed elements
    even_comps = _odd_even_merge_network(half)
    for a, b in even_comps:
        # Even positions in first half: 0, 2, 4, ..., 2*half-2
        # Even positions in second half: n, n+2, ..., n+2*half-2
        actual_a = 2 * a if a < half else n + 2 * (a - half)
        actual_b = 2 * b if b < half else n + 2 * (b - half)
        if actual_a < actual_b:
            comparators.append((actual_a, actual_b))
        else:
            comparators.append((actual_b, actual_a))
    
    # Step 2: Compare adjacent pairs
    for i in range(0, 2 * n - 1, 2):
        comparators.append((i, i + 1))
    
    return comparators


def _batcher_oes_network(n):
    """Generate Batcher's odd-even merge sort network for n elements.
    n must be a power of 2.
    Only uses ascending comparators.
    """
    if n <= 1:
        return []
    if n == 2:
        return [(0, 1)]
    
    half = n // 2
    comparators = []
    
    # Sort the two halves recursively
    left_comps = _batcher_oes_network(half)
    comparators.extend(left_comps)
    
    right_comps = _batcher_oes_network(half)
    for a, b in right_comps:
        comparators.append((a + half, b + half))
    
    # Odd-even merge the two sorted halves
    merge_comps = _odd_even_merge_network(half)
    comparators.extend(merge_comps)
    
    return comparators


def _verify_network_vectorized(comparators, n=16):
    """Verify network using zero-one principle with vectorized NumPy operations."""
    num_inputs = 1 << n
    # Create all possible input vectors as a (num_inputs, n) array
    # For each bit position, the bit is set for half the inputs
    inputs = np.zeros((num_inputs, n), dtype=np.uint8)
    for i in range(n):
        # Bit i is set for inputs where (mask >> i) & 1 == 1
        # This creates a pattern: 0 for 2^i inputs, then 1 for 2^i inputs, repeating
        period = 1 << (i + 1)
        half_period = 1 << i
        # Create the pattern
        pattern = np.zeros(period, dtype=np.uint8)
        pattern[half_period:] = 1
        # Tile to fill num_inputs
        num_tiles = num_inputs // period
        inputs[:, i] = np.tile(pattern, num_tiles)
    
    # Apply each comparator
    for i, j in comparators:
        # For each input, if wires[i] > wires[j], swap them
        # Since values are 0 or 1, swap happens when wires[i]=1 and wires[j]=0
        swap_mask = (inputs[:, i] == 1) & (inputs[:, j] == 0)
        inputs[swap_mask, i] = 0
        inputs[swap_mask, j] = 1
    
    # Check if all outputs are sorted (non-decreasing)
    # For each adjacent pair, check if wires[k] > wires[k+1] for any k
    for k in range(n - 1):
        if np.any(inputs[:, k] > inputs[:, k + 1]):
            return False
    
    return True


def _local_search(comparators, max_time=60.0):
    """Local search to reduce comparator count by trying deletions."""
    start_time = time.time()
    best = list(comparators)
    best_len = len(best)
    
    random.seed(42)
    
    while time.time() - start_time < max_time - 5.0:
        # Try removing a random comparator
        idx = random.randint(0, len(best) - 1)
        candidate = best[:idx] + best[idx + 1:]
        if _verify_network_vectorized(candidate):
            best = candidate
            best_len -= 1
            print(f"  Reduced to {best_len} comparators", flush=True)
        
        # Try swapping two comparators (reordering)
        if len(best) >= 2:
            i, j = random.sample(range(len(best)), 2)
            candidate = best[:]
            candidate[i], candidate[j] = candidate[j], candidate[i]
            if _verify_network_vectorized(candidate):
                best = candidate
    
    return best


def construct():
    """Construct a sorting network for 16 wires."""
    start_time = time.time()
    deadline = 80.0  # safety margin before 90s kill
    
    # Start with Batcher's odd-even merge sort network
    base = _batcher_oes_network(N)
    print(f"Initial network: {len(base)} comparators", flush=True)
    
    # Verify it's correct
    if not _verify_network_vectorized(base):
        # Fallback to odd-even transposition
        base = [(i, i + 1) for phase in range(N)
                for i in range(phase % 2, N - 1, 2)]
        print(f"Fallback network: {len(base)} comparators", flush=True)
    
    # Local search to improve
    if time.time() - start_time < deadline - 10.0:
        improved = _local_search(base, max_time=deadline - time.time() - 5.0)
        if len(improved) < len(base):
            base = improved
    
    print(f"Final network: {len(base)} comparators", flush=True)
    return base

# EVOLVE-BLOCK-END
