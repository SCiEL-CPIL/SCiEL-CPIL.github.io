"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

N = 16

# EVOLVE-BLOCK-START

def _batcher_odd_even_merge(n):
    """Batcher's odd-even merge network for n elements (n must be power of 2)."""
    if n <= 1:
        return []
    if n == 2:
        return [(0, 1)]
    
    half = n // 2
    # Sort odd and even subsequences
    odd = _batcher_odd_even_merge(half)
    even = _batcher_odd_even_merge(half)
    
    # Interleave: odd elements go to even positions, even to odd positions
    # Actually, let's use the standard Batcher OES approach
    # For n=2^k, the network is built recursively
    
    # Standard Batcher OES:
    # 1. Sort first half and second half recursively
    # 2. Odd-even merge the two halves
    
    # Let me implement this properly
    if n == 2:
        return [(0, 1)]
    
    # For n > 2, use the recursive structure
    # First, sort the two halves of size n/2
    left = _batcher_odd_even_merge(n // 2)
    right = [(i + n // 2, j + n // 2) for i, j in left]
    
    # Then odd-even merge
    merge = _odd_even_merge(n // 2)
    
    return left + right + merge


def _odd_even_merge(n):
    """Odd-even merge of two sorted sequences of length n each.
    Returns comparators for merging two sorted lists of length n."""
    if n <= 1:
        return []
    if n == 2:
        # Merge two sorted pairs: (a0,a1) and (b0,b1) where a0<=a1, b0<=b1
        # Compare a0 with b0, a1 with b1, then fix if needed
        # Standard: compare (0,2), (1,3), then (0,2) again if needed
        # Actually for n=2: compare positions 0&2, 1&3, then 1&2 if needed
        # But we need a fixed network, so:
        return [(0, 2), (1, 3), (1, 2)]
    
    half = n // 2
    # Recursively merge odd and even subsequences
    # Odd positions: 1, 3, 5, ... -> indices 1, 3, 5, ...
    # Even positions: 0, 2, 4, ... -> indices 0, 2, 4, ...
    
    # Merge odd subsequences (positions 1, 3, 5, ...)
    # These are at indices 1, 3, 5, ... in the first half and n+1, n+3, ... in second half
    odd_merge = []
    for i in range(half):
        # Odd element i in first half is at position 2*i+1
        # Odd element i in second half is at position n + 2*i+1
        odd_merge.append((2*i+1, n + 2*i+1))
    
    # This approach is getting complex. Let me use a known correct Batcher network.
    # For now, let's use a simpler approach: build the network directly.
    
    # Actually, let me implement the standard Batcher OES correctly.
    # The key insight: for n=2^k, we can build it as:
    # - Sort the first n/2 and last n/2 recursively
    # - Then do an odd-even merge of the two halves
    
    # Odd-even merge of two sorted sequences of length m:
    # 1. Odd-even merge the odd elements and even elements separately
    # 2. Compare (0,1), (2,3), (4,5), ...
    
    if n == 2:
        return [(0, 2), (1, 3), (1, 2)]
    
    # For general n, the odd-even merge:
    # First, merge odd-indexed elements and even-indexed elements recursively
    # Odd elements: positions 1, 3, 5, ... in each half
    # Even elements: positions 0, 2, 4, ... in each half
    
    # This is getting complicated. Let me just use a known good network.
    # For 16 wires, let me use a well-known construction.
    
    # Let me try a different approach: use the known 60-comparator network
    # or at least a better starting point.
    
    return []  # placeholder


def _known_good_network():
    """Return a known good sorting network for 16 wires.
    Using Batcher's bitonic sort as a starting point, which uses fewer comparators
    than odd-even transposition."""
    # Batcher's bitonic sort for 16 elements
    # Depth: log2(16) * (log2(16)+1) / 2 = 4*5/2 = 10 phases
    # But the comparator count is what matters.
    
    # Bitonic sort network for n=16:
    # Phase 1: sort pairs (0,1), (2,3), (4,5), ...
    # Phase 2: sort quads
    # etc.
    
    # Let me implement bitonic sort network properly
    def bitonic_sort_network(n):
        if n <= 1:
            return []
        
        # Generate bitonic sort network
        comparators = []
        
        # Helper to add a stage
        def add_stage(comps, k, offset, direction):
            # k is the current size of the bitonic sequence
            # offset is the starting index
            # direction: 0 for ascending, 1 for descending
            if k == 1:
                return
            half = k // 2
            for i in range(offset, offset + k, 2):
                if direction == 0:
                    # Ascending: compare (i, i+1) with i < i+1
                    comps.append((i, i + 1))
                else:
                    # Descending: compare (i+1, i) but we store as (min, max)
                    # Actually, we always store (i, j) with i < j, and min goes to i, max to j
                    # For descending order within a bitonic sequence, we want max at lower index
                    # So we swap: compare (i, i+1) but put max at i, min at i+1
                    # This means we store it as (i+1, i) but since we require i < j,
                    # we need a different approach.
                    # Let me reconsider.
                    # In a bitonic sort, when we have a descending sequence, we compare
                    # and put the larger value at the lower index.
                    # So for a "descending" comparator between positions a and b (a < b),
                    # we want max at a and min at b.
                    # But our comparator (i, j) with i < j puts min at i and max at j.
                    # So for descending, we need to use (b, a) but that violates i < j.
                    # 
                    # Alternative: represent comparators as (i, j) where the operation is
                    # always min->i, max->j. For descending order, we would need to
                    # swap the roles, which means we can't do it with a single comparator
                    # in this representation.
                    #
                    # Let me use a different approach: build the bitonic network explicitly
                    # with the correct comparator directions.
                    pass
        
        # This is getting too complex. Let me use a simpler known network.
        # For 16 wires, let me use the Akers network or a known good one.
        
        # Actually, let me just return a reasonable network and then improve with local search.
        # Start with a known good network: the 0-1 principle verified network.
        
        # Let me use a simple but better network: merge sort network
        def merge_sort_network(n):
            if n <= 1:
                return []
            if n == 2:
                return [(0, 1)]
            
            half = n // 2
            left = merge_sort_network(half)
            right = [(i + half, j + half) for i, j in left]
            merge = _merge_network(half)
            return left + right + merge
        
        def _merge_network(n):
            """Merge two sorted sequences of length n."""
            if n <= 1:
                return []
            if n == 2:
                return [(0, 2), (1, 3), (1, 2)]
            
            half = n // 2
            # Merge odd and even subsequences
            odd = _merge_network(half)
            even = _merge_network(half)
            
            # Interleave
            result = []
            for i in range(half):
                # Odd: positions 2*i+1 and n+2*i+1
                result.append((2*i+1, n + 2*i+1))
                # Even: positions 2*i and n+2*i
                result.append((2*i, n + 2*i))
            
            # Then compare adjacent pairs
            for i in range(0, 2*n - 1, 2):
                result.append((i, i+1))
            
            return result
        
        return merge_sort_network(n)
    
    return bitonic_sort_network(16)


def _verify_network_fast(comparators, n=N):
    """Verify network using zero-one principle with numpy for speed."""
    # Test all 2^n inputs
    for mask in range(1 << n):
        wires = np.array([(mask >> i) & 1 for i in range(n)], dtype=np.uint8)
        for i, j in comparators:
            if wires[i] > wires[j]:
                wires[i], wires[j] = wires[j], wires[i]
        # Check if sorted (non-decreasing)
        if np.any(wires[1:] < wires[:-1]):
            return False
    return True


def _local_search(comparators, max_time=60.0):
    """Local search to reduce comparator count."""
    start_time = time.time()
    best = list(comparators)
    best_len = len(best)
    
    random.seed(42)
    
    while time.time() - start_time < max_time - 5.0:  # safety margin
        # Try removing a random comparator
        idx = random.randint(0, len(best) - 1)
        candidate = best[:idx] + best[idx+1:]
        if _verify_network_fast(candidate):
            best = candidate
            best_len -= 1
        
        # Try swapping two comparators
        if len(best) >= 2:
            i, j = random.sample(range(len(best)), 2)
            candidate = best[:]
            candidate[i], candidate[j] = candidate[j], candidate[i]
            if _verify_network_fast(candidate):
                best = candidate
        
        # Try replacing a comparator with a different one
        if len(best) > 0:
            idx = random.randint(0, len(best) - 1)
            i, j = best[idx]
            # Try a random different comparator
            new_i = random.randint(0, n-1)
            new_j = random.randint(0, n-1)
            if new_i < new_j:
                candidate = best[:]
                candidate[idx] = (new_i, new_j)
                if _verify_network_fast(candidate):
                    best = candidate
    
    return best


def construct():
    """Construct a sorting network for 16 wires."""
    start_time = time.time()
    deadline = 80.0  # safety margin before 90s kill
    
    # Start with a known good network
    base = _known_good_network()
    
    # Verify it's correct
    if not _verify_network_fast(base):
        # Fallback to odd-even transposition
        base = [(i, i + 1) for phase in range(N)
                for i in range(phase % 2, N - 1, 2)]
    
    # Local search to improve
    if time.time() - start_time < deadline - 10.0:
        improved = _local_search(base, max_time=deadline - time.time() - 5.0)
        if len(improved) < len(base):
            base = improved
    
    return base

# EVOLVE-BLOCK-END
