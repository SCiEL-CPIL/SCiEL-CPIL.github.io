import time
import numpy as np

N = 18

# EVOLVE-BLOCK-START

def _odd_even_transposition_network(n):
    """Generate an odd-even transposition sorting network for n wires."""
    comparators = []
    for phase in range(n):
        for i in range(phase % 2, n - 1, 2):
            comparators.append((i, i + 1))
    return comparators


def _verify_network(comparators, n=N, num_samples=200000):
    """Verify that the network sorts all boolean inputs."""
    if not comparators:
        return n <= 1
    
    for comp in comparators:
        if comp[0] >= comp[1] or comp[0] < 0 or comp[1] >= n:
            return False
    
    # Check all boolean inputs
    for mask in range(1 << n):
        wires = np.array([(mask >> i) & 1 for i in range(n)], dtype=np.int8)
        for i, j in comparators:
            if wires[i] > wires[j]:
                wires[i], wires[j] = wires[j], wires[i]
        # Check sorted: all 1s before all 0s
        ones = np.sum(wires)
        if not (all(wires[k] == 1 for k in range(n - ones)) and 
                all(wires[k] == 0 for k in range(n - ones, n))):
            return False
    return True


def _batcher_merge(a, b, offset):
    """
    Generate comparators to merge two sorted sequences of lengths a and b
    starting at 'offset'.
    """
    if a == 0:
        return []
    if b == 0:
        return []
    
    m = max(a, b)
    if a > b:
        a, b = b, a
    
    # Standard Batcher merge for two sorted sequences
    comparators = []
    
    # For merging two sorted sequences of length m, we need m compare-exchange steps
    # where in step k, we compare positions that are k apart in a specific pattern
    
    # Simple but correct approach: use the fact that merging two sorted lists
    # of length m requires at most m*m/4 + m comparators in the worst case,
    # but Batcher's method uses a specific pattern.
    
    # Batcher's merge network for two sorted sequences of length m:
    # Phase k (0-indexed): compare (offset + i, offset + m + i) for appropriate i
    
    # Actually, let me use a simpler correct method:
    # Generate all necessary compare-exchanges for merging
    
    # The key insight: to merge two sorted sequences of length m,
    # we can use the following pattern:
    # For d in range(m):
    #   For i in range(m - d):
    #     compare(offset + i, offset + m + i + d) if within bounds
    
    # But this isn't quite right either. Let me use a different approach.
    
    # For correctness, let me just generate a network that definitely works:
    # Use odd-even transposition on the combined sequence of length a+b
    # restricted to just the merge region.
    
    # Actually, the simplest correct approach for Batcher's odd-even merge:
    # Two sorted sequences of length m can be merged in m stages.
    # In stage s (0-indexed), compare elements at positions that are s apart.
    
    # Let me use the standard Batcher odd-even merge:
    # For two sorted sequences of length m:
    #   for k in range(m):
    #     for i in range(0, 2*m - 2*k - 1, 2):
    #       compare(i, i + 2*k + 1)  [within the merge region]
    
    # This is getting complex. Let me use a simpler but still correct method:
    # Just do a full odd-even transposition on the combined region.
    
    combined_len = a + b
    comps = []
    for phase in range(combined_len):
        for i in range(phase % 2, combined_len - 1, 2):
            p1 = offset + i
            p2 = offset + i + 1
            if p1 < offset + combined_len and p2 < offset + combined_len:
                if p1 != p2:
                    comps.append((p1, p2))
    return comps


def _batcher_sort(n, offset=0):
    """
    Generate a Batcher sorting network for n wires starting at 'offset'.
    """
    if n <= 1:
        return []
    if n == 2:
        return [(offset, offset + 1)]
    
    # Split into two halves
    m = n // 2
    rest = n - m
    
    # Recursively sort each half
    comparators = []
    comparators.extend(_batcher_sort(m, offset))
    comparators.extend(_batcher_sort(rest, offset + m))
    
    # Merge the two sorted halves
    comparators.extend(_batcher_merge(m, rest, offset))
    
    return comparators


def _remove_redundant_comparators(comparators, n=N):
    """
    Try to remove redundant comparators from the network.
    A comparator (i,j) is redundant if removing it still leaves the network correct.
    """
    if not comparators:
        return comparators
    
    result = list(comparators)
    
    for idx in range(len(result) - 1, -1, -1):
        if idx >= len(result):
            continue
        # Try removing this comparator
        candidate = result[:idx] + result[idx+1:]
        if _verify_network(candidate, n):
            result = candidate
    
    return result


def construct():
    """
    Construct a sorting network for N=18 wires.
    Returns an ordered list of integer pairs (i, j) where i < j.
    """
    start_time = time.time()
    deadline = start_time + 85  # Leave 5s margin before 90s kill
    
    # Start with Batcher's odd-even merge sort
    network = _batcher_sort(N, 0)
    
    # Verify it's correct
    if not _verify_network(network, N):
        # Fallback to odd-even transposition
        network = _odd_even_transposition_network(N)
        if not _verify_network(network, N):
            # This shouldn't happen, but just in case
            return _odd_even_transposition_network(N)
    
    # Try to optimize by removing redundant comparators
    # Work backwards through the network, removing comparators if possible
    optimized = list(network)
    
    for idx in range(len(optimized) - 1, -1, -1):
        if time.time() > deadline - 10:  # Leave 10s for final verification
            break
        
        if idx >= len(optimized):
            continue
            
        candidate = optimized[:idx] + optimized[idx+1:]
        if _verify_network(candidate, N):
            optimized = candidate
    
    # Final verification
    if not _verify_network(optimized, N):
        # Fall back to the original Batcher network
        return network
    
    return optimized

# EVOLVE-BLOCK-END
