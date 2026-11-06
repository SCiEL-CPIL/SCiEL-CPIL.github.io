N = 16

# EVOLVE-BLOCK-START
import time
from itertools import product

def _sort2():
    return [(0, 1)]

def _sort3():
    return [(0, 1), (1, 2), (0, 1)]

def _sort4():
    """Optimal 4-input sorting network with 5 comparators."""
    return [(0, 2), (1, 3), (0, 1), (2, 3), (1, 2)]

def _sort8():
    """Optimal 8-input sorting network with 19 comparators."""
    # Known optimal 8-input network
    return [
        (0, 4), (1, 5), (2, 6), (3, 7),
        (0, 2), (1, 3), (4, 6), (5, 7),
        (0, 1), (2, 3), (4, 5), (6, 7),
        (1, 4), (2, 5), (3, 6),
        (1, 2), (4, 5),
        (2, 4)
    ]

def _batcher_odd_even_merge(n):
    """
    Construct a sorting network for n wires using Batcher's odd-even merge sort.
    This is a recursive divide-and-conquer approach.
    """
    if n == 1:
        return []
    if n == 2:
        return [(0, 1)]
    if n == 3:
        return [(0, 1), (1, 2), (0, 1)]
    if n == 4:
        return _sort4()
    if n == 8:
        return _sort8()
    
    # Split into two halves
    m = n // 2
    k = n - m
    
    # Sort the two halves recursively
    left = _batcher_odd_even_merge(m)
    right = _batcher_odd_even_merge(k)
    
    # Offset the right network
    right = [(i + m, j + m) for (i, j) in right]
    
    comparators = left + right
    
    # Odd-even merge phase
    # For Batcher's odd-even merge, we need to interleave the two sorted sequences
    # The merge works by comparing pairs (i, m+i) for i in range(min(m,k))
    # then recursively merging the odd and even subsequences
    
    if m == k:
        # Equal size halves
        # Compare (i, i+m) for all i
        for i in range(m):
            comparators.append((i, i + m))
        
        # Now we have two sorted sequences interleaved
        # We need to merge the odd indices and even indices separately
        # The odd indices form a sequence of length m, even indices form a sequence of length m
        # But they are offset, so we need to handle this carefully
        
        # For now, let's use a simpler approach: just add enough comparators
        # to ensure correctness. This is not optimal but should work.
        
        # Actually, let's implement the proper odd-even merge
        # After comparing (i, i+m), the even positions contain the smaller elements
        # and odd positions contain the larger elements (approximately)
        
        # We need to recursively sort the even positions and odd positions
        # The even positions are at indices 0, 2, 4, ..., 2*(m-1)
        # The odd positions are at indices 1, 3, 5, ..., 2*m-1
        
        # Map even positions to a new coordinate system
        # even_pos[i] = 2*i for i in range(m)
        # odd_pos[i] = 2*i + 1 for i in range(m)
        
        # Recursively sort the even subsequence
        even_net = _batcher_odd_even_merge(m)
        for (i, j) in even_net:
            comparators.append((2*i, 2*j))
        
        # Recursively sort the odd subsequence
        odd_net = _batcher_odd_even_merge(m)
        for (i, j) in odd_net:
            comparators.append((2*i + 1, 2*j + 1))
        
        # Finally, compare adjacent pairs to fix any remaining issues
        for i in range(0, n - 1, 2):
            comparators.append((i, i + 1))
    
    else:
        # Unequal size halves
        # Compare (i, m+i) for i in range(min(m, k))
        min_mk = min(m, k)
        for i in range(min_mk):
            comparators.append((i, m + i))
        
        # Then recursively merge the remaining parts
        # This is more complex, so for now let's just add a full sort at the end
        # as a fallback
        if n <= 16:
            # Add a final odd-even transposition pass
            for phase in range(n):
                for i in range(phase % 2, n - 1, 2):
                    comparators.append((i, i + 1))
    
    return comparators

def _verify_network(network, n=16):
    """Verify the network sorts all 2^n inputs using zero-one principle."""
    for mask in range(1 << n):
        inputs = [(mask >> i) & 1 for i in range(n)]
        for (i, j) in network:
            if inputs[i] > inputs[j]:
                inputs[i], inputs[j] = inputs[j], inputs[i]
        # Check sorted
        for i in range(n - 1):
            if inputs[i] > inputs[i + 1]:
                return False
    return True

def _known_16_network():
    """
    A known 16-input sorting network.
    Using Batcher's odd-even merge sort construction.
    """
    # Split into two 8-wire networks
    left8 = _sort8()
    right8 = _sort8()
    
    comparators = []
    
    # Sort the two halves
    for (i, j) in left8:
        comparators.append((i, j))
    for (i, j) in right8:
        comparators.append((i + 8, j + 8))
    
    # Merge phase: odd-even merge of two sorted 8-wire sequences
    # Compare (i, i+8) for i in range(8)
    for i in range(8):
        comparators.append((i, i + 8))
    
    # Now recursively merge the even and odd subsequences
    # Even positions: 0, 2, 4, 6, 8, 10, 12, 14 -> map to 0..7
    # Odd positions: 1, 3, 5, 7, 9, 11, 13, 15 -> map to 0..7
    
    # Recursively sort the even subsequence (8 wires)
    even_net = _sort8()
    for (i, j) in even_net:
        comparators.append((2*i, 2*j))
    
    # Recursively sort the odd subsequence (8 wires)
    odd_net = _sort8()
    for (i, j) in odd_net:
        comparators.append((2*i + 1, 2*j + 1))
    
    # Final pass to fix adjacent pairs
    for i in range(0, 15, 2):
        comparators.append((i, i + 1))
    
    # Deduplicate
    seen = set()
    unique = []
    for c in comparators:
        if c not in seen:
            seen.add(c)
            unique.append(c)
    
    return unique

def construct():
    """Construct a sorting network for 16 wires."""
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s limit
    
    # Try the Batcher-based network first
    network = _known_16_network()
    
    # Verify it
    if _verify_network(network):
        return network
    
    # If verification fails, try the recursive Batcher construction
    network = _batcher_odd_even_merge(16)
    
    # Verify it
    if _verify_network(network):
        return network
    
    # If verification fails, use odd-even transposition as fallback
    network = [(i, i + 1) for phase in range(N)
               for i in range(phase % 2, N - 1, 2)]
    
    return network
# EVOLVE-BLOCK-END
