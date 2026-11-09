"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

N = 16

# EVOLVE-BLOCK-START

def _apply_network_bitmask(network, mask):
    """Apply comparator network to a bitmask representing boolean state.
    Bit k represents wire k. Comparator (i,j) with i<j: min goes to i, max to j.
    For boolean: if bit i=1 and bit j=0, swap them (set i=0, j=1).
    """
    for i, j in network:
        # Check if wire i has 1 and wire j has 0
        bit_i = (mask >> i) & 1
        bit_j = (mask >> j) & 1
        if bit_i == 1 and bit_j == 0:
            # Swap: clear bit i, set bit j
            mask &= ~(1 << i)
            mask |= (1 << j)
    return mask

def _verify_network_fast(network):
    """Verify network sorts all 2^N boolean inputs using zero-one principle.
    Uses bitmask representation for speed.
    """
    for mask in range(1 << N):
        result = _apply_network_bitmask(network, mask)
        # Check if sorted: non-decreasing means 0s then 1s
        # For a sorted boolean array, it should be of the form 0...01...1
        # This means all 1s are at higher indices than all 0s
        # Count the number of 1s
        num_ones = bin(result).count('1')
        # The sorted form should have 1s in the top num_ones positions
        expected = ((1 << num_ones) - 1) << (N - num_ones)
        if result != expected:
            return False
    return True

def _batcher_oe_merge_sort(n):
    """Generate Batcher's odd-even merge sort network for n wires.
    Returns list of (i, j) comparators.
    """
    comparators = []
    
    def _bitonic_merge(lo, hi, count):
        """Merge two sorted subarrays [lo, lo+count) and [lo+count, lo+2*count)
        into a sorted array of size 2*count. Assumes first is ascending, second descending.
        """
        if count == 1:
            return
        # Odd-even merge
        for i in range(0, count, 2):
            comparators.append((lo + i, lo + count + i))
        # Recursively sort even and odd positions
        _bitonic_merge(lo, lo + count, count // 2)
        _bitonic_merge(lo + 1, lo + count + 1, count // 2)
    
    def _bitonic_split(lo, hi, count):
        """Create a bitonic sequence from a sorted sequence.
        First half ascending, second half descending.
        """
        if count <= 1:
            return
        mid = lo + count // 2
        _bitonic_split(lo, mid, count // 2)
        # Reverse second half to make it descending
        # We need to sort the second half in descending order
        # Use a helper to create descending sort
        _sort_descending(mid, mid + count // 2, count // 2)
    
    def _sort_ascending(lo, hi, count):
        """Sort wires [lo, hi) in ascending order."""
        if count <= 1:
            return
        mid = lo + count // 2
        _sort_ascending(lo, mid, count // 2)
        _sort_ascending(mid, mid + count // 2, count // 2)
        _bitonic_merge(lo, hi, count // 2)
    
    def _sort_descending(lo, hi, count):
        """Sort wires [lo, hi) in descending order."""
        if count <= 1:
            return
        mid = lo + count // 2
        _sort_descending(lo, mid, count // 2)
        _sort_descending(mid, mid + count // 2, count // 2)
        # For descending merge, we need to merge with reversed logic
        # Actually, let's use a different approach: sort ascending then reverse
        # Or use the fact that descending sort = ascending sort with reversed comparator direction
        # Let me rethink: for bitonic sort, we alternate directions
        
    # Simpler approach: implement standard Batcher odd-even merge sort
    def _generate(n):
        """Generate sorting network for n wires using odd-even merge sort."""
        if n <= 1:
            return []
        if n == 2:
            return [(0, 1)]
        
        # Split into two halves
        half = n // 2
        # Sort each half
        left = _generate(half)
        right = [(i + half, j + half) for i, j in _generate(half)]
        
        # Merge the two sorted halves
        merge = _merge(half, half)
        
        return left + right + merge
    
    def _merge(n1, n2):
        """Merge two sorted sequences of lengths n1 and n2 (n1 == n2 for our case)."""
        if n1 == 1 and n2 == 1:
            return [(0, 1)]
        
        # For odd-even merge sort, we assume n1 == n2
        n = n1 + n2
        half = n // 2
        
        if half <= 1:
            return [(0, 1)]
        
        # Odd-even merge: compare (i, i+half) for i in 0, 2, 4, ...
        comps = []
        for i in range(0, half, 2):
            comps.append((i, i + half))
        
        # Recursively sort even-indexed and odd-indexed subproblems
        even = _merge(half // 2, half // 2)
        odd = _merge(half // 2, half // 2)
        
        # Offset even by 0, odd by 1
        even_offset = [(i, j) for i, j in even]
        odd_offset = [(i + 1, j + 1) for i, j in odd]
        
        return comps + even_offset + odd_offset
    
    return _generate(n)

def _local_search_fast(network, deadline):
    """Local search to remove redundant comparators using fast verification."""
    current_network = list(network)
    best_network = list(current_network)
    best_count = len(current_network)
    
    # Verify initial
    if not _verify_network_fast(current_network):
        return current_network
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for idx in range(len(current_network) - 1, -1, -1):
            if time.time() >= deadline:
                break
            
            # Try removing comparator at idx
            test_net = current_network[:idx] + current_network[idx+1:]
            if _verify_network_fast(test_net):
                current_network = test_net
                best_network = list(current_network)
                best_count = len(current_network)
                improved = True
                break
    
    return best_network

# EVOLVE-BLOCK-END

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 10s safety margin
    
    # Start with Batcher's odd-even merge sort
    net = _batcher_oe_merge_sort(N)
    
    # Verify it's correct
    if not _verify_network_fast(net):
        # Fallback to odd-even transposition
        net = [(i, i + 1) for phase in range(N)
               for i in range(phase % 2, N - 1, 2)]
    
    # Local search to remove redundant comparators
    improved_net = _local_search_fast(net, deadline)
    
    return improved_net
