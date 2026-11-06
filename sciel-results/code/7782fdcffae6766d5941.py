"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

N = 18

# EVOLVE-BLOCK-START
def _batcher_odd_even_merge_sort(n):
    """
    Generate Batcher's odd-even merge sort network for n wires.
    Returns list of (i, j) comparators.
    """
    if n <= 1:
        return []
    
    # For powers of 2, use recursive odd-even merge
    # For non-powers of 2, we can still use the algorithm
    comps = []
    
    def _odd_even_merge(a, b):
        """Merge two sorted sequences of lengths a and b (a <= b)."""
        if a == 0:
            return []
        if a == 1:
            # Interleave single element with b elements
            result = []
            for i in range(b):
                result.append((0, i))  # compare wire 0 with wire i
            return result
        if b == 1:
            return []
        
        # Split into halves
        half_a = a // 2
        half_b = b // 2
        
        # Recursively merge
        result = []
        # Merge first halves
        result.extend(_odd_even_merge(half_a, half_b))
        # Merge second halves (offset by half_a + half_b)
        result2 = _odd_even_merge(a - half_a, b - half_b)
        for (i, j) in result2:
            result.append((i + half_a + half_b, j + half_a + half_b))
        
        # Cross comparisons
        # Compare elements at positions that need cross-wiring
        # For odd-even merge, we compare (i, i + half_a) for appropriate i
        # Actually, the cross step compares pairs that span the two halves
        
        # The standard odd-even merge cross step:
        # After merging two halves, compare (i, i + a) for i in range(b - a)
        # Wait, let me reconsider the algorithm.
        
        return result
    
    # Actually, let me use a cleaner implementation of Batcher's odd-even merge sort
    
    def _oesort(n):
        """Batcher's odd-even merge sort for n wires."""
        if n <= 1:
            return []
        if n == 2:
            return [(0, 1)]
        
        # Find the largest power of 2 less than or equal to n
        # For simplicity, let's handle n by padding or direct construction
        # Actually, let's use the recursive definition properly
        
        # Split n into two parts: a = floor(n/2), b = ceil(n/2)
        a = n // 2
        b = n - a
        
        # Recursively sort each half
        left = _oesort(a)
        right = _oesort(b)
        for (i, j) in right:
            left.append((i + a, j + a))
        
        # Odd-even merge the two sorted halves
        left.extend(_odd_even_merge(a, b))
        
        return left
    
    def _odd_even_merge(a, b):
        """
        Odd-even merge two sorted sequences of lengths a and b (a <= b).
        The sequences are on wires 0..a-1 and a..a+b-1.
        Returns comparators to merge them.
        """
        if a == 0:
            return []
        if a == 1:
            # One element to merge with b elements
            # The single element is at wire 0, the b elements are at wires 1..b
            # We need to insert wire 0 into the sorted sequence on wires 1..b
            # This is done by comparing (0, 1), then (1, 2), etc.
            # Actually for odd-even merge with a=1:
            # Compare (0, 1), (2, 3), ..., then (1, 2), (3, 4), ...
            # Wait, this is getting complex. Let me use a different approach.
            return [(0, i) for i in range(1, b + 1)]
        
        half_a = a // 2
        half_b = b // 2
        
        result = []
        
        # Merge first halves: wires 0..half_a-1 and half_a..half_a+half_b-1
        # But wait, the two sequences are on 0..a-1 and a..a+b-1
        # First half of seq1: 0..half_a-1
        # First half of seq2: a..a+half_b-1
        # We need to merge these, but they're not adjacent
        
        # Actually, the standard odd-even merge works on adjacent sorted sequences.
        # Let me reconsider: the two sorted sequences are on wires 0..a-1 and a..a+b-1.
        # They are adjacent!
        
        # Split seq1 into first half (0..half_a-1) and second half (half_a..a-1)
        # Split seq2 into first half (a..a+half_b-1) and second half (a+half_b..a+b-1)
        
        # Merge (first half of seq1, first half of seq2): wires 0..half_a-1 and a..a+half_b-1
        # These are NOT adjacent, so we can't directly apply _odd_even_merge
        
        # I think I'm overcomplicating this. Let me use a known-good implementation.
        
        return result
    
    # Let me just use a simpler approach: Batcher's bitonic sort, which is well-defined
    def _bitonic_sort(n):
        """Batcher's bitonic sort network for n wires."""
        if n <= 1:
            return []
        if n == 2:
            return [(0, 1)]
        
        # For non-powers of 2, pad to next power of 2
        # But we want exactly n wires, so let's handle it differently
        
        # Bitonic sort requires n to be a power of 2 for the standard algorithm
        # For n=18, let's use a different strategy
        
        return []
    
    # Given the complexity, let me use a different approach:
    # Start with a known good network and optimize
    
    # Actually, let me implement Batcher's odd-even merge sort correctly
    # Reference: https://en.wikipedia.org/wiki/Sorting_network
    
    def _batcher_oesort(n):
        """
        Batcher's odd-even merge sort for n wires.
        Returns a list of comparators.
        """
        if n <= 1:
            return []
        if n == 2:
            return [(0, 1)]
        
        # Split into two halves
        a = n // 2
        b = n - a
        
        # Recursively sort each half
        comps = _batcher_oesort(a)
        comps_right = _batcher_oesort(b)
        for (i, j) in comps_right:
            comps.append((i + a, j + a))
        
        # Now odd-even merge the two sorted halves
        comps.extend(_batcher_odd_even_merge(a, b))
        
        return comps
    
    def _batcher_odd_even_merge(a, b):
        """
        Odd-even merge two sorted sequences of lengths a and b (a <= b).
        Sequence 1 is on wires 0..a-1, sequence 2 is on wires a..a+b-1.
        Returns comparators to produce a single sorted sequence on wires 0..a+b-1.
        """
        if a == 0:
            return []
        if a == 1:
            # Insert single element (wire 0) into sorted sequence on wires 1..b
            # Use a series of comparators to bubble it into place
            # Compare (0,1), then if 0 went down, compare (1,2), etc.
            # For a sorting network, we just do all necessary comparisons
            # The correct way: compare (0,1), (1,2), ..., (b-1,b)
            # But that's not quite right either.
            # Actually, for odd-even merge with a=1:
            # We compare (0,1), then (2,3), ..., then (1,2), (3,4), ...
            # This is the odd-even transposition sort for merging
            result = []
            # Phase 1: compare (0,1), (2,3), (4,5), ...
            for i in range(0, b, 2):
                result.append((i, i + 1))
            # Phase 2: compare (1,2), (3,4), ...
            for i in range(1, b, 2):
                result.append((i, i + 1))
            return result
        
        if b == 1:
            return []
        
        # Split into halves
        half_a = a // 2
        half_b = b // 2
        
        result = []
        
        # Merge first halves: seq1[0..half_a-1] with seq2[0..half_b-1]
        # seq1[0..half_a-1] is on wires 0..half_a-1
        # seq2[0..half_b-1] is on wires a..a+half_b-1
        # These are not adjacent, so we need to be careful
        
        # Actually, in odd-even merge, the two sequences are interleaved in a specific way.
        # Let me look at this more carefully.
        
        # The standard odd-even merge for sequences A (length a) and B (length b):
        # 1. Split A into A1 (first half) and A2 (second half)
        # 2. Split B into B1 (first half) and B2 (second half)
        # 3. Recursively merge A1 with B1, and A2 with B2
        # 4. Cross-compare: compare A1[i] with B1[i] for appropriate i
        
        # But in our wire layout, A is on wires 0..a-1 and B is on wires a..a+b-1.
        # A1 is on wires 0..half_a-1, A2 is on wires half_a..a-1
        # B1 is on wires a..a+half_b-1, B2 is on wires a+half_b..a+b-1
        
        # To merge A1 and B1, they need to be on adjacent wires.
        # This suggests the algorithm works on a different wire layout.
        
        # I think the issue is that odd-even merge assumes the sequences are 
        # interleaved or laid out in a specific way.
        
        # Let me just use a brute-force approach for now:
        # Generate a network that's known to work, then optimize.
        
        return result
    
    # Given the complexity of getting Batcher's algorithm exactly right for non-powers-of-2,
    # let me use a different strategy:
    # 1. Generate a correct but potentially suboptimal network
    # 2. Use local search to remove redundant comparators
    
    # Strategy: Use the 0-1 principle and local search
    
    return []

def _verify_network(comps, n=N):
    """
    Verify that a sorting network sorts all binary inputs.
    Returns True if the network is correct.
    """
    num_inputs = 1 << n
    for mask in range(num_inputs):
        # Create the input vector
        vec = [(mask >> i) & 1 for i in range(n)]
        
        # Apply each comparator
        for (i, j) in comps:
            if vec[i] > vec[j]:
                vec[i], vec[j] = vec[j], vec[i]
        
        # Check if sorted
        for i in range(n - 1):
            if vec[i] > vec[i + 1]:
                return False
    
    return True

def _verify_network_fast(comps, n=N):
    """
    Faster verification using numpy vectorization.
    """
    num_inputs = 1 << n
    # Create all input vectors as a 2D array: shape (num_inputs, n)
    # Each row is a binary vector
    all_vecs = np.zeros((num_inputs, n), dtype=np.uint8)
    for i in range(n):
        all_vecs[:, i] = (np.arange(num_inputs) >> i) & 1
    
    # Apply each comparator
    for (i, j) in comps:
        # Where vec[i] > vec[j], swap
        mask = all_vecs[:, i] > all_vecs[:, j]
        # Swap values where mask is True
        tmp = all_vecs[mask, i].copy()
        all_vecs[mask, i] = all_vecs[mask, j]
        all_vecs[mask, j] = tmp
    
    # Check if all rows are sorted
    # A row is sorted if all_vecs[:, i] <= all_vecs[:, i+1] for all i
    for i in range(n - 1):
        if np.any(all_vecs[:, i] > all_vecs[:, i + 1]):
            return False
    
    return True

def _initial_network():
    """
    Generate an initial correct sorting network for 18 wires.
    Uses a simple but correct approach.
    """
    comps = []
    
    # Use a series of odd-even transposition phases
    # This gives a correct but suboptimal network
    for phase in range(N):
        start = phase % 2
        for i in range(start, N - 1, 2):
            comps.append((i, i + 1))
    
    return comps

def _remove_redundant_comparators(comps, n=N, time_limit=50):
    """
    Iteratively remove redundant comparators from the network.
    A comparator is redundant if removing it still leaves the network correct.
    """
    start_time = time.time()
    current = list(comps)
    
    # Try removing each comparator, starting from the end
    # Use a greedy approach: try to remove each one
    changed = True
    while changed:
        changed = False
        for idx in range(len(current) - 1, -1, -1):
            if time.time() - start_time > time_limit:
                break
            
            # Try removing comparator at idx
            test = current[:idx] + current[idx+1:]
            
            # Verify
            if _verify_network_fast(test, n):
                current = test
                changed = True
        
        if time.time() - start_time > time_limit:
            break
    
    return current

def construct():
    """
    Construct a sorting network for 18 wires.
    Returns a list of (i, j) comparators.
    """
    start_time = time.time()
    deadline = start_time + 85  # Leave 5s margin for the 90s limit
    
    # Generate initial network
    comps = _initial_network()
    
    # Remove redundant comparators
    comps = _remove_redundant_comparators(comps, N, time_limit=75)
    
    # Verify the final network
    if not _verify_network_fast(comps, N):
        # Fallback to the initial network
        comps = _initial_network()
    
    return comps
# EVOLVE-BLOCK-END
