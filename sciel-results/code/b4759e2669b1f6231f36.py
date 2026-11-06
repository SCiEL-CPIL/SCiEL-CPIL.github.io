N = 18

def batcher_oes(n):
    """Generate Batcher's odd-even mergesort network for n wires."""
    if n <= 1:
        return []
    if n == 2:
        return [(0, 1)]
    
    # Split into two halves
    m = n // 2
    k = n - m  # k >= m
    
    # Recursively sort each half
    left = batcher_oes(m)
    right = batcher_oes(k)
    # Offset right half by m
    right = [(i + m, j + m) for (i, j) in right]
    
    # Merge phase: odd-even merge
    def odd_even_merge(a, b, offset=0):
        """Merge two sorted sequences of lengths a and b, starting at offset."""
        if a == 0 or b == 0:
            return []
        if a == 1 and b == 1:
            return [(offset, offset + 1)]
        
        comparators = []
        # Recursive merge
        # Split into odd and even positions
        # For merging a and b, we need to interleave
        
        # Standard odd-even merge:
        # If both even, split into halves
        if a % 2 == 0 and b % 2 == 0:
            a2, b2 = a // 2, b // 2
            # Merge first halves
            comparators += odd_even_merge(a2, b2, offset)
            # Merge second halves (offset by a2+b2... wait, this is getting complex)
            # Actually the standard approach:
            # Merge(a,b): if a,b even, Merge(a/2,b/2) then Merge(a/2,b/2) on second halves
            # The offsets are tricky. Let me use a different formulation.
            pass
        
        # This is getting complicated. Let me use a simpler recursive formulation.
        return comparators
    
    # Let me use a cleaner implementation
    comparators = left + right
    
    # Now merge the two sorted halves of size m and k
    # Using Batcher's odd-even merge
    def oem(a, b):
        """Odd-even merge of two sorted sequences of length a and b (a<=b).
           Returns list of (i,j) comparators for wires 0..a+b-1."""
        if a == 0:
            return []
        if a == 1 and b == 1:
            return [(0, 1)]
        
        comps = []
        if a % 2 == 0 and b % 2 == 0:
            a2, b2 = a // 2, b // 2
            # Merge first halves: wires 0..a2-1 and a2..a2+b2-1
            # Wait, the standard OES merge:
            # To merge two sorted arrays of size a and b (a<=b):
            # If both even: 
            #   Merge(a/2, b/2) on first halves, then Merge(a/2, b/2) on second halves
            #   Then compare-exchange (i, i+a/2+b/2) for appropriate i
            # Actually let me look at this more carefully.
            
            # Standard Batcher OES merge for equal sizes n,n:
            # OES(n): 
            #   OES(n/2) on first half, OES(n/2) on second half
            #   Then compare (i, i+n/2) for i=0..n/2-1
            #   Then OES(n/2) on even-indexed, OES(n/2) on odd-indexed
            
            # For unequal sizes a,b (a<=b), the merge is more complex.
            # Let me just use the fact that for our purposes, we can pad to power of 2
            # or use a different approach.
            
            # Simpler: just do a full odd-even transposition on the merged part
            # That's not optimal but correct.
            pass
        
        return comps
    
    # This is getting too complex inline. Let me just generate the network
    # and verify it.
    
    # For now, let me return a known good network.
    # Batcher's OES for 18 wires:
    # Let me just implement it properly.
    
    return comparators

def construct():
    """Construct a sorting network for 18 wires."""
    # Use Batcher's odd-even mergesort
    def oes(n, offset=0):
        if n <= 1:
            return []
        m = n // 2
        k = n - m
        left = oes(m, offset)
        right = oes(k, offset + m)
        merge = merge_oes(m, k, offset)
        return left + right + merge
    
    def merge_oes(a, b, offset):
        """Odd-even merge of two sorted sequences of length a and b at offset."""
        if a == 0 or b == 0:
            return []
        if a == 1 and b == 1:
            return [(offset, offset + 1)]
        
        if a == 1:
            # Insert one element into sorted sequence of length b
            # Binary insert
            comps = []
            # Compare with middle, then recurse
            mid = b // 2
            comps.append((offset, offset + mid))
            if mid > 0:
                comps += merge_oes(1, mid, offset + 1)  # Not quite right
            # This isn't quite right either. Let me just use a simple approach.
            pass
        
        # For simplicity, use odd-even transposition on the merged section
        # This is correct but not optimal
        comps = []
        total = a + b
        for phase in range(total):
            start = phase % 2
            for i in range(start, total - 1, 2):
                comps.append((offset + i, offset + i + 1))
        return comps
    
    return oes(N)
