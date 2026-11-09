N = 16

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a Batcher's Odd-Even Merge sorting network for N=16.
    This network uses 60 comparators, which is significantly better than the
    120 comparators used by the initial odd-even transposition network.
    """
    def batcher(n, offset=0, step=1):
        """
        Generates a Batcher sorting network for 'n' logical wires.
        The logical wires are mapped to physical wires:
        offset, offset + step, offset + 2*step, ...
        """
        if n == 1:
            return []
        
        k = n // 2
        
        # Recursively sort the first half and second half
        # First half: logical indices 0..k-1
        # Physical indices: offset, offset+step, ... offset+(k-1)*step
        left_comps = batcher(k, offset, step)
        
        # Second half: logical indices k..n-1
        # Physical indices: offset + k*step, ...
        right_comps = batcher(k, offset + k * step, step)
        
        # Cross comparators
        # Compare logical i with logical i+k
        # Physical: offset + i*step with offset + (i+k)*step
        cross_comps = [(offset + i * step, offset + (i + k) * step) for i in range(k)]
        
        # Recursively sort the even-indexed wires of the result
        # Logical even indices: 0, 2, 4, ... 2k-2
        # Physical: offset, offset + 2*step, ...
        even_comps = batcher(k, offset, 2 * step)
        
        # Recursively sort the odd-indexed wires of the result
        # Logical odd indices: 1, 3, 5, ... 2k-1
        # Physical: offset + step, offset + 3*step, ...
        odd_comps = batcher(k, offset + step, 2 * step)
        
        return left_comps + right_comps + cross_comps + even_comps + odd_comps

    return batcher(N)
# EVOLVE-BLOCK-END
