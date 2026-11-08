"""Rewritten program for the third autocorrelation inequality task."""
import numpy as np

N = 400

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a signed step function with 400 equal intervals on [-1/4, 1/4].
    
    The goal is to minimize the upper bound on C3, which is related to the
    autoconvolution of the signed step function. We use a Thue-Morse sequence
    to create a deterministic, high-contrast binary mask that breaks the symmetry
    of the all-ones seed. This exploits the orthogonality properties of low-frequency
    components in the autoconvolution spectrum.
    
    The Thue-Morse sequence has the property that its Fourier transform has
    controlled decay, which helps in reducing the peak values of the autoconvolution.
    
    Returns:
        A numpy array of 400 signed step heights (+1 or -1).
    """
    # Generate the Thue-Morse sequence of length N
    # The Thue-Morse sequence is defined by t(n) = parity of the number of 1s in binary representation of n
    # Alternatively, it can be generated recursively: t(2n) = t(n), t(2n+1) = 1 - t(n)
    
    def thue_morse(n):
        """Generate the first n terms of the Thue-Morse sequence."""
        if n <= 0:
            return np.array([], dtype=np.int8)
        if n == 1:
            return np.array([0], dtype=np.int8)
        
        # Use the recursive definition
        # t(n) = popcount(n) % 2
        arr = np.arange(n)
        # Count bits set in each number
        # Use numpy to count bits
        # We can use a lookup table or iterative approach
        # For efficiency, use a vectorized approach with bit manipulation
        
        # Simple approach: use bin(x).count('1') but vectorized
        # Actually, let's use the property that Thue-Morse can be generated iteratively
        # Start with [0], then repeatedly append the complement
        seq = np.array([0], dtype=np.int8)
        while len(seq) < n:
            seq = np.concatenate([seq, 1 - seq])
        return seq[:n]
    
    # Generate Thue-Morse sequence
    tm = thue_morse(N)
    
    # Convert to signed heights: 0 -> +1, 1 -> -1
    heights = 1 - 2 * tm
    heights = heights.astype(np.float64)
    
    return heights
# EVOLVE-BLOCK-END
