import numpy as np

N = 400

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a signed step function with 400 intervals using a truncated
    maximal length sequence (M-sequence) derived from the primitive polynomial
    x^9 + x^4 + 1. This deterministic sequence has near-optimal autocorrelation
    properties, significantly lower than random signs.
    """
    # Generate M-sequence of length 511 (2^9 - 1)
    # Polynomial: x^9 + x^4 + 1
    # Recurrence: s_{n+9} = s_{n+4} + s_n
    # LFSR configuration: 9-bit state, right-shift, feedback from bit 4 and bit 0
    state = 1
    seq_bits = []
    for _ in range(511):
        seq_bits.append(state & 1)
        # Calculate feedback bit
        # b4 = (state >> 4) & 1
        # b0 = state & 1
        new_bit = ((state >> 4) ^ state) & 1
        # Shift right and insert new bit at MSB (position 8)
        state = ((state >> 1) | (new_bit << 8)) & 0x1FF  # Mask to 9 bits
        
    # Truncate to N=400
    seq_400 = seq_bits[:400]
    
    # Map bits to signed heights: 0 -> 1.0, 1 -> -1.0
    signs = np.array([1.0 if b == 0 else -1.0 for b in seq_400])
    return signs
# EVOLVE-BLOCK-END
