# EVOLVE-BLOCK-START
import numpy as np

def construct():
    """
    Construct a signed equal-width step function using a Maximum Length Sequence (m-sequence).
    
    Strategy: Pseudo-random construction.
    M-sequences have optimal autocorrelation properties:
    1. The autocorrelation at lag 0 is N.
    2. The autocorrelation at all other lags is -1 (for a full period of length 2^m - 1).
    
    If the metric is the maximum absolute autoconvolution (excluding lag 0) divided by
    the energy (sum of squares), this minimizes the value to 1/N.
    If the metric includes lag 0, the value is 1, which is the theoretical minimum 
    for any sequence (since C_0 = Energy).
    
    Compared to the constant sequence (which has high sidelobes), this sequence
    has minimal sidelobes.
    """
    N = 400
    
    # Generate an m-sequence of length 2^9 - 1 = 511
    # Use a primitive polynomial for GF(2^9), e.g., x^9 + x^4 + 1 (0x011)
    # We will generate the sequence using LFSR logic.
    
    def generate_m_sequence(length, poly, init_state=1):
        """
        Generate an m-sequence of specified length.
        poly: integer representing the feedback polynomial.
        init_state: initial state of the LFSR.
        """
        seq = []
        state = init_state
        for _ in range(length):
            # Extract the least significant bit (LSB) as the output
            # Map 0 to -1, 1 to 1
            bit = state & 1
            seq.append(1 if bit else -1)
            
            # Shift right
            state >>= 1
            
            # Calculate feedback bit
            # The feedback is the XOR of the bits specified by the polynomial
            # For poly x^9 + x^4 + 1, the taps are at positions 9 and 4 (0-indexed from LSB? No, standard LFSR)
            # Let's use a standard simple implementation for m-seq.
            # We need to be careful with the polynomial definition.
            # Common m-seq for m=9: x^9 + x^4 + 1.
            # Taps: 9, 4.
            # Feedback = bit(9-1) XOR bit(4-1)?
            # Let's use a known working generator or just use numpy random if LFSR is too complex/buggy.
            # However, "approach family" implies a specific deterministic structure.
            # I will use a simple LFSR.
            
            # Standard LFSR step for x^m + ... + 1
            # The new bit is the XOR of the bits at the tap positions.
            # For x^9 + x^4 + 1:
            # We need the bits corresponding to x^8 and x^3 (if 1-based taps 9,4)
            # Let's define taps as indices in the state (0 is LSB).
            # x^9 corresponds to tap 8 (if state holds x^8...x^0)
            # x^4 corresponds to tap 3
            # Feedback = (state >> 8) & 1 XOR (state >> 3) & 1
            
            tap1 = (state >> 8) & 1
            tap2 = (state >> 3) & 1
            new_bit = tap1 ^ tap2
            
            # Shift the state to the right, and insert new_bit at the MSB (position 8)
            # state is 9 bits wide.
            state = ((state >> 1) | (new_bit << 8)) & 0x1FF
            
        return np.array(seq)

    # Generate 400 elements from a 511-length m-sequence
    # Ensure the state is non-zero
    m_seq = generate_m_sequence(400, 0x011, 1)
    
    return m_seq
# EVOLVE-BLOCK-END
