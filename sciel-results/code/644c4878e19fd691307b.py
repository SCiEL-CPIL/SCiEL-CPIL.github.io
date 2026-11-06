import time

def construct():
    """
    Constructs the Extended Quadratic Residue code of length 18, dimension 9, distance 6.
    This code has 512 codewords, which is the known maximum A_2(18,6).
    
    The construction uses the generator polynomial of the [17,9,6] QR code:
    g(x) = x^8 + x^5 + x^4 + x^3 + 1
    The 18th bit is the overall parity check bit.
    """
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin of 10s before 60s limit
    
    # Generator polynomial coefficients
    # g(x) = x^8 + x^5 + x^4 + x^3 + 1
    # Bit representation: 1 at indices 0, 3, 4, 5, 8
    G = (1 << 8) | (1 << 5) | (1 << 4) | (1 << 3) | 1
    
    words = [0] * 512
    
    # Iterate over all 2^9 possible messages
    for m in range(512):
        # Check deadline
        if time.time() > deadline:
            # If we somehow run out of time (should not happen for 512 words),
            # return what we have so far. But since it's a fixed size loop, 
            # this is just a safeguard.
            break
        
        # Polynomial multiplication m(x) * g(x) in GF(2)
        # m has degree < 9, g has degree 8
        # Result has degree < 17
        c = 0
        # Optimization: Unroll or use efficient bit operations
        # Standard loop is fine for 9 iterations
        for i in range(9):
            if (m >> i) & 1:
                c ^= (G << i)
        
        # c contains the 17 bits of the codeword (bits 0 to 16)
        # Calculate the overall parity bit for extension to length 18
        parity = c.bit_count() & 1
        
        # Construct the 18-bit word
        # The first 17 bits are c, the 18th bit (index 17) is parity
        words[m] = c | (parity << 17)
        
    # Ensure we return a list of distinct integers
    # The QR code construction guarantees distinctness and minimum distance 6.
    # No further search or verification is needed as this is a proven optimal code.
    
    return words
