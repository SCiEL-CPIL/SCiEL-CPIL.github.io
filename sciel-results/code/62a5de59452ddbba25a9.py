import time

def construct():
    """
    Constructs the Extended Quadratic Residue code of length 18, dimension 9, distance 6.
    This code has 512 codewords.
    """
    # Generator polynomial for the length 17 Q.R. code [17, 9, 6]
    # Roots are quadratic residues mod 17: 1, 2, 4, 8, 9, 13, 15, 16
    # g(x) = x^8 + x^5 + x^4 + x^3 + 1
    G = (1 << 8) | (1 << 5) | (1 << 4) | (1 << 3) | 1
    
    words = [0] * 512
    
    # Iterate over all 2^9 possible messages
    for m in range(512):
        # Polynomial multiplication m(x) * g(x) in GF(2)
        # m has degree < 9, g has degree 8
        # Result has degree < 17
        c = 0
        for i in range(9):
            if (m >> i) & 1:
                c ^= (G << i)
        
        # c contains the 17 bits of the codeword (bits 0 to 16)
        # Calculate the overall parity bit for extension to length 18
        parity = c.bit_count() & 1
        
        # Construct the 18-bit word
        # The first 17 bits are c, the 18th bit (index 17) is parity
        words[m] = c | (parity << 17)
        
    return words
