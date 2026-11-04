"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    """
    Construct the rank-49 tensor decomposition for 4x4 matrix multiplication.
    This uses the tensor square of Strassen's 2x2 algorithm.
    """
    # Strassen 2x2 decomposition matrices
    # U maps input A (2x2) to 7 intermediate values
    # V maps input B (2x2) to 7 intermediate values
    # W maps 7 products to output C (2x2)
    
    # Standard Strassen indices:
    # M1 = (a00+a11)*(b00+b11)
    # M2 = (a10+a11)*b00
    # M3 = a00*(b01-b11)
    # M4 = a11*(b10-b00)
    # M5 = (a00+a01)*b11
    # M6 = (a10-a00)*(b00+b01)
    # M7 = (a01-a11)*(b10+b11)
    
    # U rows correspond to A elements [a00, a01, a10, a11]
    # U cols correspond to M1..M7
    U2 = np.array([
        [1, 0, 1, 0, 1, 0, 0], # a00
        [0, 0, 0, 0, 1, 0, -1],# a01
        [0, 1, 0, 0, 0, 1, 0], # a10
        [1, 1, 0, 1, 0, 0, 0]  # a11
    ])
    
    # V rows correspond to B elements [b00, b01, b10, b11]
    # V cols correspond to M1..M7
    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0], # b00
        [0, 0, 1, 0, 0, 1, 0],  # b01
        [0, 0, 0, 1, 0, 0, 1],  # b10
        [1, 0, -1, 0, 1, 0, 1]  # b11
    ])
    
    # W rows correspond to C elements [c00, c01, c10, c11]
    # W cols correspond to M1..M7
    # c00 = M1 + M4 - M5 + M7
    # c01 = M3 + M5
    # c10 = M2 + M4
    # c11 = M1 - M2 + M3 + M6
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1], # c00
        [0, 0, 1, 0, 1, 0, 0],  # c01
        [0, 1, 0, 1, 0, 0, 0],  # c10
        [1, -1, 1, 0, 0, 1, 0]  # c11
    ])
    
    # Construct 4x4 decomposition via Kronecker products
    # The indices for 4x4 are formed by the Kronecker structure
    # U4 = U2 \otimes U2 has shape (4*4, 7*7) = (16, 49)
    # V4 = V2 \otimes V2 has shape (4*4, 7*7) = (16, 49)
    # W4 = W2 \otimes W2 has shape (4*4, 7*7) = (16, 49)
    
    U4 = np.kron(U2, U2)
    V4 = np.kron(V2, V2)
    W4 = np.kron(W2, W2)
    
    # Convert to lists of lists as required by the interface
    # The interface expects {'U': ..., 'V': ..., 'W': ...}
    # Shapes: U (n*m, rank) = (16, 49), V (m*p, rank) = (16, 49), W (p*n, rank) = (16, 49)
    
    return {
        "U": U4.tolist(),
        "V": V4.tolist(),
        "W": W4.tolist()
    }
# EVOLVE-BLOCK-END
