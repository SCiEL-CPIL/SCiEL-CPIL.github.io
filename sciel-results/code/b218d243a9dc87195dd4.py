"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2 by 4 matrix by a 4 by 8 matrix.
    Uses a structured algebraic decomposition based on Strassen's algorithm for 2x2x2 blocks.
    The 2x4x8 multiplication is decomposed into 8 independent 2x2x2 multiplications.
    Each 2x2x2 multiplication is performed using Strassen's algorithm with rank 7.
    Total rank = 8 * 7 = 56.
    """
    n, m, p = DIMENSIONS
    rank = 56
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Define Strassen coefficients for 2x2x2
    # M1: A(0,0)+A(1,1), B(0,0)+B(1,1) -> C(0,0)+, C(1,1)+
    # M2: A(1,0)+A(1,1), B(0,0)         -> C(1,0)+, C(1,1)-
    # M3: A(0,0), B(1,0)-B(1,1)          -> C(0,1)+, C(1,1)+
    # M4: A(1,1), B(0,1)-B(0,0)          -> C(0,0)+, C(1,0)+
    # M5: A(0,0)+A(0,1), B(1,1)          -> C(0,0)-, C(0,1)+
    # M6: A(1,0)-A(0,0), B(0,0)+B(0,1)   -> C(1,1)+
    # M7: A(0,1)-A(1,1), B(1,0)+B(1,1)   -> C(0,0)+
    
    # Structure for each term s (0-indexed 0..6):
    # U_coeffs: list of (i, j, coeff)
    # V_coeffs: list of (j, k, coeff)
    # W_coeffs: list of (k, i, coeff)
    
    strassen_terms = [
        # M1
        {
            'U': [(0, 0, 1), (1, 1, 1)],
            'V': [(0, 0, 1), (1, 1, 1)],
            'W': [(0, 0, 1), (1, 1, 1)]
        },
        # M2
        {
            'U': [(1, 0, 1), (1, 1, 1)],
            'V': [(0, 0, 1)],
            'W': [(0, 1, 1), (1, 1, -1)] # C(1,0) -> k=0, i=1; C(1,1) -> k=1, i=1
        },
        # M3
        {
            'U': [(0, 0, 1)],
            'V': [(1, 0, 1), (1, 1, -1)],
            'W': [(1, 0, 1), (1, 1, 1)] # C(0,1) -> k=1, i=0; C(1,1) -> k=1, i=1
        },
        # M4
        {
            'U': [(1, 1, 1)],
            'V': [(0, 1, 1), (0, 0, -1)],
            'W': [(0, 0, 1), (0, 1, 1)] # C(0,0) -> k=0, i=0; C(1,0) -> k=0, i=1
        },
        # M5
        {
            'U': [(0, 0, 1), (0, 1, 1)],
            'V': [(1, 1, 1)],
            'W': [(0, 0, -1), (1, 0, 1)] # C(0,0) -> k=0, i=0; C(0,1) -> k=1, i=0
        },
        # M6
        {
            'U': [(1, 0, 1), (0, 0, -1)],
            'V': [(0, 0, 1), (0, 1, 1)],
            'W': [(1, 1, 1)] # C(1,1) -> k=1, i=1
        },
        # M7
        {
            'U': [(0, 1, 1), (1, 1, -1)],
            'V': [(1, 0, 1), (1, 1, 1)],
            'W': [(0, 0, 1)] # C(0,0) -> k=0, i=0
        }
    ]
    
    # Loop over blocks
    # a: 0, 1 (row block of A, row block of B)
    # b: 0, 1, 2, 3 (col block of B, col block of C)
    for a in range(2):
        for b in range(4):
            for s in range(7):
                col = 7 * (a * 4 + b) + s
                term = strassen_terms[s]
                
                # Map local indices (0,1) to global indices
                # j_local 0 -> j_global 2a
                # j_local 1 -> j_global 2a+1
                # k_local 0 -> k_global 2b
                # k_local 1 -> k_global 2b+1
                # i_local 0 -> i_global 0
                # i_local 1 -> i_global 1
                
                # Fill U
                for i_l, j_l, coeff in term['U']:
                    i_g = i_l
                    j_g = 2 * a + j_l
                    U[i_g * m + j_g, col] += coeff
                
                # Fill V
                for j_l, k_l, coeff in term['V']:
                    j_g = 2 * a + j_l
                    k_g = 2 * b + k_l
                    V[j_g * p + k_g, col] += coeff
                
                # Fill W
                # W index is k*n + i
                for k_l, i_l, coeff in term['W']:
                    k_g = 2 * b + k_l
                    i_g = i_l
                    W[k_g * n + i_g, col] += coeff
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
