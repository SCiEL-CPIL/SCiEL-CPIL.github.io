import numpy as np
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # We decompose the 6x6x6 tensor into 9 disjoint 2x2x2 block tensors.
    # Each 2x2x2 block is multiplied using Strassen's algorithm (rank 7).
    # Total rank = 9 * 7 = 63.
    
    rank = 63
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    idx = 0
    # Iterate over the 3x3 grid of 2x2 blocks
    for i in range(3):  # Block row index
        for k in range(3): # Block col index (of A) / row index of B
            for j in range(3): # Block col index of B / col index of C
                # Indices in the 6x6 matrices
                # A is 6x6, B is 6x6, C is 6x6
                # Block A_ik is at rows 2i, 2i+1 and cols 2k, 2k+1
                # Block B_kj is at rows 2k, 2k+1 and cols 2j, 2j+1
                # Block C_ij is at rows 2j, 2j+1 and cols 2i, 2i+1
                
                # Local indices for the 2x2 block multiplication
                # A: (0,0)->a00, (0,1)->a01, (1,0)->a10, (1,1)->a11
                # B: (0,0)->b00, (0,1)->b01, (1,0)->b10, (1,1)->b11
                # C: (0,0)->c00, (0,1)->c01, (1,0)->c10, (1,1)->c11
                
                a00 = (2*i) * 6 + 2*k
                a01 = (2*i) * 6 + 2*k + 1
                a10 = (2*i+1) * 6 + 2*k
                a11 = (2*i+1) * 6 + 2*k + 1
                
                b00 = (2*k) * 6 + 2*j
                b01 = (2*k) * 6 + 2*j + 1
                b10 = (2*k+1) * 6 + 2*j
                b11 = (2*k+1) * 6 + 2*j + 1
                
                c00 = (2*j) * 6 + 2*i
                c01 = (2*j+1) * 6 + 2*i
                c10 = (2*j) * 6 + (2*i+1)
                c11 = (2*j+1) * 6 + (2*i+1)
                
                # Strassen's 7 multiplications
                # M1 = (A11 + A22)(B11 + B22)
                U[a00, idx] = 1
                U[a11, idx] = 1
                V[b00, idx] = 1
                V[b11, idx] = 1
                W[c00, idx] = 1
                W[c11, idx] = 1
                idx += 1
                
                # M2 = (A21 + A22)B11
                U[a10, idx] = 1
                U[a11, idx] = 1
                V[b00, idx] = 1
                W[c10, idx] = 1
                W[c11, idx] = -1
                idx += 1
                
                # M3 = A11(B12 - B22)
                U[a00, idx] = 1
                V[b01, idx] = 1
                V[b11, idx] = -1
                W[c01, idx] = 1
                W[c11, idx] = 1
                idx += 1
                
                # M4 = A22(B21 - B11)
                U[a11, idx] = 1
                V[b10, idx] = 1
                V[b00, idx] = -1
                W[c00, idx] = 1
                W[c10, idx] = 1
                idx += 1
                
                # M5 = (A11 + A12)B22
                U[a00, idx] = 1
                U[a01, idx] = 1
                V[b11, idx] = 1
                W[c00, idx] = -1
                W[c01, idx] = 1
                idx += 1
                
                # M6 = (A21 - A11)(B11 + B12)
                U[a10, idx] = 1
                U[a00, idx] = -1
                V[b00, idx] = 1
                V[b01, idx] = 1
                W[c11, idx] = 1
                idx += 1
                
                # M7 = (A12 - A22)(B21 + B22)
                U[a01, idx] = 1
                U[a11, idx] = -1
                V[b10, idx] = 1
                V[b11, idx] = 1
                W[c00, idx] = 1
                idx += 1
                
    assert idx == rank
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
