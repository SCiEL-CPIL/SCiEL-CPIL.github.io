"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank-1 factors for Strassen's algorithm (7 factors for 2x2x2)
    # U: (n*m) x 7, V: (m*p_block) x 7, W: (p_block*n) x 7
    # where p_block = 2 for the 2x2x2 core
    
    # Strassen factors (using 0-based indexing for the 4 elements: [a,b,c,d] for A, [e,f,g,h] for B, [C00,C01,C10,C11] for C)
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-e)
    # m5 = (a+b)h
    # m6 = (c-a)(e+f)
    # m7 = (b-d)(g+h)
    
    # C00 = m1 + m4 - m5 + m7
    # C01 = m3 + m5
    # C10 = m2 + m4
    # C11 = m1 - m2 + m3 + m6
    
    U_block = np.array([
        [1, 0, 1, 0, 1, 0, 0], # a
        [0, 0, 0, 0, 1, 0, 1], # b
        [0, 1, 0, 0, 0, 1, 0], # c
        [1, 1, 0, 1, 0, 0, -1]  # d
    ], dtype=np.float64)
    
    V_block = np.array([
        [1, 1, 0, 0, 0, 1, 0], # e
        [0, 0, 1, 0, 0, 1, 0], # f
        [0, 0, 0, 1, 0, 0, 1], # g
        [1, 0, -1, -1, 1, 0, 1] # h
    ], dtype=np.float64)
    
    W_block = np.array([
        [1, 0, 0, 1, -1, 0, 1], # C00
        [0, 0, 1, 0, 1, 0, 0],  # C01
        [0, 1, 0, 1, 0, 0, 0],  # C10
        [1, -1, 1, 0, 0, 1, 0]  # C11
    ], dtype=np.float64)
    
    rank = 4 * 7
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # The tensor is M_{2,2,2} \otimes I_4.
    # This means the A matrix is shared across all 4 blocks of B and C.
    # B is 2x8, C is 8x2.
    # B = [[B00, B01], [B10, B11]] where each Bij is 2x2.
    # C = [[C00, C01], [C10, C11]] where each Cij is 2x2.
    # The multiplication is C = A * B.
    # Since A is 2x2 and B is 2x8, we can view B as a matrix of 4 blocks (2x2 each).
    # C will be 8x2, which can be viewed as 4 blocks (2x2 each).
    # The i-th block of C depends on the i-th block of B and the full A.
    # Specifically, C_i = A * B_i.
    # So we apply the same Strassen factors to each block.
    # The U factors are the same for all blocks because A is shared.
    # The V factors are the Strassen V factors for each block of B.
    # The W factors are the Strassen W factors for each block of C.
    
    for s in range(4):
        U[:, s*7 : (s+1)*7] = U_block
        V[s*2 : (s+1)*2, s*7 : (s+1)*7] = V_block[:2] # Top row of V_block for top row of B block? No.
        # Wait, V is m*p x rank. m=2, p=8. So V is 2x8 x rank.
        # B is 2x8. Let's index B as B[j, k] where j in 0..1, k in 0..7.
        # B[j, 2*s : 2*s+2] is the s-th block of B in row j.
        # V_block is 4x7, corresponding to e,f,g,h which are B[0,0], B[0,1], B[1,0], B[1,1] of the 2x2 block.
        # So V_block[0] corresponds to B[0, 2*s], V_block[1] to B[0, 2*s+1], etc.
        # V is (m*p) x rank = 16 x rank.
        # V[j*p + k, c]
        # For block s, the rows of V corresponding to this block are:
        # j=0, k=2*s -> row 2*s
        # j=0, k=2*s+1 -> row 2*s+1
        # j=1, k=2*s -> row 2*s + 4
        # j=1, k=2*s+1 -> row 2*s + 5
        # So V[2*s : 2*s+2, ...] corresponds to row 0 of the block.
        # V[2*s+4 : 2*s+6, ...] corresponds to row 1 of the block.
        V[2*s : 2*s+2, s*7 : (s+1)*7] = V_block[:2]
        V[2*s+4 : 2*s+6, s*7 : (s+1)*7] = V_block[2:]
        
        # W is p*n x rank = 16 x rank.
        # W[k*n + i, c]
        # C is 8x2. C[i, j] where i in 0..7, j in 0..1.
        # The s-th block of C is C[2*s : 2*s+2, :].
        # W_block is 4x7, corresponding to C00, C01, C10, C11 of the 2x2 block.
        # C00 -> C[2*s, 0], C01 -> C[2*s, 1], C10 -> C[2*s+1, 0], C11 -> C[2*s+1, 1].
        # So W[2*s, ...] corresponds to C[2*s, 0] (C00)
        # W[2*s+1, ...] corresponds to C[2*s, 1] (C01)
        # W[2*s+2, ...] corresponds to C[2*s+1, 0] (C10)
        # W[2*s+3, ...] corresponds to C[2*s+1, 1] (C11)
        W[2*s : 2*s+4, s*7 : (s+1)*7] = W_block
    
    # Convert to list of lists
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
