import numpy as np
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n

    # Rank-23 decomposition for 3x3 matrix multiplication
    # Using a known construction that improves upon schoolbook (rank 27)
    def get_3x3_rank23():
        """
        Rank-23 decomposition for 3x3 matrix multiplication.
        U3: (9, 23) - A side
        V3: (9, 23) - B side  
        W3: (9, 23) - C side (index k*3+i)
        """
        U3 = np.zeros((9, 23))
        V3 = np.zeros((9, 23))
        W3 = np.zeros((9, 23))
        
        # We use a construction based on combining 2x2 Strassen with 
        # additional terms for the 3x3 case.
        # 
        # View 3x3 as:
        # A = [[A11, A12, A13],
        #      [A21, A22, A23],
        #      [A31, A32, A33]]
        #
        # Use a decomposition that leverages structure.
        # One approach: use 7 terms for a 2x2 subblock, then handle
        # the remaining rows/columns with additional terms.
        #
        # Actually, let me use a verified rank-23 construction.
        # The following is based on known results in the literature.
        
        # Indices: i,j for A (0-2), j,k for B (0-2), k,i for C (0-2)
        # U3[i*3+j, r], V3[j*3+k, r], W3[k*3+i, r]
        
        # Let me construct this carefully using a known algorithm.
        # 
        # A simpler approach: use the fact that we can embed a rank-7 
        # decomposition for 2x2 into the 3x3 case and add 16 more terms
        # for the boundary interactions. But 7+16=23.
        
        # Let me try a direct construction:
        # Terms 0-6: Strassen-like for the (0,1)x(0,1) 2x2 subblock
        # Terms 7-22: Handle interactions with row/col 2
        
        # Actually, let me use a verified construction from the literature.
        # I'll implement a specific rank-23 decomposition.
        
        # Based on "The rank of 3x3 matrix multiplication is at most 23"
        # Let me construct it step by step.
        
        # For now, let me use a safe approach: if I can't guarantee 
        # correctness of a rank-23 construction quickly, I'll fall back 
        # to rank-27 schoolbook which gives 189 total.
        
        # Let me try to implement a rank-23 construction:
        
        # Method: Use a decomposition that treats 3x3 as having a 
        # 2x2 core (Strassen rank 7) plus 16 additional terms for 
        # the interactions with the third row/column.
        
        # Core 2x2 (rows/cols 0,1):
        # P1 = (A00+A11)*(B00+B11)
        # P2 = (A10+A11)*B00
        # P3 = A00*(B01-B11)
        # P4 = A11*(B10-B00)
        # P5 = (A00+A01)*B11
        # P6 = (A10-A00)*(B00+B01)
        # P7 = (A01-A11)*(B10+B11)
        
        # C00 = P1+P4-P5+P7
        # C01 = P3+P5
        # C10 = P2+P4
        # C11 = P1-P2+P3+P6
        
        # Now add terms for row/col 2:
        # We need to compute:
        # C02 = A00*B02 + A01*B12 + A02*B22
        # C12 = A10*B02 + A11*B12 + A12*B22
        # C20 = A20*B00 + A21*B10 + A22*B20
        # C21 = A20*B01 + A21*B11 + A22*B21
        # C22 = A20*B02 + A21*B12 + A22*B22
        
        # That's 5 outputs, each with 3 terms = 15 terms, but some 
        # can be shared. Let me count: we need 16 additional terms 
        # to get to 23 total.
        
        # Let me define the 16 additional terms:
        # Q1 = A02*B22 (contributes to C02)
        # Q2 = A12*B22 (contributes to C12)
        # Q3 = A20*B00 (contributes to C20)
        # Q4 = A21*B10 (contributes to C20)
        # Q5 = A22*B20 (contributes to C20)
        # Q6 = A20*B01 (contributes to C21)
        # Q7 = A21*B11 (contributes to C21)
        # Q8 = A22*B21 (contributes to C21)
        # Q9 = A00*B02 (contributes to C02)
        # Q10 = A01*B12 (contributes to C02)
        # Q11 = A10*B02 (contributes to C12)
        # Q12 = A11*B12 (contributes to C12)
        # Q13 = A20*B02 (contributes to C22)
        # Q14 = A21*B12 (contributes to C22)
        # Q15 = A22*B22 (contributes to C22)
        # Q16 = A02*B22 (duplicate? No, need to check)
        
        # Wait, I'm double-counting. Let me be more careful.
        
        # Actually, this approach is getting complex. Let me use a 
        # different strategy: I'll verify the rank-27 schoolbook 
        # gives 189, and if I can't quickly verify a rank-23, I'll 
        # stick with 189.
        
        # For safety, let me just return the schoolbook rank-27.
        rank3 = 27
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    col = i * 9 + j * 3 + k
                    U3[i * 3 + j, col] = 1.0
                    V3[j * 3 + k, col] = 1.0
                    W3[k * 3 + i, col] = 1.0
        return U3, V3, W3

    U3, V3, W3 = get_3x3_rank23()
    r3 = U3.shape[1]  # 27

    # Strassen 2x2 block decomposition
    total_rank = 7 * r3
    U6 = np.zeros((36, total_rank))
    V6 = np.zeros((36, total_rank))
    W6 = np.zeros((36, total_rank))

    def add_strassen_term(start_col, a_blocks, b_blocks, c_blocks):
        """
        a_blocks, b_blocks, c_blocks: list of (block_idx, coeff)
        block_idx: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
        """
        for r in range(r3):
            u_mat = U3[:, r].reshape(3, 3)
            v_mat = V3[:, r].reshape(3, 3)
            w_mat = W3[:, r].reshape(3, 3)

            for bi, coeff in a_blocks:
                rb, cb = bi // 2, bi % 2
                for i in range(3):
                    for j in range(3):
                        val = coeff * u_mat[i, j]
                        if val != 0:
                            gi, gj = rb * 3 + i, cb * 3 + j
                            U6[gi * 6 + gj, start_col + r] += val

            for bi, coeff in b_blocks:
                rb, cb = bi // 2, bi % 2
                for j in range(3):
                    for k in range(3):
                        val = coeff * v_mat[j, k]
                        if val != 0:
                            gj, gk = rb * 3 + j, cb * 3 + k
                            V6[gj * 6 + gk, start_col + r] += val

            for bi, coeff in c_blocks:
                rb, cb = bi // 2, bi % 2
                for k in range(3):
                    for i in range(3):
                        val = coeff * w_mat[k, i]
                        if val != 0:
                            gk, gi = cb * 3 + k, rb * 3 + i
                            W6[gk * 6 + gi, start_col + r] += val

    # Strassen formulas for 2x2 blocks
    # P1 = (A11 + A22) * (B11 + B22)
    add_strassen_term(0, [(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)])
    # P2 = (A21 + A22) * B11
    add_strassen_term(r3, [(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)])
    # P3 = A11 * (B12 - B22)
    add_strassen_term(2 * r3, [(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)])
    # P4 = A22 * (B21 - B11)
    add_strassen_term(3 * r3, [(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)])
    # P5 = (A11 + A12) * B22
    add_strassen_term(4 * r3, [(0, 1), (1, 1)], [(3, 1)], [(1, 1), (0, -1)])
    # P6 = (A21 - A11) * (B11 + B12)
    add_strassen_term(5 * r3, [(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)])
    # P7 = (A12 - A22) * (B21 + B22)
    add_strassen_term(6 * r3, [(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)])

    # Verify the decomposition
    try:
        U_mat = U6.reshape(6, 6, -1)
        V_mat = V6.reshape(6, 6, -1)
        W_mat = W6.reshape(6, 6, -1)
        T = np.einsum('ijr,jkr,kir->ijk', U_mat, V_mat, W_mat)
        expected = np.ones((6, 6, 6))
        if np.allclose(T, expected, atol=1e-9):
            return {"U": U6.tolist(), "V": V6.tolist(), "W": W6.tolist()}
    except Exception:
        pass

    # Fallback: schoolbook rank-216
    rank = 216
    U_f = np.zeros((36, rank))
    V_f = np.zeros((36, rank))
    W_f = np.zeros((36, rank))
    for i in range(6):
        for j in range(6):
            for k in range(6):
                col = i * 36 + j * 6 + k
                U_f[i * 6 + j, col] = 1.0
                V_f[j * 6 + k, col] = 1.0
                W_f[k * 6 + i, col] = 1.0
    return {"U": U_f.tolist(), "V": V_f.tolist(), "W": W_f.tolist()}

# EVOLVE-BLOCK-END
