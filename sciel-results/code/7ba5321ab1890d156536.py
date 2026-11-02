import time
import numpy as np

START_TIME = time.time()
TIME_LIMIT = 140

def time_up():
    return (time.time() - START_TIME) > TIME_LIMIT

def construct():
    """
    Construct a rank-minimal (rank 28) tensor decomposition for 2x4 @ 4x4 matrix multiplication.
    
    The tensor T has indices (i, j, k) where i in {0,1}, j in {0,1,2,3}, k in {0,1,2,3}.
    T[i,j,k] = 1 for all valid (i,j,k).
    
    Decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    where n=2, m=4, p=4, rank=28.
    
    We use the block-Strassen approach:
    - Split the 2x4 matrix into two 2x2 blocks (columns 0-1 and 2-3)
    - Split the 4x4 matrix into four 2x2 blocks
    - Apply Strassen's 7-multiplication algorithm to each combination of blocks
    
    This gives 4 block combinations * 7 Strassen steps = 28 rank-1 terms.
    """
    n, m, p = 2, 4, 4
    rank = 28
    
    U = np.zeros((n * m, rank), dtype=np.float64)  # 8 x 28
    V = np.zeros((m * p, rank), dtype=np.float64)  # 16 x 28
    W = np.zeros((p * n, rank), dtype=np.float64)  # 8 x 28
    
    # Strassen's 7 products for 2x2 matrix multiplication:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B10 - B11)  -- wait, let me use the standard form
    # Actually, standard Strassen:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B10 - B11)
    # M4 = A11(B01 - B00)  -- no, let me be more careful
    #
    # Standard Strassen (7 products):
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B10-B11)
    # M4 = A11*(B01-B00)  -- actually this should be A11*(B01-B00) but let me use the conventional form
    #
    # Let me use the well-known form:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # where:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)  -- wait, this doesn't match. Let me use the standard reference.
    #
    # Standard Strassen 7 products:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B10 - B11)
    # M4 = A11 * (B01 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    #
    # Then:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For each of the 4 block combinations (block_row, block_col):
    # block_row determines which 2x2 block of A (row 0 or 1 of the 2x4 matrix -> columns 0-1 or 2-3)
    # block_col determines which 2x2 block of B (row/col of the 2x2 blocks in the 4x4)
    
    # Block layout for A (2x4):
    # Block 0: A[:, 0:2] -> 2x2, indices in U: rows 0-3 (i*4 + j for i=0,1 and j=0,1)
    # Block 1: A[:, 2:4] -> 2x2, indices in U: rows 4-7 (i*4 + j for i=0,1 and j=2,3)
    
    # Block layout for B (4x4):
    # Block (0,0): B[0:2, 0:2] -> V rows: 0*4+0, 0*4+1, 1*4+0, 1*4+1 = 0,1,4,5
    # Block (0,1): B[0:2, 2:4] -> V rows: 0*4+2, 0*4+3, 1*4+2, 1*4+3 = 2,3,6,7
    # Block (1,0): B[2:4, 0:2] -> V rows: 2*4+0, 2*4+1, 3*4+0, 3*4+1 = 8,9,12,13
    # Block (1,1): B[2:4, 2:4] -> V rows: 2*4+2, 2*4+3, 3*4+2, 3*4+3 = 10,11,14,15
    
    # For W (8x28), index is k*n+i = k*2+i where k is the column of C (0-3) and i is the row of C (0-1)
    # Block row 0 of C: k=0,1 -> W rows: 0*2+0=0, 1*2+0=2, 0*2+1=1, 1*2+1=3
    # Block row 1 of C: k=2,3 -> W rows: 2*2+0=4, 3*2+0=6, 2*2+1=5, 3*2+1=7
    
    # Strassen U coefficients for A block (a00, a01, a10, a11):
    # M1: (a00+a11) -> [1, 0, 0, 1]
    # M2: (a10+a11) -> [0, 0, 1, 1]
    # M3: a00 -> [1, 0, 0, 0]
    # M4: a11 -> [0, 0, 0, 1]
    # M5: (a00+a01) -> [1, 1, 0, 0]
    # M6: (a10-a00) -> [-1, 0, 1, 0]
    # M7: (a01-a11) -> [0, 1, 0, -1]
    
    U_coeffs = [
        [1, 0, 0, 1],   # M1
        [0, 0, 1, 1],   # M2
        [1, 0, 0, 0],   # M3
        [0, 0, 0, 1],   # M4
        [1, 1, 0, 0],   # M5
        [-1, 0, 1, 0],  # M6
        [0, 1, 0, -1]   # M7
    ]
    
    # Strassen V coefficients for B block (b00, b01, b10, b11):
    # M1: (b00+b11) -> [1, 0, 0, 1]
    # M2: b00 -> [1, 0, 0, 0]
    # M3: (b10-b11) -> [0, 0, 1, -1]
    # M4: (b01-b00) -> [-1, 1, 0, 0]
    # M5: b11 -> [0, 0, 0, 1]
    # M6: (b00+b01) -> [1, 1, 0, 0]
    # M7: (b10+b11) -> [0, 0, 1, 1]
    
    V_coeffs = [
        [1, 0, 0, 1],   # M1
        [1, 0, 0, 0],   # M2
        [0, 0, 1, -1],  # M3
        [-1, 1, 0, 0],  # M4
        [0, 0, 0, 1],   # M5
        [1, 1, 0, 0],   # M6
        [0, 0, 1, 1]    # M7
    ]
    
    # Strassen W coefficients for C block (c00, c01, c10, c11):
    # C00 = M1 + M4 - M5 + M7 -> [1, 0, 0, 1] for (M1,M2,M3,M4) ... wait, I need per-product
    # For each product Ms, what's its coefficient in C00, C01, C10, C11?
    # C00: M1:+1, M2:0, M3:0, M4:+1, M5:-1, M6:0, M7:+1
    # C01: M1:0, M2:0, M3:+1, M4:0, M5:+1, M6:0, M7:0
    # C10: M1:0, M2:+1, M3:0, M4:+1, M5:0, M6:0, M7:0
    # C11: M1:+1, M2:-1, M3:+1, M4:0, M5:0, M6:+1, M7:0
    
    W_coeffs = [
        [1, 0, 0, 1],   # M1: C00=1, C01=0, C10=0, C11=1
        [0, 0, 1, -1],  # M2: C00=0, C01=0, C10=1, C11=-1
        [0, 1, 0, 1],   # M3: C00=0, C01=1, C10=0, C11=1
        [1, 0, 1, 0],   # M4: C00=1, C01=0, C10=1, C11=0
        [-1, 1, 0, 0],  # M5: C00=-1, C01=1, C10=0, C11=0
        [0, 0, 0, 1],   # M6: C00=0, C01=0, C10=0, C11=1
        [1, 0, 0, 0]    # M7: C00=1, C01=0, C10=0, C11=0
    ]
    
    # For each block combination (a_block, b_block) and Strassen step s:
    for a_block in range(2):  # 0 or 1: which 2x2 block of A
        for b_row in range(2):  # 0 or 1: row of B block
            for b_col in range(2):  # 0 or 1: col of B block
                for s in range(7):
                    r = (a_block * 4 + b_row * 2 + b_col) * 7 + s
                    
                    # U: A block (a_block)
                    # A block 0: rows 0-3 of U (i*4 + j for i=0,1, j=0,1)
                    # A block 1: rows 4-7 of U (i*4 + j for i=0,1, j=2,3)
                    # Within block, local indices: a00=(i=0,j=0), a01=(i=0,j=1), a10=(i=1,j=0), a11=(i=1,j=1)
                    for local_idx, coeff in enumerate(U_coeffs[s]):
                        if coeff != 0:
                            # local_idx: 0=a00, 1=a01, 2=a10, 3=a11
                            li, lj = divmod(local_idx, 2)
                            u_row = a_block * 4 + li * 2 + lj
                            U[u_row, r] += coeff
                    
                    # V: B block (b_row, b_col)
                    # B block (br, bc): local b00=(i=0,j=0), b01=(i=0,j=1), b10=(i=1,j=0), b11=(i=1,j=1)
                    # Global V index: (b_row*2 + i) * p + (b_col*2 + j) = (b_row*2+i)*4 + (b_col*2+j)
                    for local_idx, coeff in enumerate(V_coeffs[s]):
                        if coeff != 0:
                            li, lj = divmod(local_idx, 2)
                            v_row = (b_row * 2 + li) * p + (b_col * 2 + lj)
                            V[v_row, r] += coeff
                    
                    # W: C block row (b_row) -- the output row block matches the B row block
                    # C block (b_row, b_col): local c00=(i=0,j=0), c01=(i=0,j=1), c10=(i=1,j=0), c11=(i=1,j=1)
                    # Global W index: k*n + i where k = b_col*2 + local_j, i = local_i
                    for local_idx, coeff in enumerate(W_coeffs[s]):
                        if coeff != 0:
                            li, lj = divmod(local_idx, 2)
                            k_global = b_col * 2 + lj
                            i_global = li
                            w_row = k_global * n + i_global
                            W[w_row, r] += coeff
    
    return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}

if __name__ == "__main__":
    result = construct()
    n, m, p = 2, 4, 4
    U = np.array(result['U'], dtype=float)
    V = np.array(result['V'], dtype=float)
    W = np.array(result['W'], dtype=float)
    rank = U.shape[1]
    
    T_recon = np.zeros((n, m, p), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_recon[i, j, k] = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
    
    T_target = np.ones((n, m, p), dtype=float)
    max_err = np.max(np.abs(T_recon - T_target))
    print(f"Rank: {rank}, Max error: {max_err}, Valid: {max_err < 1e-9}")
