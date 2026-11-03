"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START

def _strassen_4x4(U_rows, V_cols, W_rows):
    """
    Return rank-25 Strassen decomposition for 4x4 * 4x4.
    U is a dict mapping (i,j) -> column index in U (for row i, col j of A)
    V is a dict mapping (j,k) -> column index in V (for row j, col k of B)
    W is a dict mapping (k,i) -> column index in W (for row k, col i of C)
    Actually we just return the three lists of entries.
    
    For 4x4*4x4 with Strassen (2 levels of 2x2):
    A = [[A11, A12], [A21, A22]] where each Ai is 2x2
    B = [[B11, B12], [B21, B22]] where each Bi is 2x2
    
    M1 = (A11+A22)(B11+B22)
    M2 = (A21+A22)B11
    M3 = A11(B12-B22)
    M4 = A22(B21-B11)
    M5 = (A11+A12)B22
    M6 = (A21-A11)(B11+B12)
    M7 = (A12-A22)(B21+B22)
    
    C11 = M1+M4-M5+M7
    C12 = M3+M5
    C21 = M2+M4
    C22 = M1-M2+M3+M6
    
    Each Mi is a 2x2*2x2 product, which needs rank 7 (Strassen) or rank 8 (schoolbook).
    Using Strassen for each: total rank = 7 * 7 = 49.
    
    But we can do better by noting the structure. Let's use rank-7 Strassen for each 2x2 block.
    
    Actually, let's just build it directly. For each of the 7 M terms, we need a rank-7 
    decomposition of 2x2*2x2.
    """
    pass

def _strassen_2x2_entries():
    """
    Return the Strassen rank-7 decomposition for 2x2 * 2x2.
    A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    M1 = (a+d)(e+h)
    M2 = (c+d)e
    M3 = a(f-h)
    M4 = d(g-e)
    M5 = (a+b)h
    M6 = (c-a)(e+f)
    M7 = (b-d)(g+h)
    C11 = M1+M4-M5+M7
    C12 = M3+M5
    C21 = M2+M4
    C22 = M1-M2+M3+M6
    
    U entries (rows of A: a=0, b=1, c=2, d=3):
    M1: a=1, d=1
    M2: c=1, d=1
    M3: a=1
    M4: d=1
    M5: a=1, b=1
    M6: c=1, a=-1
    M7: b=1, d=-1
    
    V entries (cols of B: e=0, f=1, g=2, h=3):
    M1: e=1, h=1
    M2: e=1
    M3: f=1, h=-1
    M4: g=1, e=-1
    M5: h=1
    M6: e=1, f=1
    M7: g=1, h=1
    
    W entries (rows of C: C11=0, C12=1, C21=2, C22=3):
    M1: C11=1, C22=1
    M2: C21=1, C22=-1
    M3: C12=1, C22=1
    M4: C11=1, C21=1
    M5: C11=-1, C12=1
    M6: C22=1
    M7: C11=1
    """
    # U: 4 rows (a,b,c,d), 7 cols (M1..M7)
    U_2x2 = [
        [1, 0, 1, 0, 1, -1, 0],  # a
        [0, 0, 0, 0, 1, 0, 1],   # b
        [0, 1, 0, 0, 0, 1, 0],   # c
        [1, 1, 0, 1, 0, 0, -1],  # d
    ]
    # V: 4 rows (e,f,g,h), 7 cols
    V_2x2 = [
        [1, 1, 0, -1, 0, 1, 0],  # e
        [0, 0, 1, 0, 0, 1, 0],   # f
        [0, 0, 0, 1, 0, 0, 1],   # g
        [1, 0, -1, 0, 1, 0, 1],  # h
    ]
    # W: 4 rows (C11,C12,C21,C22), 7 cols
    W_2x2 = [
        [1, 0, 0, 1, -1, 0, 1],  # C11
        [0, 0, 1, 0, 1, 0, 0],   # C12
        [0, 1, 0, 1, 0, 0, 0],   # C21
        [1, -1, 1, 0, 0, 1, 0],  # C22
    ]
    return U_2x2, V_2x2, W_2x2

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 2x4 * 4x8 matrix multiplication.
    
    Strategy: Pad A to 4x4 (add 2 zero rows), pad B to 8x8 (add 4 zero columns).
    Then compute 4x4 * 8x8 using block decomposition.
    
    Actually, let's think more carefully. We have A (2x4) and B (4x8).
    C = A * B is 2x8.
    
    We can view this as:
    A = [[A1, A2]] where A1, A2 are 2x2
    B = [[B1, B2, B3, B4]] where Bi are 2x2
    
    C = [A1*B1 + A2*B3, A1*B2 + A2*B4]
    
    Wait, that's not right. Let me be more careful with indexing.
    
    A is 2x4: rows i=0,1; cols j=0,1,2,3
    B is 4x8: rows j=0,1,2,3; cols k=0,1,2,3,4,5,6,7
    C is 2x8: rows i=0,1; cols k=0,1,2,3,4,5,6,7
    
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    Let's split A into two 2x2 blocks: A = [A_left | A_right] where A_left = A[:, 0:2], A_right = A[:, 2:4]
    And B into four 2x2 blocks: B = [B_0, B_1, B_2, B_3] where B_t = B[:, 2t:2t+2]
    
    C = A_left * [B_0, B_1] + A_right * [B_2, B_3]
       = [A_left*B_0 + A_right*B_2, A_left*B_1 + A_right*B_3]
    
    Each term A_left*B_t is a 2x2 * 2x2 product, which has rank 7 (Strassen).
    We have 4 such products, so total rank = 4 * 7 = 28.
    
    But we can do better by sharing. Let's use the 4x4 * 8x8 approach.
    
    Pad A to 4x4: A_pad = [[A, 0], [0, 0]] where 0 is 2x4
    Pad B to 8x8: B_pad = [[B], [0]] where 0 is 4x8
    
    Then A_pad * B_pad = [[A*B, 0], [0, 0]]
    
    The top-left 2x8 block is A*B.
    
    Now compute 4x4 * 8x8 using Strassen-like recursion.
    Split 4x4 into 2x2 blocks: A_pad = [[A11, A12], [A21, A22]] where each is 2x2
    A11 = A[:, 0:2], A12 = A[:, 2:4], A21 = zeros(2,2), A22 = zeros(2,2)
    
    Split 8x8 into 4x4 blocks... this is getting complex.
    
    Let me just use the direct 4 * 2x2*2x2 approach with rank 28, then see if we can reduce.
    """
    n, m, p = DIMENSIONS  # 2, 4, 8
    
    # Build rank-28 decomposition using 4 Strassen 2x2*2x2 blocks
    U_2x2, V_2x2, W_2x2 = _strassen_2x2_entries()
    
    rank = 28
    # U: n*m = 8 rows
    U = [[0] * rank for _ in range(n * m)]
    # V: m*p = 32 rows
    V = [[0] * rank for _ in range(m * p)]
    # W: p*n = 16 rows
    W = [[0] * rank for _ in range(p * n)]
    
    # A has rows i=0,1 and cols j=0,1,2,3
    # B has rows j=0,1,2,3 and cols k=0,1,2,3,4,5,6,7
    # C has rows i=0,1 and cols k=0,1,2,3,4,5,6,7
    
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k] + A[i,3]*B[3,k]
    
    # Group by 2x2 blocks:
    # For k in 0,1: C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k] + A[i,3]*B[3,k]
    #   = (A[i,0]*B[0,k] + A[i,1]*B[1,k]) + (A[i,2]*B[2,k] + A[i,3]*B[3,k])
    #   = (A_left * B_0)[i,k] + (A_right * B_2)[i,k]
    # where A_left = A[:, 0:2], A_right = A[:, 2:4], B_0 = B[:, 0:2], B_2 = B[:, 4:6]
    
    # Wait, I need to be more careful.
    # A_left is 2x2: rows i=0,1; cols j=0,1
    # A_right is 2x2: rows i=0,1; cols j=2,3
    # B_0 is 2x2: rows j=0,1; cols k=0,1
    # B_1 is 2x2: rows j=0,1; cols k=2,3
    # B_2 is 2x2: rows j=2,3; cols k=0,1
    # B_3 is 2x2: rows j=2,3; cols k=2,3
    
    # C[:, 0:2] = A_left * B_0 + A_right * B_2
    # C[:, 2:4] = A_left * B_1 + A_right * B_3
    # C[:, 4:6] = A_left * B_4 + A_right * B_6  (but B only has 4 rows, so B_4 doesn't exist)
    
    # I made an error. B is 4x8, so it has 4 rows and 8 columns.
    # Let me re-split:
    # A: 2x4, split into A_left (2x2, cols 0-1) and A_right (2x2, cols 2-3)
    # B: 4x8, split into 4 blocks of 2x2: B_0 (rows 0-1, cols 0-1), B_1 (rows 0-1, cols 2-3),
    #   B_2 (rows 0-1, cols 4-5), B_3 (rows 0-1, cols 6-7),
    #   B_4 (rows 2-3, cols 0-1), B_5 (rows 2-3, cols 2-3),
    #   B_6 (rows 2-3, cols 4-5), B_7 (rows 2-3, cols 6-7)
    
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k] + A[i,3]*B[3,k]
    # For k in 0-1: C[i,k] = (A[i,0]*B[0,k] + A[i,1]*B[1,k]) + (A[i,2]*B[2,k] + A[i,3]*B[3,k])
    #   = (A_left * B_0)[i,k] + (A_right * B_4)[i,k]
    # For k in 2-3: C[i,k] = (A_left * B_1)[i,k] + (A_right * B_5)[i,k]
    # For k in 4-5: C[i,k] = (A_left * B_2)[i,k] + (A_right * B_6)[i,k]
    # For k in 6-7: C[i,k] = (A_left * B_3)[i,k] + (A_right * B_7)[i,k]
    
    # So C = [A_left*B_0 + A_right*B_4, A_left*B_1 + A_right*B_5, A_left*B_2 + A_right*B_6, A_left*B_3 + A_right*B_7]
    
    # Each 2x2*2x2 product has rank 7. We have 8 such products, total rank = 8*7 = 56.
    
    # Let's build this.
    U = [[0] * 56 for _ in range(8)]   # n*m = 2*4 = 8
    V = [[0] * 56 for _ in range(32)]  # m*p = 4*8 = 32
    W = [[0] * 56 for _ in range(16)]  # p*n = 8*2 = 16
    
    # Strassen 2x2*2x2 entries
    U_s = np.array(U_2x2, dtype=float)  # 4x7
    V_s = np.array(V_2x2, dtype=float)  # 4x7
    W_s = np.array(W_2x2, dtype=float)  # 4x7
    
    for block_idx in range(8):
        # block_idx 0-3: A_left * B_{block_idx}
        # block_idx 4-7: A_right * B_{block_idx-4}
        col_offset = block_idx * 7
        
        if block_idx < 4:
            # A_left: rows i=0,1; cols j=0,1
            # B_{block_idx}: rows j=0,1; cols k=2*block_idx to 2*block_idx+1
            for r in range(4):  # A rows (i*2+j)
                for c in range(7):
                    U[r][col_offset + c] = U_s[r][c]
            for r in range(4):  # B rows (j*4+k_local) where j=0,1
                j = r // 2
                k_local = r % 2
                k = 2 * block_idx + k_local
                for c in range(7):
                    V[j * 8 + k][col_offset + c] = V_s[r][c]
            # W: C rows (i*8+k) where i=0,1; k=2*block_idx to 2*block_idx+1
            for r in range(4):  # C rows (i*2+k_local)
                i = r // 2
                k_local = r % 2
                k = 2 * block_idx + k_local
                for c in range(7):
                    W[k * 2 + i][col_offset + c] = W_s[r][c]
        else:
            # A_right: rows i=0,1; cols j=2,3
            # B_{block_idx-4}: rows j=2,3; cols k=2*(block_idx-4) to 2*(block_idx-4)+1
            for r in range(4):  # A rows (i*2+j) where j=2,3
                i = r // 2
                j = 2 + (r % 2)
                for c in range(7):
                    U[i * 4 + j][col_offset + c] = U_s[r][c]
            for r in range(4):  # B rows (j*4+k_local) where j=2,3
                j = 2 + (r // 2)
                k_local = r % 2
                k = 2 * (block_idx - 4) + k_local
                for c in range(7):
                    V[j * 8 + k][col_offset + c] = V_s[r][c]
            # W: C rows (i*8+k) where i=0,1; k=2*(block_idx-4) to 2*(block_idx-4)+1
            for r in range(4):  # C rows (i*2+k_local)
                i = r // 2
                k_local = r % 2
                k = 2 * (block_idx - 4) + k_local
                for c in range(7):
                    W[k * 2 + i][col_offset + c] = W_s[r][c]
    
    # Verify
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Test with random matrices
    np.random.seed(42)
    A_test = np.random.randn(2, 4)
    B_test = np.random.randn(4, 8)
    C_expected = A_test @ B_test
    
    # Compute C from decomposition
    C_computed = np.zeros((2, 8))
    for col in range(56):
        u_vec = U_arr[:, col]
        v_vec = V_arr[:, col]
        w_vec = W_arr[:, col]
        # u_vec is 8-vector: A[i*m+j] = A[i*4+j]
        A_sub = u_vec.reshape(2, 4)
        # v_vec is 32-vector: B[j*p+k] = B[j*8+k]
        B_sub = v_vec.reshape(4, 8)
        # w_vec is 16-vector: C[k*n+i] = C[k*2+i]
        C_sub = w_vec.reshape(8, 2).T  # shape (2,8)
        C_computed += A_sub @ B_sub * C_sub
    
    if not np.allclose(C_computed, C_expected, atol=1e-10):
        # Fallback to schoolbook
        rank = 64
        U = [[0] * rank for _ in range(8)]
        V = [[0] * rank for _ in range(32)]
        W = [[0] * rank for _ in range(16)]
        col = 0
        for i in range(2):
            for j in range(4):
                for k in range(8):
                    U[i * 4 + j][col] = 1
                    V[j * 8 + k][col] = 1
                    W[k * 2 + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
