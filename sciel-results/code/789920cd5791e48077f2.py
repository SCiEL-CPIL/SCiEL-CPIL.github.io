import time
import numpy as np
from itertools import product

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition is correct."""
    rank = len(U[0])
    U_arr = np.array(U)
    V_arr = np.array(V)
    W_arr = np.array(W)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U_arr[i*m+j] * V_arr[j*p+k] * W_arr[k*n+i])
                if abs(s - 1) > tol:
                    return False
    return True

def _strassen_2x2_decomposition():
    """Return a rank-7 Strassen decomposition for 2x2x2 matrix multiplication."""
    # Strassen's 7 products for 2x2x2
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # C = [[ae+bg, af+bh], [ce+dg, cf+dh]]
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-e)
    # m5 = (a+b)h
    # m6 = (c-d)(f+h)
    # m7 = (b-d)(e+g)
    # C11 = m1 - m4 + m5 + m7
    # C12 = m3 + m5
    # C21 = m2 + m4
    # C22 = m1 - m3 + m2 - m6
    
    # U: 4x7 (rows: a,b,c,d)
    # V: 4x7 (rows: e,f,g,h)
    # W: 4x7 (rows: C11,C12,C21,C22)
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # m1 = (a+d)(e+h)
    U[0, 0] = 1; U[1, 0] = 0; U[2, 0] = 0; U[3, 0] = 1
    V[0, 0] = 1; V[1, 0] = 0; V[2, 0] = 0; V[3, 0] = 1
    W[0, 0] = 1; W[1, 0] = 0; W[2, 0] = 0; W[3, 0] = 1
    
    # m2 = (c+d)e
    U[0, 1] = 0; U[1, 1] = 0; U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1; V[1, 1] = 0; V[2, 1] = 0; V[3, 1] = 0
    W[0, 1] = 0; W[1, 1] = 0; W[2, 1] = 1; W[3, 1] = 1
    
    # m3 = a(f-h)
    U[0, 2] = 1; U[1, 2] = 0; U[2, 2] = 0; U[3, 2] = 0
    V[0, 2] = 0; V[1, 2] = 1; V[2, 2] = 0; V[3, 2] = -1
    W[0, 2] = 0; W[1, 2] = 1; W[2, 2] = 0; W[3, 2] = -1
    
    # m4 = d(g-e)
    U[0, 3] = 0; U[1, 3] = 0; U[2, 3] = 0; U[3, 3] = 1
    V[0, 3] = -1; V[1, 3] = 0; V[2, 3] = 1; V[3, 3] = 0
    W[0, 3] = -1; W[1, 3] = 0; W[2, 3] = 1; W[3, 3] = 0
    
    # m5 = (a+b)h
    U[0, 4] = 1; U[1, 4] = 1; U[2, 4] = 0; U[3, 4] = 0
    V[0, 4] = 0; V[1, 4] = 0; V[2, 4] = 0; V[3, 4] = 1
    W[0, 4] = 1; W[1, 4] = 1; W[2, 4] = 0; W[3, 4] = 0
    
    # m6 = (c-d)(f+h)
    U[0, 5] = 0; U[1, 5] = 0; U[2, 5] = 1; U[3, 5] = -1
    V[0, 5] = 0; V[1, 5] = 1; V[2, 5] = 0; V[3, 5] = 1
    W[0, 5] = 0; W[1, 5] = 0; W[2, 5] = 0; W[3, 5] = -1
    
    # m7 = (b-d)(e+g)
    U[0, 6] = 0; U[1, 6] = 1; U[2, 6] = 0; U[3, 6] = -1
    V[0, 6] = 1; V[1, 6] = 0; V[2, 6] = 1; V[3, 6] = 0
    W[0, 6] = 1; W[1, 6] = 0; W[2, 6] = 0; W[3, 6] = 0
    
    U_list = [[float(U[r, c]) for c in range(7)] for r in range(4)]
    V_list = [[float(V[r, c]) for c in range(7)] for r in range(4)]
    W_list = [[float(W[r, c]) for c in range(7)] for r in range(4)]
    
    return U_list, V_list, W_list

def _block_decomposition_3x3x6():
    """
    Build a decomposition for 3x3 * 3x6 using block structure.
    
    We can write:
    A (3x3) * B (3x6) = C (3x6)
    
    Split B into two 3x3 blocks: B = [B1 | B2]
    Then C = [A*B1 | A*B2]
    
    So we need two 3x3x3 multiplications.
    
    Using Strassen's algorithm on 3x3 matrices by padding to 4x4:
    - Pad A (3x3) to A' (4x4) with zeros
    - Pad B1, B2 (3x3) to B1' (4x4) and B2' (4x4) with zeros
    - Use Strassen 4x4 (which uses 7*2^2 = 28 scalar products for 4x4)
    - Actually, 4x4 Strassen uses 7*(2x2 Strassen) = 7*7 = 49... no
    - 4x4 via Strassen: split into four 2x2 blocks, use 7 multiplications of 2x2
    - Each 2x2 uses 7 scalar products, so 7*7 = 49 total
    
    But we only need the 3x3 part of the result.
    
    Alternatively, use a direct rank-19 decomposition for 3x3x3.
    The known rank for 3x3x3 is 19.
    
    Let me try to construct a rank-19 decomposition for 3x3x3.
    
    Actually, let me use a different approach. The tensor rank of 3x3x3 matrix multiplication
    is known to be 19 (over the reals). But constructing it explicitly is complex.
    
    For a simpler approach, let me use the "middle product" decomposition.
    
    For 3x3 * 3x6:
    - Think of it as computing C[i,k] = sum_j A[i,j] * B[j,k] for i=0,1,2 and k=0,1,2,3,4,5
    - This is 18 dot products of length 3
    
    A simple decomposition: rank = 3*3*6 = 54 (schoolbook)
    
    To improve, we can use Strassen on the 3x3 blocks.
    
    Let me try: pad to 4x4, use Strassen.
    A 4x4 matrix can be multiplied using 7*(4/2)^2 = 7*4 = 28 rank-1 terms? No.
    
    Actually, the rank of 4x4x4 matrix multiplication is at most 7^2 = 49 by applying
    Strassen to 2x2 blocks (7 multiplications of 2x2, each of which uses 7 scalar products).
    
    But we want 3x3x6, not 4x4x4.
    
    Let me try a different decomposition. I'll use the fact that:
    3x3 * 3x6 can be viewed as two 3x3 * 3x3 multiplications.
    
    If I can find a rank-r decomposition for 3x3x3, I can get rank-2r for 3x3x6.
    
    The known minimal rank for 3x3x3 is 19, giving 38 for 3x3x6.
    
    Let me try to find a rank-19 decomposition for 3x3x3 using numerical methods.
    """
    
    # Use Strassen on 4x4 padded matrices
    # This gives rank at most 49 for 4x4x4, but we only need 3x3 part
    
    # Let's build a 4x4x4 Strassen decomposition
    U4, V4, W4 = _strassen_2x2_decomposition()  # 2x2x2, rank 7
    
    # Now build 4x4x4 by blocking: split each 4x4 into four 2x2 blocks
    # A = [[A11, A12], [A21, A22]], B = [[B11, B12], [B21, B22]], C = A*B
    # Use Strassen on the 2x2 blocks
    
    # The 4x4x4 tensor can be decomposed using 7 multiplications of 2x2x2
    # Each 2x2x2 uses 7 scalar products
    
    # Let me construct this properly
    # For 4x4, we have indices (i,j) where i,j in {0,1,2,3}
    # Split into blocks: A11 = A[0:2,0:2], A12 = A[0:2,2:4], etc.
    
    # Strassen on 4x4:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A22)(B12+B22)
    # M7 = (A12-A22)(B11+B21)
    # C11 = M1 - M4 + M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M3 + M2 - M6
    
    # Each Mi is a 2x2 product, decomposable into 7 rank-1 terms
    
    U44 = np.zeros((16, 49))  # 4x4 = 16 elements, 49 rank terms
    V44 = np.zeros((16, 49))
    W44 = np.zeros((16, 49))
    
    # Block indices: (block_row, block_col, elem_row, elem_col) -> 4*4*4 = 64? No, 16
    # A[i,j] where i,j in 0..3: index = i*4+j
    # Blocks: A11 = rows 0-1, cols 0-1; A12 = rows 0-1, cols 2-3
    #          A21 = rows 2-3, cols 0-1; A22 = rows 2-3, cols 2-3
    
    def block_index(block_r, block_c, elem_r, elem_c):
        """Map block coordinates to flat index for 4x4 matrix."""
        row = block_r * 2 + elem_r
        col = block_c * 2 + elem_c
        return row * 4 + col
    
    # For each of the 7 Strassen products, add the 7 rank-1 terms from 2x2 Strassen
    for m in range(7):
        # Determine which A-blocks and B-blocks are used in product m
        # M1: A-blocks (0,0) and (1,1), B-blocks (0,0) and (1,1)
        # M2: A-blocks (1,0) and (1,1), B-block (0,0)
        # M3: A-block (0,0), B-blocks (0,1) and (1,1)
        # M4: A-block (1,1), B-blocks (1,0) and (0,0)
        # M5: A-blocks (0,0) and (0,1), B-block (1,1)
        # M6: A-blocks (1,0) and (1,1), B-blocks (0,1) and (1,1)
        # M7: A-blocks (0,1) and (1,1), B-blocks (0,0) and (1,0)
        
        if m == 0:  # M1 = (A11+A22)(B11+B22)
            a_blocks = [(0,0,1), (1,1,1)]
            b_blocks = [(0,0,1), (1,1,1)]
            c_blocks = [(0,0,1), (0,1,1), (1,0,1), (1,1,1)]  # all four C blocks contribute
            c_coeffs = [(0,0,1), (0,1,1), (1,0,1), (1,1,1)]
            # C11 += M1, C12 += M1, C21 += M1, C22 += M1
        elif m == 1:  # M2 = (A21+A22)B11
            a_blocks = [(1,0,1), (1,1,1)]
            b_blocks = [(0,0,1)]
            c_blocks = [(1,0,1), (1,1,1)]
        elif m == 2:  # M3 = A11(B12-B22)
            a_blocks = [(0,0,1)]
            b_blocks = [(0,1,1), (1,1,-1)]
            c_blocks = [(0,1,1), (1,1,-1)]
        elif m == 3:  # M4 = A22(B21-B11)
            a_blocks = [(1,1,1)]
            b_blocks = [(1,0,1), (0,0,-1)]
            c_blocks = [(0,0,-1), (1,0,1)]
        elif m == 4:  # M5 = (A11+A12)B22
            a_blocks = [(0,0,1), (0,1,1)]
            b_blocks = [(1,1,1)]
            c_blocks = [(0,0,1), (0,1,1)]
        elif m == 5:  # M6 = (A21-A22)(B12+B22)
            a_blocks = [(1,0,1), (1,1,-1)]
            b_blocks = [(0,1,1), (1,1,1)]
            c_blocks = [(1,1,-1)]
        elif m == 6:  # M7 = (A12-A22)(B11+B21)
            a_blocks = [(0,1,1), (1,1,-1)]
            b_blocks = [(0,0,1), (1,0,1)]
            c_blocks = [(0,0,1)]
        
        # For each rank-1 term in the 2x2 decomposition
        for t in range(7):
            idx = m * 7 + t
            # U and V contributions from the 2x2 Strassen decomposition
            u2 = U4[t]  # 4 values for the 2x2 A-block
            v2 = V4[t]  # 4 values for the 2x2 B-block
            w2 = W4[t]  # 4 values for the 2x2 C-block
            
            # Distribute to 4x4
            for br, bc, coeff in a_blocks:
                for er in range(2):
                    for ec in range(2):
                        flat_idx = block_index(br, bc, er, ec)
                        U44[flat_idx, idx] += coeff * u2[er * 2 + ec]
            
            for br, bc, coeff in b_blocks:
                for er in range(2):
                    for ec in range(2):
                        flat_idx = block_index(br, bc, er, ec)
                        V44[flat_idx, idx] += coeff * v2[er * 2 + ec]
            
            for br, bc, coeff in c_blocks:
                for er in range(2):
                    for ec in range(2):
                        flat_idx = block_index(br, bc, er, ec)
                        W44[flat_idx, idx] += coeff * w2[er * 2 + ec]
    
    # Now extract the 3x3x3 part (drop row/col 3)
    # A: 3x3 -> indices where i,j in 0,1,2 -> flat indices 0,1,2,4,5,6,8,9,10
    keep_A = [i * 3 + j for i in range(3) for j in range(3)]
    
    # B: 3x3 for each half
    # For B1 (cols 0-2 of the 3x6): same keep pattern
    # For B2 (cols 3-5): same
    
    # C: 3x6 -> indices (i,k) where i in 0,1,2 and k in 0,1,2,3,4,5
    # W indexing: k*n + i = k*3 + i
    
    # Actually, for 3x3 * 3x6, let me think differently.
    # Pad A (3x3) to A' (4x4) with row/col 3 zeroed.
    # Pad B (3x6) to B' (4x8)? No, let's split into two 3x3 and pad each to 4x4.
    
    # This is getting complex. Let me just use the 4x4x4 decomposition and 
    # restrict to the 3x3x3 parts for two blocks of B.
    
    # For 3x3 * 3x6:
    # C[i,k] = sum_j A[i,j] * B[j,k] for i=0,1,2, k=0,1,2,3,4,5
    # Split B into B1 (3x3, cols 0-2) and B2 (3x3, cols 3-5)
    # C1 = A * B1, C2 = A * B2
    # Use the 4x4x4 decomposition, take the 3x3x3 part for each
    
    # U: 3x3 = 9 rows
    U3 = np.zeros((9, 49))
    for old_i in range(3):
        for old_j in range(3):
            new_i = old_i
            new_j = old_j
            U3[old_i * 3 + old_j, :] = U44[new_i * 4 + new_j, :]
    
    # V: 3x6 = 18 rows
    # For B1 (cols 0-2): V1[j,k] = V44[j*4+k, :] for j,k in 0,1,2
    # For B2 (cols 3-5): V2[j,k] = V44[j*4+k, :] for j,k in 0,1,2
    V18 = np.zeros((18, 49))
    for j in range(3):
        for k in range(3):
            V18[j * 3 + k, :] = V44[j * 4 + k, :]  # B1: cols 0-2
            V18[j * 3 + 3 + k, :] = V44[j * 4 + k, :]  # B2: cols 3-5
    
    # W: 6x3 = 18 rows (index: k*3 + i)
    # For C1 (cols 0-2): W1[k,i] = W44[k*4+i, :] for k,i in 0,1,2
    # For C2 (cols 3-5): W2[k,i] = W44[k*4+i, :] for k,i in 0,1,2
    W18 = np.zeros((18, 49))
    for k in range(3):
        for i in range(3):
            W18[k * 3 + i, :] = W44[k * 4 + i, :]  # C1: cols 0-2
            W18[(3 + k) * 3 + i, :] = W44[k * 4 + i, :]  # C2: cols 3-5
    
    U_list = [[float(U3[r, c]) for c in range(49)] for r in range(9)]
    V_list = [[float(V18[r, c]) for c in range(49)] for r in range(18)]
    W_list = [[float(W18[r, c]) for c in range(49)] for r in range(18)]
    
    # Verify
    if _verify_decomposition(U_list, V_list, W_list, 3, 3, 6):
        return U_list, V_list, W_list
    
    return None

def _schoolbook_decomposition(n, m, p):
    """Return schoolbook rank n*m*p decomposition."""
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return U, V, W

def construct():
    """
    Construct an exact tensor decomposition for 3x3 * 3x6 matrix multiplication.
    
    Strategy: 
    1. Try block Strassen-based decomposition (rank ~49)
    2. Fall back to schoolbook (rank 54) if that fails
    """
    n, m, p = 3, 3, 6
    start_time = time.time()
    
    # Try the block decomposition
    result = _block_decomposition_3x3x6()
    if result is not None:
        U, V, W = result
        rank = len(U[0])
        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    U, V, W = _schoolbook_decomposition(n, m, p)
    return {"U": U, "V": V, "W": W}

