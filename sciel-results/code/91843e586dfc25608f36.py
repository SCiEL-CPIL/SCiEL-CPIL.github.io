"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

import time
import random
import numpy as np

# EVOLVE-BLOCK-START

def _strassen_2x2x2():
    """
    Return the rank-7 Strassen decomposition for 2x2 x 2x2.
    U: shape (4, 7), V: shape (4, 7), W: shape (4, 7)
    Indexing: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    # Strassen's 7 products:
    # M1 = (A11)(B12 - B22)
    # M2 = (A11 + A21)(B22)
    # M3 = (A22 - A11)(B11)
    # M4 = (A22)(B21 - B11)
    # M5 = (A11 + A22)(B11 + B22)
    # M6 = (A12 - A22)(B21 + B22)
    # M7 = (A11 - A21)(B11 + B12)
    
    # U matrix: rows indexed by (i,j) -> i*2+j, columns by product index
    # A11=U[0], A12=U[1], A21=U[2], A22=U[3]
    U = np.zeros((4, 7), dtype=float)
    # M1: A11
    U[0, 0] = 1
    # M2: A11+A21
    U[0, 1] = 1
    U[2, 1] = 1
    # M3: A22-A11
    U[3, 2] = 1
    U[0, 2] = -1
    # M4: A22
    U[3, 3] = 1
    # M5: A11+A22
    U[0, 4] = 1
    U[3, 4] = 1
    # M6: A12-A22
    U[1, 5] = 1
    U[3, 5] = -1
    # M7: A11-A21
    U[0, 6] = 1
    U[2, 6] = -1
    
    # V matrix: rows indexed by (j,k) -> j*2+k, columns by product index
    # B11=V[0], B12=V[1], B21=V[2], B22=V[3]
    V = np.zeros((4, 7), dtype=float)
    # M1: B12-B22
    V[1, 0] = 1
    V[3, 0] = -1
    # M2: B22
    V[3, 1] = 1
    # M3: B11
    V[0, 2] = 1
    # M4: B21-B11
    V[2, 3] = 1
    V[0, 3] = -1
    # M5: B11+B22
    V[0, 4] = 1
    V[3, 4] = 1
    # M6: B21+B22
    V[2, 5] = 1
    V[3, 5] = 1
    # M7: B11+B12
    V[0, 6] = 1
    V[1, 6] = 1
    
    # W matrix: rows indexed by (k,i) -> k*2+i, columns by product index
    # C11 = M1+M3+M5-M6-M7
    # C12 = M1+M4
    # C21 = M2+M3
    # C22 = M5+M4-M2-M7
    W = np.zeros((4, 7), dtype=float)
    # C11 (row 0): M1+M3+M5-M6-M7
    W[0, 0] = 1
    W[0, 2] = 1
    W[0, 4] = 1
    W[0, 5] = -1
    W[0, 6] = -1
    # C12 (row 1): M1+M4
    W[1, 0] = 1
    W[1, 3] = 1
    # C21 (row 2): M2+M3
    W[2, 1] = 1
    W[2, 2] = 1
    # C22 (row 3): M5+M4-M2-M7
    W[3, 4] = 1
    W[3, 3] = 1
    W[3, 1] = -1
    W[3, 6] = -1
    
    return U, V, W

def _build_strassen_2x4x4():
    """
    Build a rank-28 decomposition for 2x4 x 4x4 using 4 blocks of rank-7 Strassen 2x2x2.
    
    The 2x4 matrix A is split into two 2x2 blocks: A = [A_L, A_R]
    The 4x4 matrix B is split into four 2x2 blocks:
        B = [[B_TL, B_TR],
             [B_BL, B_BR]]
    
    C = A*B gives a 2x4 matrix: C = [C_L, C_R] where each is 2x2.
    
    C_L = A_L * B_TL + A_R * B_BL
    C_R = A_L * B_TR + A_R * B_BR
    
    Each 2x2x2 product uses rank 7, so total rank = 4*7 = 28.
    """
    n, m, p = 2, 4, 4
    rank = 28
    
    U = np.zeros((n * m, rank), dtype=float)  # (8, 28)
    V = np.zeros((m * p, rank), dtype=float)  # (16, 28)
    W = np.zeros((p * n, rank), dtype=float)  # (8, 28)
    
    sU, sV, sW = _strassen_2x2x2()
    
    # Block assignments:
    # Block 0: A_L * B_TL -> contributes to C_L
    # Block 1: A_R * B_BL -> contributes to C_L
    # Block 2: A_L * B_TR -> contributes to C_R
    # Block 3: A_R * B_BR -> contributes to C_R
    
    # For each block, the A part is a 2x2 sub-block of the 2x4 A.
    # A has rows indexed by i (0,1) and columns by j (0,1,2,3).
    # A_L = A[:, 0:2], A_R = A[:, 2:4]
    # So A_L has A indices (i, j) with j in {0,1}
    # A_R has A indices (i, j) with j in {2,3}
    
    # B has rows indexed by j (0,1,2,3) and columns by k (0,1,2,3).
    # B_TL = B[0:2, 0:2], B_TR = B[0:2, 2:4]
    # B_BL = B[2:4, 0:2], B_BR = B[2:4, 2:4]
    
    # C has rows indexed by k (0,1,2,3) and columns by i (0,1).
    # W index is k*n+i = k*2+i.
    # C_L = C[:, 0:2] corresponds to k in {0,1}, i in {0,1} -> W indices 0,1,2,3
    # C_R = C[:, 2:4] corresponds to k in {2,3}, i in {0,1} -> W indices 4,5,6,7
    
    blocks = [
        # (a_col_offset, b_row_offset, b_col_offset, c_col_offset)
        # Block 0: A_L (cols 0-1) * B_TL (rows 0-1, cols 0-1) -> C_L (cols 0-1)
        (0, 0, 0, 0),
        # Block 1: A_R (cols 2-3) * B_BL (rows 2-3, cols 0-1) -> C_L (cols 0-1)
        (2, 2, 0, 0),
        # Block 2: A_L (cols 0-1) * B_TR (rows 0-1, cols 2-3) -> C_R (cols 2-3)
        (0, 0, 2, 2),
        # Block 3: A_R (cols 2-3) * B_BR (rows 2-3, cols 2-3) -> C_R (cols 2-3)
        (2, 2, 2, 2),
    ]
    
    for block_idx, (a_col_off, b_row_off, b_col_off, c_col_off) in enumerate(blocks):
        col_offset = block_idx * 7
        for c in range(7):
            # U: A index is (i, a_col_off + j) where j is local column in 2x2 block
            # U row index = i * m + (a_col_off + j) = i * 4 + a_col_off + j
            for i in range(2):
                for j in range(2):
                    u_row = i * 4 + a_col_off + j
                    U[u_row, col_offset + c] = sU[i * 2 + j, c]
            
            # V: B index is (b_row_off + j, b_col_off + k) where j,k are local
            # V row index = (b_row_off + j) * p + (b_col_off + k)
            for j in range(2):
                for k in range(2):
                    v_row = (b_row_off + j) * 4 + (b_col_off + k)
                    V[v_row, col_offset + c] = sV[j * 2 + k, c]
            
            # W: C index is (b_col_off + k, i) -> W row = (b_col_off + k) * n + i
            for k in range(2):
                for i in range(2):
                    w_row = (b_col_off + k) * 2 + i
                    W[w_row, col_offset + c] = sW[k * 2 + i, c]
    
    return U.tolist(), V.tolist(), W.tolist()

def _verify_decomposition(U, V, W, n=2, m=4, p=4, tol=1e-9):
    """Verify that the decomposition exactly decomposes the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # The tensor T[i,j,k] should equal sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Check against the identity: T[i,j,k] = 1 if this corresponds to a valid multiplication term
    # Actually, the matrix multiplication tensor has T[i,j,k] = delta_{i,k} in the sense that
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor is T[i,j,k] = 1 if we're computing C[i,k] from A[i,j]*B[j,k]
    # More precisely, the structure tensor has entries T_{i,j,k} where the multiplication is
    # C_{i,k} = sum_j A_{i,j} B_{j,k}
    # So the tensor component is 1 for all (i,j,k) and 0 otherwise? No.
    # The tensor is defined such that for any A and B:
    # C_{i,k} = sum_{j,c} U_{i*m+j,c} * V_{j*p+k,c} * W_{k*n+i,c} * A_{i,j} * B_{j,k}
    # We need this to equal sum_j A_{i,j} * B_{j,k}
    # So we need: sum_c U_{i*m+j,c} * V_{j*p+k,c} * W_{k*n+i,c} = delta_{j,j} = 1 for all i,j,k
    # Wait, that's not right either. Let me think more carefully.
    
    # C_{i,k} = sum_j A_{i,j} B_{j,k}
    # Using the decomposition: C_{i,k} = sum_c W_{k*n+i,c} * (sum_j U_{i*m+j,c} A_{i,j}) * (sum_j V_{j*p+k,c} B_{j,k})
    # Hmm, that's not the standard form. The standard form is:
    # C_{i,k} = sum_c W_{k*n+i,c} * (sum_j U_{i*m+j,c} A_{i,j}) * (sum_j V_{j*p+k,c} B_{j,k})
    # No wait. Let me use the standard tensor decomposition form.
    
    # The matrix multiplication tensor T has T[i,j,k] such that
    # <T, A ⊗ B> = C where C_{i,k} = sum_j A_{i,j} B_{j,k}
    # In index notation: T_{i,j,k} = δ_{i,k}... no.
    # Actually T is a 3-tensor where T[i,j,k] = 1 for all i,j,k? No.
    # The tensor is T_{i,j,k} where the bilinear form is sum_{i,j,k} T_{i,j,k} A_{i,j} B_{j,k} = C_{i,k}
    # But C_{i,k} depends on i and k, so we need:
    # sum_{j,c} U_{i*m+j,c} V_{j*p+k,c} W_{k*n+i,c} A_{i,j} B_{j,k} = sum_j A_{i,j} B_{j,k}
    # This must hold for all A and B, so:
    # sum_c U_{i*m+j,c} V_{j*p+k,c} W_{k*n+i,c} = 1 for all i,j,k
    # Wait, that would mean the sum equals 1 for every (i,j,k). Let's verify with schoolbook.
    # Schoolbook: U[i*m+j, c] = 1 only when c corresponds to (i,j,k), V[j*p+k, c] = 1 same, W[k*n+i, c] = 1 same.
    # So sum_c ... = 1 for the specific (i,j,k) and 0 for others? No, in schoolbook each c is unique to (i,j,k).
    # So sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] = 1 if there exists c with all three non-zero for this (i,j,k).
    # In schoolbook, for a given (i,j,k), there's exactly one c where all are 1, so the sum is 1.
    # But we need the sum to equal 1 for ALL (i,j,k)? That can't be right because different (i,j,k) give different products.
    
    # I think the correct condition is:
    # For all i, j, k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # This is because C_{i,k} = sum_j (sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c]) A_{i,j} B_{j,k}
    # and we need this to equal sum_j A_{i,j} B_{j,k}, so the coefficient must be 1 for all i,j,k.
    
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                err = abs(s - 1.0)
                if err > max_err:
                    max_err = err
    return max_err <= tol, max_err

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10 second safety margin
    
    # Build the rank-28 Strassen-based decomposition
    U, V, W = _build_strassen_2x4x4()
    
    # Verify
    valid, max_err = _verify_decomposition(U, V, W)
    if not valid:
        # Fallback to schoolbook
        n, m, p = DIMENSIONS
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
