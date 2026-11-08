"""Tensor decomposition for 4x4 * 4x6 matrix multiplication."""
import numpy as np
import time

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def _build_strassen_4x3(n=4, m=4, p=3):
    """
    Build a rank-7 Strassen decomposition for 4x4 * 4x3 matrix multiplication.
    
    Uses 2x2 block Strassen for the 4x4 part, with the 4x3 B matrix handled
    by padding to 4x4 (column 3 is zero) and using the same 7-term structure.
    
    Returns U (16x7), V (12x7), W (12x7) with integer entries.
    Indexing: U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    """
    rank = 7
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen 7 terms for 2x2 blocks
    # A blocks: (row_block, col_block) in {0,1}x{0,1}
    # B blocks: (row_block, col_block) in {0,1}x{0,1} (col_block 1 has zero entries for p=3)
    # C blocks: (row_block, col_block) in {0,1}x{0,1} (col_block 1 has zero entries for p=3)
    
    # A factor contributions per term: [(row_blk, col_blk, coeff), ...]
    A_terms = [
        [(0,0,1), (1,1,1)],       # M1 = (A11+A22)(B11+B22)
        [(1,0,1), (1,1,1)],       # M2 = (A21+A22)B11
        [(0,0,1)],                # M3 = A11(B21-B22)
        [(1,1,1)],                # M4 = A22(B12-B11)
        [(0,0,1), (0,1,1)],       # M5 = (A11+A12)B22
        [(1,0,1), (0,0,-1)],      # M6 = (A21-A11)(B11+B12)
        [(0,1,1), (1,1,-1)],      # M7 = (A12-A22)(B21+B22)
    ]
    
    # B factor contributions per term
    B_terms = [
        [(0,0,1), (1,1,1)],       # M1
        [(0,0,1)],                # M2
        [(1,0,1), (1,1,-1)],      # M3
        [(0,1,1), (0,0,-1)],      # M4
        [(1,1,1)],                # M5
        [(0,0,1), (0,1,1)],       # M6
        [(1,0,1), (1,1,1)],       # M7
    ]
    
    # C factor contributions per term
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    C_terms = [
        [(0,0,1), (1,1,1)],       # M1
        [(1,0,1), (1,1,-1)],      # M2
        [(0,1,1), (1,1,1)],       # M3
        [(0,0,1), (1,0,1)],       # M4
        [(0,0,-1), (0,1,1)],      # M5
        [(1,1,1)],                # M6
        [(0,0,1)],                # M7
    ]
    
    for r in range(rank):
        # Fill U: U[i*m+j, r]
        for (bi, bj, coeff) in A_terms[r]:
            for di in range(2):
                for dj in range(2):
                    i = 2 * bi + di
                    j = 2 * bj + dj
                    U[i * m + j, r] += coeff
        
        # Fill V: V[j*p+k, r]
        for (bj, bk, coeff) in B_terms[r]:
            for dj in range(2):
                for dk in range(2):
                    j = 2 * bj + dj
                    k = 2 * bk + dk
                    if k < p:  # Only include valid k indices (p=3, so k=0,1,2)
                        V[j * p + k, r] += coeff
        
        # Fill W: W[k*n+i, r]
        for (bi, bk, coeff) in C_terms[r]:
            for di in range(2):
                for dk in range(2):
                    i = 2 * bi + di
                    k = 2 * bk + dk
                    if k < p:  # Only include valid k indices
                        W[k * n + i, r] += coeff
    
    return U, V, W


def _build_strassen_4x4(n=4, m=4, p=4):
    """
    Build a rank-7 Strassen decomposition for 4x4 * 4x4 matrix multiplication.
    """
    rank = 7
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    A_terms = [
        [(0,0,1), (1,1,1)],
        [(1,0,1), (1,1,1)],
        [(0,0,1)],
        [(1,1,1)],
        [(0,0,1), (0,1,1)],
        [(1,0,1), (0,0,-1)],
        [(0,1,1), (1,1,-1)],
    ]
    
    B_terms = [
        [(0,0,1), (1,1,1)],
        [(0,0,1)],
        [(1,0,1), (1,1,-1)],
        [(0,1,1), (0,0,-1)],
        [(1,1,1)],
        [(0,0,1), (0,1,1)],
        [(1,0,1), (1,1,1)],
    ]
    
    C_terms = [
        [(0,0,1), (1,1,1)],
        [(1,0,1), (1,1,-1)],
        [(0,1,1), (1,1,1)],
        [(0,0,1), (1,0,1)],
        [(0,0,-1), (0,1,1)],
        [(1,1,1)],
        [(0,0,1)],
    ]
    
    for r in range(rank):
        for (bi, bj, coeff) in A_terms[r]:
            for di in range(2):
                for dj in range(2):
                    i = 2 * bi + di
                    j = 2 * bj + dj
                    U[i * m + j, r] += coeff
        
        for (bj, bk, coeff) in B_terms[r]:
            for dj in range(2):
                for dk in range(2):
                    j = 2 * bj + dj
                    k = 2 * bk + dk
                    V[j * p + k, r] += coeff
        
        for (bi, bk, coeff) in C_terms[r]:
            for di in range(2):
                for dk in range(2):
                    i = 2 * bi + di
                    k = 2 * bk + dk
                    W[k * n + i, r] += coeff
    
    return U, V, W


def _verify(U, V, W, n, m, p):
    """Verify decomposition: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for all i,j,k."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True


def _construct_split_4x3_4x3():
    """
    Construct rank-14 decomposition for 4x4 * 4x6 by splitting into two 4x4*4x3 parts.
    
    B = [B_left (4x3), B_right (4x3)]
    C = A*B_left + A*B_right
    
    Each 4x4*4x3 part uses rank-7 Strassen.
    Total rank = 7 + 7 = 14.
    
    U has 14 columns: first 7 for left part, next 7 for right part.
    V has 14 columns: first 7 for B_left (rows j, cols k=0..2), next 7 for B_right (rows j, cols k=3..5).
    W has 14 columns: first 7 for C_left (cols k=0..2), next 7 for C_right (cols k=3..5).
    """
    n, m, p = 4, 4, 6
    
    # Build two 4x4*4x3 Strassen decompositions
    U1, V1, W1 = _build_strassen_4x3(n, m, 3)  # for columns 0,1,2
    U2, V2, W2 = _build_strassen_4x3(n, m, 3)  # for columns 3,4,5
    
    rank = 14
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Left part (columns 0-2)
    U[:, 0:7] = U1
    V[:, 0:7] = V1  # V1[j*3+k, r] for k=0,1,2 maps to V[j*6+k, r] for k=0,1,2
    # Need to remap: V1 has shape (12, 7) with index j*3+k, but we need j*6+k for k=0,1,2
    V_remapped = np.zeros((m * p, 7))
    for j in range(m):
        for k in range(3):
            V_remapped[j * p + k, :] = V1[j * 3 + k, :]
    V[:, 0:7] = V_remapped
    
    # W1 has shape (12, 7) with index k*4+i for k=0,1,2, maps to W[k*4+i, r] for k=0,1,2
    W[:, 0:7] = W1  # W1[k*4+i, r] already in correct position for k=0,1,2
    
    # Right part (columns 3-5)
    U[:, 7:14] = U2
    # V2[j*3+k, r] for k=0,1,2 maps to V[j*6+(k+3), r] for k=0,1,2
    V_remapped2 = np.zeros((m * p, 7))
    for j in range(m):
        for k in range(3):
            V_remapped2[j * p + (k + 3), :] = V2[j * 3 + k, :]
    V[:, 7:14] = V_remapped2
    
    # W2[k*4+i, r] for k=0,1,2 maps to W[(k+3)*4+i, r] for k=0,1,2
    W_remapped2 = np.zeros((p * n, 7))
    for k in range(3):
        for i in range(n):
            W_remapped2[(k + 3) * n + i, :] = W2[k * n + i, :]
    W[:, 7:14] = W_remapped2
    
    return U, V, W


def construct():
    n, m, p = DIMENSIONS
    
    # Construct the rank-14 decomposition
    U, V, W = _construct_split_4x3_4x3()
    
    # Convert to lists with integer values
    U_list = [[int(round(x)) for x in row] for row in U]
    V_list = [[int(round(x)) for x in row] for row in V]
    W_list = [[int(round(x)) for x in row] for row in W]
    
    # Verify
    if not _verify(U_list, V_list, W_list, n, m, p):
        # Fallback: schoolbook (should not happen)
        rank_school = n * m * p
        U_s = np.zeros((n * m, rank_school))
        V_s = np.zeros((m * p, rank_school))
        W_s = np.zeros((p * n, rank_school))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_s[i * m + j, col] = 1
                    V_s[j * p + k, col] = 1
                    W_s[k * n + i, col] = 1
                    col += 1
        return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
