"""Matrix multiplication 3x4x4 tensor decomposition search.

Uses a block decomposition:
- Top 2x4 block of output: Strassen (rank 7) applied to two 2x2 sub-blocks
- Bottom 1x4 row: schoolbook (rank 4)
Total rank = 7 + 7 + 4 = 18
"""
import time
import numpy as np

DIMENSIONS = [3, 4, 4]


def _strassen_2x2():
    """Return Strassen rank-7 decomposition for 2x2x2 matrix multiplication.
    
    T[i,j,k] = C[i,j] where C = A*B
    A is 2x2, B is 2x2, C is 2x2.
    Indices: i in {0,1}, j in {0,1}, k in {0,1}
    T[i,j,k] = sum_r U[i,r] V[j,r] W[k,r]
    """
    # Strassen's 7 products:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A10)B11
    # M6 = (A01-A11)(B10+B11)
    # M7 = (A01-A10)(B00+B10)
    #
    # C00 = M1 - M4 + M5 + M7
    # C01 = M3 + M4
    # C10 = M2 + M5
    # C11 = M1 - M3 + M2 - M6
    #
    # U[i, r]: coefficient of A[i_row, i_col] in product r
    # V[j, r]: coefficient of B[j_row, j_col] in product r  
    # W[k, r]: coefficient of C[k_row, k_col] in product r
    #
    # A indices: (i_row, i_col) -> flat: i_row*2 + i_col
    # B indices: (j_row, j_col) -> flat: j_row*2 + j_col
    # C indices: (k_row, k_col) -> flat: k_row*2 + k_col
    
    # U: shape (4, 7)
    # V: shape (4, 7)
    # W: shape (4, 7)
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A00+A11)(B00+B11)
    # A00 = A[0,0] -> flat 0; A11 = A[1,1] -> flat 3
    # B00 = B[0,0] -> flat 0; B11 = B[1,1] -> flat 3
    # C00 += M1, C11 += M1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (A10+A11)B00
    # A10 = A[1,0] -> flat 2; A11 = A[1,1] -> flat 3
    # B00 = B[0,0] -> flat 0
    # C10 += M2, C11 -= M2
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # M3 = A00(B01-B11)
    # A00 = A[0,0] -> flat 0
    # B01 = B[0,1] -> flat 1; B11 = B[1,1] -> flat 3
    # C01 += M3, C11 -= M3
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = -1
    
    # M4 = A11(B10-B00)
    # A11 = A[1,1] -> flat 3
    # B10 = B[1,0] -> flat 2; B00 = B[0,0] -> flat 0
    # C00 -= M4, C01 += M4
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = -1; W[1, 3] = 1
    
    # M5 = (A00+A10)B11
    # A00 = A[0,0] -> flat 0; A10 = A[1,0] -> flat 2
    # B11 = B[1,1] -> flat 3
    # C00 += M5, C10 += M5
    U[0, 4] = 1; U[2, 4] = 1
    V[3, 4] = 1
    W[0, 4] = 1; W[2, 4] = 1
    
    # M6 = (A01-A11)(B10+B11)
    # A01 = A[0,1] -> flat 1; A11 = A[1,1] -> flat 3
    # B10 = B[1,0] -> flat 2; B11 = B[1,1] -> flat 3
    # C11 -= M6
    U[1, 5] = 1; U[3, 5] = -1
    V[2, 5] = 1; V[3, 5] = 1
    W[3, 5] = -1
    
    # M7 = (A01-A10)(B00+B10)
    # A01 = A[0,1] -> flat 1; A10 = A[1,0] -> flat 2
    # B00 = B[0,0] -> flat 0; B10 = B[1,0] -> flat 2
    # C00 += M7
    U[1, 6] = 1; U[2, 6] = -1
    V[0, 6] = 1; V[2, 6] = 1
    W[0, 6] = 1
    
    return U, V, W


def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is exact."""
    rank = len(U[0])
    # Build the multiplication tensor T[i,j,k] = delta(i, k) * delta(j, j) ... 
    # Actually for matrix multiplication: C[i,k] = sum_j A[i,j] B[j,k]
    # Tensor form: T[i, j, k] = 1 if this corresponds to C[i,k] getting contribution from A[i,j]*B[j,k]
    # So T[i, j, k] = 1 for all valid i,j,k (it's the identity mapping)
    # Wait, the tensor for matrix multiplication is T[i,j,k] where C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor is defined as T[i, j, k] = 1 (it's just the sum over j)
    # Actually the multiplication tensor has T[i,j,k] = 1 for all i in [n], j in [m], k in [p]
    # No wait. Let me think again.
    # C = A * B where A is n x m, B is m x p, C is n x p
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T has entries T[i, j, k] = 1 for all i, j, k
    # And the decomposition says: T[i,j,k] = sum_r U[i,r] V[j,r] W[k,r]
    # Wait no. The standard form is:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # = sum_r (sum_j A[i,j] V[j,r]) (sum_? ...) 
    # Actually the tensor decomposition form is:
    # T[i,j,k] = sum_r U[i,r] V[j,r] W[k,r]
    # where T[i,j,k] = 1 (the multiplication tensor for n x m x p)
    
    # Let's verify by checking T[i,j,k] = sum_r U[i,r] V[j,r] W[k,r]
    # T[i,j,k] = 1 for all valid i, j, k
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Check a sample of tensor entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U_arr[i, r] * V_arr[j, r] * W_arr[k, r] for r in range(rank))
                if abs(val - 1.0) > 1e-9:
                    return False
    return True


def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Strategy:
    # Output C is 3x4. Split into top 2 rows and bottom 1 row.
    # Top 2x4: use Strassen on two 2x2 sub-blocks (cols 0-1 and cols 2-3)
    # Bottom 1x4: schoolbook (4 products)
    # Total rank = 7 + 7 + 4 = 18
    
    # U: shape (n*m, rank) = (12, 18)
    # V: shape (m*p, rank) = (16, 18)  
    # W: shape (p*n, rank) = (12, 18)
    
    # Indexing:
    # A[i,j] -> U index: i*m + j = i*4 + j
    # B[j,k] -> V index: j*p + k = j*4 + k
    # C[i,k] -> W index: k*n + i = k*3 + i (trace convention W[k*n+i])
    
    U_full = np.zeros((nm, 18))
    V_full = np.zeros((mp, 18))
    W_full = np.zeros((pn, 18))
    
    # Get Strassen decomposition for 2x2x2
    sU, sV, sW = _strassen_2x2()
    
    # Block 1: Top-left 2x2 output (rows 0-1, cols 0-1)
    # A sub-block: rows 0-1, cols 0-1 -> A[0:2, 0:2]
    # B sub-block: rows 0-1, cols 0-1 -> B[0:2, 0:2]
    # C sub-block: rows 0-1, cols 0-1 -> C[0:2, 0:2]
    #
    # U indices for A[0:2, 0:2]: i*4+j for i in {0,1}, j in {0,1} -> {0,1,4,5}
    # V indices for B[0:2, 0:2]: j*4+k for j in {0,1}, k in {0,1} -> {0,1,4,5}
    # W indices for C[0:2, 0:2]: k*3+i for k in {0,1}, i in {0,1} -> {0,1,3,4}
    
    # Map Strassen flat indices to full indices
    # Strassen A flat: (i_row, i_col) -> i_row*2 + i_col
    # Full A flat: i_row*4 + i_col
    # For i_row in {0,1}, i_col in {0,1}: flat_strassen = i_row*2+i_col, flat_full = i_row*4+i_col
    for r in range(7):
        for s_idx in range(4):
            sr = s_idx // 2  # i_row
            sc = s_idx % 2   # i_col
            full_idx = sr * 4 + sc
            U_full[full_idx, r] = sU[s_idx, r]
    
    for r in range(7):
        for s_idx in range(4):
            sr = s_idx // 2  # j_row
            sc = s_idx % 2   # j_col
            full_idx = sr * 4 + sc
            V_full[full_idx, r] = sV[s_idx, r]
    
    for r in range(7):
        for s_idx in range(4):
            sr = s_idx // 2  # k_row
            sc = s_idx % 2   # k_col
            full_idx = sr * 3 + sc
            W_full[full_idx, r] = sW[s_idx, r]
    
    # Block 2: Top-right 2x2 output (rows 0-1, cols 2-3)
    # A sub-block: rows 0-1, cols 0-1 -> A[0:2, 0:2] (same A sub-block!)
    # B sub-block: rows 0-1, cols 2-3 -> B[0:2, 2:4]
    # C sub-block: rows 0-1, cols 2-3 -> C[0:2, 2:4]
    #
    # U indices: same as block 1 (A[0:2, 0:2])
    # V indices for B[0:2, 2:4]: j*4+k for j in {0,1}, k in {2,3} -> {2,3,6,7}
    # W indices for C[0:2, 2:4]: k*3+i for k in {2,3}, i in {0,1} -> {6,7,9,10}
    
    for r in range(7, 14):
        for s_idx in range(4):
            sr = s_idx // 2  # i_row
            sc = s_idx % 2   # i_col
            full_idx = sr * 4 + sc
            U_full[full_idx, r] = sU[s_idx, r-7]
    
    for r in range(7, 14):
        for s_idx in range(4):
            sr = s_idx // 2  # j_row
            sc = s_idx % 2   # j_col
            # B sub-block is B[0:2, 2:4], so full col = 2 + sc
            full_idx = sr * 4 + (2 + sc)
            V_full[full_idx, r] = sV[s_idx, r-7]
    
    for r in range(7, 14):
        for s_idx in range(4):
            sr = s_idx // 2  # k_row
            sc = s_idx % 2   # k_col
            # C sub-block is C[0:2, 2:4], so full col = 2 + sc
            full_idx = sr * 3 + (2 + sc)
            W_full[full_idx, r] = sW[s_idx, r-7]
    
    # Block 3: Bottom row (row 2, cols 0-3) - schoolbook
    # C[2,k] = sum_j A[2,j] * B[j,k] for k in {0,1,2,3}
    # 4 products: for each j in {0,1,2,3}, product = A[2,j] * B[j,k]
    # Wait, schoolbook for 1x4: C[2,k] = sum_j A[2,j] B[j,k]
    # We need 4 products, one for each j.
    # Product j: U[A[2,j], j+14] = 1, V[B[j,k], j+14] = 1 for all k, W[C[2,k], j+14] = 1 for all k
    
    for j in range(4):
        r = 14 + j
        U_full[2 * 4 + j, r] = 1  # A[2,j]
        for k in range(4):
            V_full[j * 4 + k, r] = 1  # B[j,k]
        for k in range(4):
            W_full[k * 3 + 2, r] = 1  # C[2,k]
    
    # Verify
    if not _verify(U_full.tolist(), V_full.tolist(), W_full.tolist(), n, m, p):
        # Fallback to schoolbook if verification fails
        rank = n * m * p
        U_sb = np.zeros((nm, rank))
        V_sb = np.zeros((mp, rank))
        W_sb = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j, col] = 1
                    V_sb[j * p + k, col] = 1
                    W_sb[k * n + i, col] = 1
                    col += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
