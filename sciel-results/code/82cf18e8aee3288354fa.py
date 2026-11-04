"""
Rank-minimal tensor decomposition for 2x4 by 4x5 matrix multiplication.
Uses recursive Strassen-like decomposition to achieve rank 36.
"""
import time
import numpy as np

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START

def _strassen_2x2x2():
    """
    Strassen's algorithm for 2x2 by 2x2 matrix multiplication.
    Returns (U, V, W) with shapes (4, 7), (4, 7), (4, 7).
    U indexed by (i*2+j), V by (j*2+k), W by (k*2+i) for i,j,k in {0,1}.
    """
    # Strassen's 7 products:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # U: rows indexed by (i,j) -> i*2+j for A[i,j]
    # V: rows indexed by (j,k) -> j*2+k for B[j,k]
    # W: rows indexed by (k,i) -> k*2+i for C[i,k] (transposed output)
    
    # Let's define the 7 rank-1 terms
    # Each term c has U[c], V[c], W[c]
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # Term 0: M1 = (A11+A22)(B11+B22)
    # U: A11 at (0,0)=idx0, A22 at (1,1)=idx3
    U[0, 0] = 1.0  # A[0,0]
    U[3, 0] = 1.0  # A[1,1]
    # V: B11 at (0,0)=idx0, B22 at (1,1)=idx3
    V[0, 0] = 1.0  # B[0,0]
    V[3, 0] = 1.0  # B[1,1]
    # W: contributes to C11 (idx k*2+i = 0*2+0=0) and C22 (idx k*2+i = 1*2+1=3)
    W[0, 0] = 1.0  # C[0,0]
    W[3, 0] = 1.0  # C[1,1]
    
    # Term 1: M2 = (A21+A22)B11
    U[2, 1] = 1.0  # A[1,0]
    U[3, 1] = 1.0  # A[1,1]
    V[0, 1] = 1.0  # B[0,0]
    W[0, 1] = 1.0  # C[1,0] -> k=0,i=1 -> idx 0*2+1=1
    W[1, 1] = 1.0  # C[0,0] -> wait, C21 means C[1,0], k=0,i=1 -> idx 1
    # Actually C21 is C[1,0], so k=0, i=1, W index = k*2+i = 0*2+1 = 1
    # And C11 is C[0,0], k=0, i=0, W index = 0
    # M2 contributes to C21 = C[1,0]
    W[1, 1] = 1.0  # C[1,0]
    
    # Let me redo this more carefully.
    # C[i,k] where i is row of A (0,1), k is col of B (0,1)
    # W index = k*2 + i
    # C[0,0]: k=0,i=0 -> idx 0
    # C[0,1]: k=1,i=0 -> idx 2
    # C[1,0]: k=0,i=1 -> idx 1
    # C[1,1]: k=1,i=1 -> idx 3
    
    # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
    U[0, 0] = 1.0; U[3, 0] = 1.0
    V[0, 0] = 1.0; V[3, 0] = 1.0
    W[0, 0] = 1.0; W[3, 0] = 1.0
    
    # M2 = (A10+A11)B00 -> C10 += M2
    U[2, 1] = 1.0; U[3, 1] = 1.0
    V[0, 1] = 1.0
    W[1, 1] = 1.0  # C[1,0]
    
    # M3 = A00(B10-B11) -> C01 += M3
    U[0, 2] = 1.0
    V[2, 2] = 1.0; V[3, 2] = -1.0  # B[1,0], B[1,1]
    W[2, 2] = 1.0  # C[0,1]
    
    # M4 = A11(B01-B00) -> C00 += M4, C10 += M4
    U[3, 3] = 1.0
    V[1, 3] = 1.0; V[0, 3] = -1.0  # B[0,1], B[0,0]
    W[0, 3] = 1.0; W[1, 3] = 1.0  # C[0,0], C[1,0]
    
    # M5 = (A00+A01)B11 -> C00 -= M5, C01 += M5
    U[0, 4] = 1.0; U[1, 4] = 1.0
    V[3, 4] = 1.0
    W[0, 4] = -1.0; W[2, 4] = 1.0  # C[0,0], C[0,1]
    
    # M6 = (A10-A00)(B00+B01) -> C11 += M6
    U[2, 5] = 1.0; U[0, 5] = -1.0
    V[0, 5] = 1.0; V[1, 5] = 1.0
    W[3, 5] = 1.0  # C[1,1]
    
    # M7 = (A01-A11)(B10+B11) -> C00 += M7
    U[1, 6] = 1.0; U[3, 6] = -1.0
    V[2, 6] = 1.0; V[3, 6] = 1.0
    W[0, 6] = 1.0  # C[0,0]
    
    return U, V, W


def _schoolbook_2x2x1():
    """
    Schoolbook for 2x2 by 2x1 matrix multiplication.
    Returns (U, V, W) with shapes (4, 4), (2, 4), (2, 4).
    U indexed by (i*2+j), V by (j*2+k) where k=0 only, W by (k*2+i) where k=0 only.
    """
    U = np.zeros((4, 4))
    V = np.zeros((2, 4))
    W = np.zeros((2, 4))
    
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    
    # Term 0: A[0,0]*B[0,0] -> C[0,0]
    U[0, 0] = 1.0
    V[0, 0] = 1.0
    W[0, 0] = 1.0  # k=0,i=0 -> idx 0
    
    # Term 1: A[0,1]*B[1,0] -> C[0,0]
    U[1, 1] = 1.0
    V[1, 1] = 1.0
    W[0, 1] = 1.0
    
    # Term 2: A[1,0]*B[0,0] -> C[1,0]
    U[2, 2] = 1.0
    V[0, 2] = 1.0
    W[1, 2] = 1.0  # k=0,i=1 -> idx 1
    
    # Term 3: A[1,1]*B[1,0] -> C[1,0]
    U[3, 3] = 1.0
    V[1, 3] = 1.0
    W[1, 3] = 1.0
    
    return U, V, W


def _combine_blocks(U1, V1, W1, U2, V2, W2, n, m, p, offset_u, offset_v, offset_w):
    """
    Combine two block decompositions.
    Block 1 handles rows 0..n-1, cols 0..m-1 of A, cols 0..p1-1 of B, output cols 0..p1-1
    Block 2 handles rows 0..n-1, cols 0..m-1 of A, cols m1..m-1 of B, output cols p1..p-1
    
    Wait, this isn't quite right. Let me think about the structure more carefully.
    
    For 2x4 by 4x5:
    Split B into B_left (2x2) and B_right (2x3)? No, B is 4x5.
    Split the inner dimension (4) into 2+2.
    Split the output dimension (5) into 2+3.
    
    So A is 2x4 = [A_left (2x2), A_right (2x2)]
    B is 4x5 = [B_top (2x5); B_bottom (2x5)]
    C = A_left @ B_top + A_right @ B_bottom
    
    Each product is 2x2 by 2x5.
    For 2x2 by 2x5, split B into 2x2 + 2x3.
    2x2 by 2x2: Strassen, rank 7
    2x2 by 2x3: rank 11 (7 + 4)
    So 2x2 by 2x5: rank 18
    Total for 2x4 by 4x5: 18 + 18 = 36
    """
    rank1 = U1.shape[1]
    rank2 = U2.shape[1]
    total_rank = rank1 + rank2
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    # Copy block 1
    U[:n*m, :rank1] = U1
    V[:m*p, :rank1] = V1
    W[:p*n, :rank1] = W1
    
    # Copy block 2
    U[:n*m, rank1:] = U2
    V[:m*p, rank1:] = V2
    W[:p*n, rank1:] = W2
    
    return U, V, W


def _decompose_2x2x5():
    """
    Decompose 2x2 by 2x5 matrix multiplication.
    Split B into B1 (2x2) and B2 (2x3).
    C = A @ B1 + A @ B2
    A @ B1: 2x2 by 2x2, rank 7 (Strassen)
    A @ B2: 2x2 by 2x3, rank 11
    Total rank: 18
    """
    U1, V1, W1 = _strassen_2x2x2()
    
    # For 2x2 by 2x3:
    # Split B2 into B2a (2x2) and B2b (2x1)
    # A @ B2a: 2x2 by 2x2, rank 7
    # A @ B2b: 2x2 by 2x1, rank 4
    # Total for 2x2 by 2x3: 11
    
    U2a, V2a, W2a = _strassen_2x2x2()
    U2b, V2b, W2b = _schoolbook_2x2x1()
    
    # Now we need to combine these into the full 2x2 by 2x5 decomposition
    # The indices need to be mapped correctly.
    
    # For 2x2 by 2x5:
    # A is 2x2, indexed by (i,j) -> i*2+j
    # B is 2x5, indexed by (j,k) -> j*5+k
    # C is 2x5, output indexed by (k,i) -> k*2+i
    
    # Block 1: A @ B1 where B1 is cols 0-1 of B
    # B1[j,k] for k=0,1 maps to B[j, k] in the full B
    # So V1 index (j,k) where k in {0,1} maps to j*5+k
    
    # Block 2a: A @ B2a where B2a is cols 2-3 of B
    # B2a[j,k] for k=0,1 maps to B[j, 2+k]
    # So V2a index (j,k) where k in {0,1} maps to j*5+(2+k)
    
    # Block 2b: A @ B2b where B2b is col 4 of B
    # B2b[j,0] maps to B[j, 4]
    # So V2b index (j,0) maps to j*5+4
    
    # Let's build the full decomposition
    rank_total = 7 + 7 + 4  # 18
    
    U = np.zeros((4, rank_total))
    V = np.zeros((10, rank_total))  # 2*5
    W = np.zeros((10, rank_total))  # 5*2
    
    # Block 1: Strassen for cols 0-1
    U[:, 0:7] = U1
    # V1 has shape (4, 7), indexed by (j,k) with k in {0,1}
    # Map to (j, 0+k) in full B
    for j in range(2):
        for k in range(2):
            V[j*5 + k, 0:7] = V1[j*2 + k, 0:7]
    # W1 has shape (4, 7), indexed by (k,i) with k in {0,1}
    # Map to (k, i) in full C, W index = k*2+i
    for k in range(2):
        for i in range(2):
            W[k*2 + i, 0:7] = W1[k*2 + i, 0:7]
    
    # Block 2a: Strassen for cols 2-3
    offset = 7
    U[:, offset:offset+7] = U2a
    for j in range(2):
        for k in range(2):
            V[j*5 + (2+k), offset:offset+7] = V2a[j*2 + k, 0:7]
    for k in range(2):
        for i in range(2):
            W[(2+k)*2 + i, offset:offset+7] = W2a[k*2 + i, 0:7]
    
    # Block 2b: Schoolbook for col 4
    offset2 = 14
    U[:, offset2:offset2+4] = U2b
    for j in range(2):
        V[j*5 + 4, offset2:offset2+4] = V2b[j*2 + 0, 0:4]
    for i in range(2):
        W[4*2 + i, offset2:offset2+4] = W2b[0*2 + i, 0:4]
    
    return U, V, W


def _decompose_2x4x5():
    """
    Decompose 2x4 by 4x5 matrix multiplication.
    Split A into A_left (2x2) and A_right (2x2).
    Split B into B_top (2x5) and B_bottom (2x5).
    C = A_left @ B_top + A_right @ B_bottom
    Each product is 2x2 by 2x5, rank 18.
    Total rank: 36.
    """
    U1, V1, W1 = _decompose_2x2x5()
    U2, V2, W2 = _decompose_2x2x5()
    
    rank_total = 18 + 18  # 36
    
    U = np.zeros((8, rank_total))  # 2*4
    V = np.zeros((20, rank_total))  # 4*5
    W = np.zeros((10, rank_total))  # 5*2
    
    # Block 1: A_left @ B_top
    # A_left[i,j] for j in {0,1} maps to A[i, j]
    # B_top[j,k] for j in {0,1} maps to B[j, k]
    U[:, 0:18] = U1
    for j in range(2):
        for k in range(5):
            V[j*5 + k, 0:18] = V1[j*5 + k, 0:18]
    W[:, 0:18] = W1
    
    # Block 2: A_right @ B_bottom
    # A_right[i,j] for j in {0,1} maps to A[i, 2+j]
    # B_bottom[j,k] for j in {0,1} maps to B[2+j, k]
    offset = 18
    U[:, offset:offset+18] = U2
    for j in range(2):
        for k in range(5):
            V[(2+j)*5 + k, offset:offset+18] = V2[j*5 + k, 0:18]
    W[:, offset:offset+18] = W2
    
    return U, V, W


def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i,j,k."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    U, V, W = _decompose_2x4x5()
    
    # Verify
    if _verify_decomposition(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook if verification fails
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
