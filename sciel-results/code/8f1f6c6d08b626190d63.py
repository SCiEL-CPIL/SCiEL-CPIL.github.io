"""Matrix multiplication 4x4x7 tensor decomposition search."""
import time
import random
import numpy as np

DIMENSIONS = [4, 4, 7]
DEADLINE = 150.0
SAFETY_MARGIN = 5.0

def _time_remaining():
    return DEADLINE - time.time() - SAFETY_MARGIN

def _build_tensors(n, m, p):
    """Build U, V, W matrices for schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check if U, V, W exactly decompose the matrix multiplication tensor.
    
    The multiplication tensor T[i,j,k] = delta (where C[i,k] = sum_j A[i,j]*B[j,k])
    must satisfy: T = sum_r U[:,r] * V[:,r] * W[:,r] (outer product sum)
    
    U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    """
    rank = U.shape[1]
    # Build the multiplication tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1 if the indices are consistent
    # Actually, the tensor for matrix multiplication C = A * B where A is n x m, B is m x p
    # T[a, b, c] where a = i*m+j, b = j'*p+k, c = k'*n+i'
    # T[a, b, c] = 1 if j == j' and i == i' and k == k'
    # More precisely: T[i*m+j, j*p+k, k*n+i] = 1 for valid i,j,k
    
    # The decomposition requires: for all (a,b,c):
    # sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c]
    
    # Efficient check: compute U @ diag(V) @ W^T style
    # Actually: sum_r U[:,r] * V[:,r] * W[:,r] as a 3-tensor
    # We can check by: for each r, the outer product U[:,r] * V[:,r] * W[:,r]
    # Sum over r should give T.
    
    # Faster: reshape and use einsum
    # T_expected[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i with same i,j,k
    T_expected = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_expected[i * m + j, j * p + k, k * n + i] = 1
    
    # Compute T_computed = sum_r U[:,r] * V[:,r] * W[:,r]
    # Using einsum: 'ar,br,cr->abc'
    T_computed = np.einsum('ar,br,cr->abc', U, V, W)
    
    diff = np.abs(T_computed - T_expected)
    return np.max(diff) < tol

def _strassen_2x2x2():
    """Return the Strassen rank-7 decomposition for 2x2x2 matrix multiplication.
    
    For 2x2 matrices, the 7 products are:
    M1 = (A11)(B12 - B22)
    M2 = (A11 + A12)(B22)
    M3 = (A21 + A22)(B11)
    M4 = (A22)(B21 - B11)
    M5 = (A11 + A22)(B11 + B22)
    M6 = (A12 - A22)(B21 + B22)
    M7 = (A21 - A11)(B11 + B12)
    
    C11 = M5 + M4 - M2 - M7
    C12 = M1 + M2
    C21 = M3 + M4
    C22 = M5 + M1 - M3 - M6
    
    We need U, V, W for the 2x2x2 tensor.
    A is 2x2: indices (i,j) with i in {0,1}, j in {0,1}
    B is 2x2: indices (j,k) with j in {0,1}, k in {0,1}
    C is 2x2: indices (i,k) with i in {0,1}, k in {0,1}
    
    U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    n = m = p = 2
    rank = 7
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # M1 = (A11)(B12 - B22)
    # A11: i=0,j=0 -> U[0,0] = 1
    # B12: j=0,k=1 -> V[0*2+1,0] = 1; B22: j=1,k=1 -> V[1*2+1,0] = -1
    # C12: i=0,k=1 -> W[1*2+0,0] = 1
    U[0, 0] = 1
    V[1, 0] = 1
    V[3, 0] = -1
    W[2, 0] = 1
    
    # M2 = (A11 + A12)(B22)
    # A11: i=0,j=0 -> U[0,1] = 1; A12: i=0,j=1 -> U[1,1] = 1
    # B22: j=1,k=1 -> V[3,1] = 1
    # C11: i=0,k=0 -> W[0,1] = -1; C12: i=0,k=1 -> W[2,1] = 1
    U[0, 1] = 1
    U[1, 1] = 1
    V[3, 1] = 1
    W[0, 1] = -1
    W[2, 1] = 1
    
    # M3 = (A21 + A22)(B11)
    # A21: i=1,j=0 -> U[2,2] = 1; A22: i=1,j=1 -> U[3,2] = 1
    # B11: j=0,k=0 -> V[0,2] = 1
    # C21: i=1,k=0 -> W[1,2] = 1; C22: i=1,k=1 -> W[3,2] = -1
    U[2, 2] = 1
    U[3, 2] = 1
    V[0, 2] = 1
    W[1, 2] = 1
    W[3, 2] = -1
    
    # M4 = (A22)(B21 - B11)
    # A22: i=1,j=1 -> U[3,3] = 1
    # B21: j=1,k=0 -> V[2,3] = 1; B11: j=0,k=0 -> V[0,3] = -1
    # C11: i=0,k=0 -> W[0,3] = 1; C21: i=1,k=0 -> W[1,3] = 1
    U[3, 3] = 1
    V[2, 3] = 1
    V[0, 3] = -1
    W[0, 3] = 1
    W[1, 3] = 1
    
    # M5 = (A11 + A22)(B11 + B22)
    # A11: i=0,j=0 -> U[0,4] = 1; A22: i=1,j=1 -> U[3,4] = 1
    # B11: j=0,k=0 -> V[0,4] = 1; B22: j=1,k=1 -> V[3,4] = 1
    # C11: i=0,k=0 -> W[0,4] = 1; C22: i=1,k=1 -> W[3,4] = 1
    U[0, 4] = 1
    U[3, 4] = 1
    V[0, 4] = 1
    V[3, 4] = 1
    W[0, 4] = 1
    W[3, 4] = 1
    
    # M6 = (A12 - A22)(B21 + B22)
    # A12: i=0,j=1 -> U[1,5] = 1; A22: i=1,j=1 -> U[3,5] = -1
    # B21: j=1,k=0 -> V[2,5] = 1; B22: j=1,k=1 -> V[3,5] = 1
    # C22: i=1,k=1 -> W[3,5] = -1
    U[1, 5] = 1
    U[3, 5] = -1
    V[2, 5] = 1
    V[3, 5] = 1
    W[3, 5] = -1
    
    # M7 = (A21 - A11)(B11 + B12)
    # A21: i=1,j=0 -> U[2,6] = 1; A11: i=0,j=0 -> U[0,6] = -1
    # B11: j=0,k=0 -> V[0,6] = 1; B12: j=0,k=1 -> V[1,6] = 1
    # C11: i=0,k=0 -> W[0,6] = -1; C21: i=1,k=0 -> W[1,6] = -1
    U[2, 6] = 1
    U[0, 6] = -1
    V[0, 6] = 1
    V[1, 6] = 1
    W[0, 6] = -1
    W[1, 6] = -1
    
    return U, V, W

def _kronecker_strassen(n, m, p):
    """Build a decomposition using Kronecker product with Strassen for the 2x2x2 blocks.
    
    For 4x4x7: split A into 2x2 blocks of 2x2, B into 2x2 blocks of 2x2 (for first 4 cols) 
    plus 2x1 blocks of 2x3 (for last 3 cols).
    
    Actually, let's think more carefully.
    A is 4x4, B is 4x7, C is 4x7.
    
    Split A into 2x2 blocks: A = [[A00, A01], [A10, A11]] where each is 2x2.
    Split B into blocks: B = [[B00, B01, B02], [B10, B11, B12]] where B0*,B1* are 2x2, B02,B12 are 2x3.
    
    C = [[C00, C01, C02], [C10, C11, C12]] where C0*,C1* are 2x2, C02,C12 are 2x3.
    
    For the 2x2x2 parts (first 4 columns of B):
    C00 = A00*B00 + A01*B10
    C01 = A00*B01 + A01*B11
    C10 = A10*B00 + A11*B10
    C11 = A10*B01 + A11*B11
    
    Each 2x2x2 multiplication can be done with Strassen (rank 7).
    
    For the 2x2x3 parts (last 3 columns of B):
    C02 = A00*B02 + A01*B12
    C12 = A10*B02 + A11*B12
    
    These are 2x2 (A block) times 2x3 (B block) giving 2x3 (C block).
    This is a 2x2x3 tensor. The schoolbook rank is 2*2*3 = 12.
    
    Total rank with this approach:
    - 4 Strassen 2x2x2 multiplications: 4 * 7 = 28
    - 2 schoolbook 2x2x3 multiplications: 2 * 12 = 24
    Total: 52
    
    But wait, we can potentially share. Let me just implement the straightforward Kronecker.
    """
    n, m, p = 4, 4, 7
    
    # We'll build U, V, W by combining Strassen for 2x2x2 parts and schoolbook for 2x2x3 parts.
    # 
    # Indexing:
    # A[i,j] where i in [0,4), j in [0,4)
    # B[j,k] where j in [0,4), k in [0,7)
    # C[i,k] where i in [0,4), k in [0,7)
    #
    # U[i*m+j, r] = U[i*4+j, r]
    # V[j*p+k, r] = V[j*7+k, r]
    # W[k*n+i, r] = W[k*4+i, r]
    
    # For the first 4 columns of B (k in [0,4)):
    # Use Strassen on the 2x2 block structure.
    # A block (bi, bj) where bi, bj in {0,1} corresponds to rows 2*bi to 2*bi+1, cols 2*bj to 2*bj+1
    # B block (bj, bk) where bj in {0,1}, bk in {0,1} corresponds to rows 2*bj to 2*bj+1, cols 2*bk to 2*bk+1
    # C block (bi, bk) where bi, bk in {0,1} corresponds to rows 2*bi to 2*bi+1, cols 2*bk to 2*bk+1
    
    # Strassen U,V,W for 2x2x2:
    # U2[i2*2+j2, r] where i2,j2 in {0,1}
    # V2[j2*2+k2, r] where j2,k2 in {0,1}
    # W2[k2*2+i2, r] where k2,i2 in {0,1}
    
    SU, SV, SW = _strassen_2x2x2()
    
    # For each combination of block indices (bi, bk) in C, we need contributions from (bi,bj) in A and (bj,bk) in B.
    # Actually, the full 4x4x4 multiplication tensor decomposes as a sum over block structures.
    
    # Let me build it differently. The 4x4x4 tensor T4[i,j,k] = 1 if consistent.
    # We can write T4 = sum over (bi,bj,bk) of T2_block(bi,bj,bk)
    # where T2_block uses Strassen for the 2x2x2 sub-tensor at blocks (bi,bj,bk).
    
    # Actually, the 4x4x4 multiplication tensor is the "Kronecker square" of the 2x2x2 tensor.
    # T4 = T2 * T2 (Kronecker product in tensor sense)
    # This gives rank 7*7 = 49 for 4x4x4.
    
    # For 4x4x7, we extend B by 3 more columns.
    # The 4x4x7 tensor = 4x4x4 tensor (for first 4 cols) + 4x4x3 tensor (for last 3 cols)
    
    # 4x4x4: rank 49 via Kronecker of Strassen
    # 4x4x3: schoolbook rank 4*4*3 = 48
    
    # Total: 49 + 48 = 97
    
    # Let's build this.
    
    rank_444 = 49
    rank_443 = 4 * 4 * 3  # 48
    total_rank = rank_444 + rank_443  # 97
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    # Part 1: 4x4x4 using Kronecker product of Strassen
    # The Kronecker product of two rank-7 decompositions gives rank 49.
    # For the Kronecker product:
    # U_kron[i*4+j, r1*7+r2] = U2[i*2+i2, r1] * U2[j*2+j2, r2]  -- no, this isn't quite right.
    
    # Actually, for matrix multiplication, the tensor for (n1*n2) x (m1*m2) x (p1*p2)
    # is the tensor product of the tensors for n1 x m1 x p1 and n2 x m2 x p2.
    # The decomposition: if T1 = sum_r u1_r * v1_r * w1_r and T2 = sum_s u2_s * v2_s * w2_s
    # then T1 * T2 = sum_{r,s} (u1_r * u2_s) * (v1_r * v2_s) * (w1_r * w2_s)
    # where * denotes Kronecker product of vectors.
    
    # So U_kron[i1*n2+i2, r*7+s] = U1[i1, r] * U2[i2, s]
    # Wait, need to be careful with indexing.
    
    # For 4x4x4: n=4, m=4, p=4
    # A is 4x4, B is 4x4, C is 4x4
    # U[i*4+j, r], V[j*4+k, r], W[k*4+i, r]
    
    # Split: i = 2*i1 + i2, j = 2*j1 + j2, k = 2*k1 + k2
    # where i1,j1,k1 in {0,1} (block index), i2,j2,k2 in {0,1} (within-block index)
    
    # U[2*i1+i2, 2*j1+j2, r] = U2[i1, j1, r1] * U2[i2, j2, r2] where r = 7*r1 + r2
    # V[2*j1+j2, 2*k1+k2, r] = V2[j1, k1, r1] * V2[j2, k2, r2]
    # W[2*k1+k2, 2*i1+i2, r] = W2[k1, i1, r1] * W2[k2, i2, r2]
    
    for r1 in range(7):
        for r2 in range(7):
            r = r1 * 7 + r2
            for i1 in range(2):
                for j1 in range(2):
                    for k1 in range(2):
                        for i2 in range(2):
                            for j2 in range(2):
                                for k2 in range(2):
                                    i = 2 * i1 + i2
                                    j = 2 * j1 + j2
                                    k = 2 * k1 + k2
                                    val_U = SU[i1 * 2 + j1, r1] * SU[i2 * 2 + j2, r2]
                                    val_V = SV[j1 * 2 + k1, r1] * SV[j2 * 2 + k2, r2]
                                    val_W = SW[k1 * 2 + i1, r1] * SW[k2 * 2 + i2, r2]
                                    U[i * 4 + j, r] += val_U
                                    V[j * 4 + k, r] += val_V
                                    W[k * 4 + i, r] += val_W
    
    # Part 2: 4x4x3 schoolbook for columns 4,5,6 of B
    for i in range(4):
        for j in range(4):
            for k in range(3):
                k_actual = k + 4  # actual column index in B
                r = rank_444 + i * 4 * 3 + j * 3 + k
                U[i * 4 + j, r] = 1
                V[j * 7 + k_actual, r] = 1
                W[k_actual * 4 + i, r] = 1
    
    return U, V, W

def _construct_kronecker_strassen():
    """Build the Kronecker-Strassen decomposition."""
    n, m, p = DIMENSIONS
    U, V, W = _kronecker_strassen(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def construct():
    """Construct a low-rank exact tensor decomposition for 4x4x7 matrix multiplication."""
    start_time = time.time()
    
    # Try the Kronecker-Strassen approach first (rank 97)
    U, V, W = _kronecker_strassen(4, 4, 7)
    
    # Verify
    if _verify(U, V, W, 4, 4, 7):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook if verification fails
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
