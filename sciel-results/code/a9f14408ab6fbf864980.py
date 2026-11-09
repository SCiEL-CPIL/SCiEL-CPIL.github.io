"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 5, 5]
TIME_LIMIT = 145.0  # Safety margin for the 150s wall-clock limit

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    """Return the standard schoolbook decomposition for (n,m,p)."""
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

def _validate(U, V, W, n, m, p):
    """Check exact decomposition: C = A @ B where C[k*n+i, r] = sum_j A[i*m+j] * B[j*p+r].
    We check tensor identity: T[i*m+j, j*p+k, k*n+i] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 for all i,j,k.
    """
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = U[i * m + j]
                v = V[j * p + k]
                w = W[k * n + i]
                s = np.sum(u * v * w)
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _strassen_2x2x2():
    """
    Returns U, V, W for 2x2x2 Strassen decomposition.
    U shape: (4, 7), V shape: (4, 7), W shape: (4, 7)
    Indices:
    U: [A11, A12, A21, A22] -> rows 0, 1, 2, 3
    V: [B11, B12, B21, B22] -> rows 0, 1, 2, 3
    W: [C11, C12, C21, C22] -> rows 0, 1, 2, 3
    
    Strassen Formulas:
    M1 = (A11+A22)(B11+B22)
    M2 = (A21+A22)B11
    M3 = A11(B12-B22)
    M4 = A22(B21-B11)
    M5 = (A11+A12)B22
    M6 = (A21-A11)(B11+B12)
    M7 = (A12-A22)(B21+B22)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # Col 0: M1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # Col 1: M2
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # Col 2: M3
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # Col 3: M4
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # Col 4: M5
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # Col 5: M6
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # Col 6: M7
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _combine_decomps(list_of_decomps, n, m, p, offsets):
    """
    Combines a list of decompositions.
    Each decomposition is (U, V, W).
    offsets is a list of (i_off, j_off, k_off) for each decomposition, indicating which block of the global tensor it corresponds to.
    The global tensor is T[i*m+j, j*p+k, k*n+i].
    For a sub-decomposition covering i in [i_off, i_off+n_sub), j in [j_off, j_off+m_sub), k in [k_off, k_off+p_sub),
    we need to embed its U, V, W into the global U, V, W.
    
    U_global[i*m+j, col] += U_sub[i_sub*m_sub+j_sub, col_sub] * coeff
    V_global[j*p+k, col] += V_sub[j_sub*p_sub+k_sub, col_sub] * coeff
    W_global[k*n+i, col] += W_sub[k_sub*n_sub+i_sub, col_sub] * coeff
    
    Wait, the standard Strassen decomposition assumes square blocks of size s.
    Here we have 4x5x5.
    We can split 4x5x5 into:
    1. 2x2x2 block at (0,0,0)
    2. 2x2x2 block at (0,0,2)
    3. 2x2x2 block at (0,2,0)
    4. 2x2x2 block at (0,2,2)
    5. 2x2x2 block at (2,0,0)
    6. 2x2x2 block at (2,0,2)
    7. 2x2x2 block at (2,2,0)
    8. 2x2x2 block at (2,2,2)
    9. Remaining parts:
       - i=0..1, j=2..3, k=4 (2x2x1)
       - i=0..1, j=4, k=0..4 (2x1x5)
       - i=2..3, j=2..3, k=4 (2x2x1)
       - i=2..3, j=4, k=0..4 (2x1x5)
       - i=0..3, j=2..3, k=4 (4x2x1) -- wait, overlapping.
       
    Let's be precise.
    Global dims: n=4, m=5, p=5.
    Indices: i in 0..3, j in 0..4, k in 0..4.
    
    We can use Strassen for the 2x2x2 sub-blocks where j and k are in 0..1 or 2..3.
    Specifically:
    Block 1: i in 0..1, j in 0..1, k in 0..1. (2x2x2)
    Block 2: i in 0..1, j in 2..3, k in 0..1. (2x2x2)
    Block 3: i in 0..1, j in 0..1, k in 2..3. (2x2x2)
    Block 4: i in 0..1, j in 2..3, k in 2..3. (2x2x2)
    Block 5: i in 2..3, j in 0..1, k in 0..1. (2x2x2)
    Block 6: i in 2..3, j in 2..3, k in 0..1. (2x2x2)
    Block 7: i in 2..3, j in 0..1, k in 2..3. (2x2x2)
    Block 8: i in 2..3, j in 2..3, k in 2..3. (2x2x2)
    
    These 8 blocks cover i in 0..3, j in 0..3, k in 0..3.
    Total rank so far: 8 * 7 = 56.
    
    Remaining terms involve j=4 or k=4.
    1. j=4, k in 0..3: i in 0..3. (4x1x4)
       T[i,4,k] = A[i,4]B[4,k]. This is a rank-4 tensor? No, it's a sum of 4 rank-1 tensors?
       Actually, T[i,4,k] = A[i,4]B[4,k].
       We can decompose this as sum over c of U[i,4,c]V[4,k,c]W[k,i,c].
       If we use schoolbook for this part: 4*1*4 = 16 terms.
       U[i*m+4, c] = A[i,4], V[4*p+k, c] = B[4,k], W[k*n+i, c] = 1? No, U,V,W are constants.
       The decomposition must be constant.
       So for each (i,k), we need a term.
       Term for (i,k): U[i*m+4, col] = 1, V[4*p+k, col] = 1, W[k*n+i, col] = 1.
       This is schoolbook rank 16.
       
    2. j in 0..3, k=4: i in 0..3. (4x4x1)
       T[i,j,4] = A[i,j]B[j,4].
       Schoolbook rank 16.
       
    3. j=4, k=4: i in 0..3. (4x1x1)
       T[i,4,4] = A[i,4]B[4,4].
       Schoolbook rank 4.
       
    Total rank = 56 + 16 + 16 + 4 = 92.
    This is an improvement over 100.
    
    Can we do better?
    The remaining parts are:
    - j=4, k=0..3 (4x1x4)
    - j=0..3, k=4 (4x4x1)
    - j=4, k=4 (4x1x1)
    
    We can combine these.
    Note that T[i,4,k] for k=0..3 and T[i,j,4] for j=0..3 are independent.
    However, we might be able to use a smarter decomposition for the "cross" terms.
    
    Let's stick with the 92 rank solution first.
    """
    pass # Not needed, we will build directly in construct

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # We will build the decomposition manually.
    # Total columns (rank) = 56 (Strassen) + 16 (j=4, k<4) + 16 (j<4, k=4) + 4 (j=4, k=4) = 92
    
    rank = 92
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # 1. Strassen blocks for j in 0..3, k in 0..3
    U_s, V_s, W_s = _strassen_2x2x2()
    
    for i_block in range(2): # i in 0..1, 2..3
        for j_block in range(2): # j in 0..1, 2..3
            for k_block in range(2): # k in 0..1, 2..3
                
                i0 = i_block * 2
                j0 = j_block * 2
                k0 = k_block * 2
                
                # Embed Strassen decomposition
                for c in range(7):
                    # U
                    # U_s[0, c] corresponds to A11 -> A[i0, j0]
                    if U_s[0, c] != 0:
                        U[i0 * m + j0, col] += U_s[0, c]
                    # U_s[1, c] corresponds to A12 -> A[i0, j0+1]
                    if U_s[1, c] != 0:
                        U[i0 * m + (j0 + 1), col] += U_s[1, c]
                    # U_s[2, c] corresponds to A21 -> A[i0+1, j0]
                    if U_s[2, c] != 0:
                        U[(i0 + 1) * m + j0, col] += U_s[2, c]
                    # U_s[3, c] corresponds to A22 -> A[i0+1, j0+1]
                    if U_s[3, c] != 0:
                        U[(i0 + 1) * m + (j0 + 1), col] += U_s[3, c]
                        
                    # V
                    # V_s[0, c] corresponds to B11 -> B[j0, k0]
                    if V_s[0, c] != 0:
                        V[j0 * p + k0, col] += V_s[0, c]
                    # V_s[1, c] corresponds to B12 -> B[j0, k0+1]
                    if V_s[1, c] != 0:
                        V[j0 * p + (k0 + 1), col] += V_s[1, c]
                    # V_s[2, c] corresponds to B21 -> B[j0+1, k0]
                    if V_s[2, c] != 0:
                        V[(j0 + 1) * p + k0, col] += V_s[2, c]
                    # V_s[3, c] corresponds to B22 -> B[j0+1, k0+1]
                    if V_s[3, c] != 0:
                        V[(j0 + 1) * p + (k0 + 1), col] += V_s[3, c]
                        
                    # W
                    # W_s[0, c] corresponds to C11 -> C[k0, i0]
                    if W_s[0, c] != 0:
                        W[k0 * n + i0, col] += W_s[0, c]
                    # W_s[1, c] corresponds to C12 -> C[k0, i0+1]
                    if W_s[1, c] != 0:
                        W[k0 * n + (i0 + 1), col] += W_s[1, c]
                    # W_s[2, c] corresponds to C21 -> C[k0+1, i0]
                    if W_s[2, c] != 0:
                        W[(k0 + 1) * n + i0, col] += W_s[2, c]
                    # W_s[3, c] corresponds to C22 -> C[k0+1, i0+1]
                    if W_s[3, c] != 0:
                        W[(k0 + 1) * n + (i0 + 1), col] += W_s[3, c]
                        
                    col += 1
    
    # 2. Schoolbook for j=4, k=0..3 (4x1x4)
    for i in range(n):
        for k in range(4):
            U[i * m + 4, col] = 1
            V[4 * p + k, col] = 1
            W[k * n + i, col] = 1
            col += 1
            
    # 3. Schoolbook for j=0..3, k=4 (4x4x1)
    for i in range(n):
        for j in range(4):
            U[i * m + j, col] = 1
            V[j * p + 4, col] = 1
            W[4 * n + i, col] = 1
            col += 1
            
    # 4. Schoolbook for j=4, k=4 (4x1x1)
    for i in range(n):
        U[i * m + 4, col] = 1
        V[4 * p + 4, col] = 1
        W[4 * n + i, col] = 1
        col += 1
        
    # Verify
    if not _validate(U, V, W, n, m, p):
        # Fallback to schoolbook if validation fails
        U, V, W = _schoolbook(n, m, p)
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
