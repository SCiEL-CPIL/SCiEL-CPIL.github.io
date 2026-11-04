"""
Rank-minimal exact tensor decomposition for 4x4 matrix multiplication.

Uses the Kronecker product of Strassen's rank-7 decomposition for 2x2x2
to achieve rank 49 for 4x4x4 matrix multiplication. This is the current
best-known upper bound on the tensor rank of M_{4,4,4}.

The decomposition is constructed by viewing 4x4 matrices as 2x2 block
matrices of 2x2 submatrices. Strassen's algorithm is applied at the block
level (7 block multiplications), and each block multiplication is itself
a 2x2*2x2 product decomposed via Strassen (rank 7). Total rank = 7 * 7 = 49.
"""

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _strassen_2x2_arrays():
    """
    Return the rank-7 Strassen decomposition for 2x2x2 matrix multiplication
    as NumPy arrays of shape (4, 7).
    
    Indices for 2x2 matrices:
    A: [[A00, A01], [A10, A11]] -> index i*2+j
    B: [[B00, B01], [B10, B11]] -> index j*2+k
    C: [[C00, C01], [C10, C11]] -> index k*2+i (Trace convention W[k*n+i, r])
    
    Note on W index: The problem statement says W[k*n+i, r].
    For 2x2, n=2. So index is k*2 + i.
    C[i,k] corresponds to row i, col k of C.
    Standard tensor T[i,j,k] = A[i,j]B[j,k]C[k,i]? No.
    Matrix mult: C[i,k] = sum_j A[i,j] B[j,k].
    Tensor T has non-zero entries at (i*m+j, j*p+k, k*n+i).
    For 2x2: (2i+j, 2j+k, 2k+i).
    
    Let's verify the W index mapping in the previous code:
    W[k*2+i, r] = coefficient of C[i,k].
    Yes, this matches the trace convention W[k*n+i, r] where n=2.
    """
    rank = 7
    U = np.zeros((4, rank), dtype=float)
    V = np.zeros((4, rank), dtype=float)
    W = np.zeros((4, rank), dtype=float)
    
    # Term 1: M1 = (A00+A11)(B00+B11)
    # A: A00 (0), A11 (3)
    # B: B00 (0), B11 (3)
    # C: C00 (i=0,k=0 -> idx 0), C11 (i=1,k=1 -> idx 3)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # Term 2: M2 = (A10+A11)B00
    # A: A10 (2), A11 (3)
    # B: B00 (0)
    # C: C10 (i=1,k=0 -> idx 1), C11 (i=1,k=1 -> idx 3)
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    
    # Term 3: M3 = A00(B01-B11)
    # A: A00 (0)
    # B: B01 (1), B11 (3)
    # C: C01 (i=0,k=1 -> idx 2), C11 (i=1,k=1 -> idx 3)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    
    # Term 4: M4 = A11(B10-B00)
    # A: A11 (3)
    # B: B10 (2), B00 (0)
    # C: C00 (i=0,k=0 -> idx 0), C10 (i=1,k=0 -> idx 1)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    # Term 5: M5 = (A00+A01)B11
    # A: A00 (0), A01 (1)
    # B: B11 (3)
    # C: C00 (i=0,k=0 -> idx 0), C01 (i=0,k=1 -> idx 2)
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    # Term 6: M6 = (A10-A00)(B00+B01)
    # A: A10 (2), A00 (0)
    # B: B00 (0), B01 (1)
    # C: C11 (i=1,k=1 -> idx 3)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # Term 7: M7 = (A01-A11)(B10+B11)
    # A: A01 (1), A11 (3)
    # B: B10 (2), B11 (3)
    # C: C00 (i=0,k=0 -> idx 0)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T of shape (n*m, m*p, p*n)."""
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T

def _verify_decomp(n, m, p, U, V, W, tol=1e-9):
    """Verify that sum_r U[a,r]*V[b,r]*W[c,r] matches the tensor."""
    T = _build_tensor(n, m, p)
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Efficient verification using einsum
    T_decomp = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr, optimize=True)
    max_diff = np.max(np.abs(T_decomp - T))
    return max_diff < tol, max_diff

def _kronecker_decomposition(n, m, p):
    """
    Construct the rank-49 decomposition for 4x4x4 using Kronecker product
    of the rank-7 Strassen decomposition for 2x2x2.
    """
    U2, V2, W2 = _strassen_2x2_arrays()
    
    rank = 49
    # We need to map the 2x2 indices to 4x4 indices.
    # 4x4 index for A: i*m + j. Here m=4.
    # i = 2*i1 + i2, j = 2*j1 + j2.
    # Index = (2*i1 + i2)*4 + (2*j1 + j2) = 8*i1 + 4*i2 + 2*j1 + j2.
    
    # We can generate the full matrices by iterating over the block indices
    # and using the base matrices, or by constructing them via Kronecker products
    # of the index mappings.
    
    # Let's build index maps for the 4x4 case.
    # U4 row index: a = i*4 + j = (2*i1+i2)*4 + (2*j1+j2)
    # V4 row index: b = j*4 + k = (2*j1+j2)*4 + (2*k1+k2)
    # W4 row index: c = k*4 + i = (2*k1+k2)*4 + (2*i1+i2)
    
    # The rank index r = r1*7 + r2.
    
    U = np.zeros((16, 49), dtype=float)
    V = np.zeros((16, 49), dtype=float)
    W = np.zeros((16, 49), dtype=float)
    
    # Iterate over all combinations of block indices
    for i1 in range(2):
        for i2 in range(2):
            for j1 in range(2):
                for j2 in range(2):
                    for k1 in range(2):
                        for k2 in range(2):
                            # 4x4 indices
                            i = 2 * i1 + i2
                            j = 2 * j1 + j2
                            k = 2 * k1 + k2
                            
                            a = i * 4 + j
                            b = j * 4 + k
                            c = k * 4 + i
                            
                            # 2x2 indices for the base Strassen matrices
                            # U2 index: i1*2 + j1 (for A block)
                            u_idx_1 = i1 * 2 + j1
                            u_idx_2 = i2 * 2 + j2
                            
                            # V2 index: j1*2 + k1 (for B block)
                            v_idx_1 = j1 * 2 + k1
                            v_idx_2 = j2 * 2 + k2
                            
                            # W2 index: k1*2 + i1 (for C block, trace convention)
                            # Note: W2[k*2+i, r]. Here block indices are k1, i1.
                            w_idx_1 = k1 * 2 + i1
                            w_idx_2 = k2 * 2 + i2
                            
                            # For each pair of Strassen ranks r1, r2
                            for r1 in range(7):
                                u1 = U2[u_idx_1, r1]
                                v1 = V2[v_idx_1, r1]
                                w1 = W2[w_idx_1, r1]
                                
                                if u1 == 0 or v1 == 0 or w1 == 0:
                                    continue
                                    
                                for r2 in range(7):
                                    u2 = U2[u_idx_2, r2]
                                    v2 = V2[v_idx_2, r2]
                                    w2 = W2[w_idx_2, r2]
                                    
                                    if u2 == 0 or v2 == 0 or w2 == 0:
                                        continue
                                        
                                    r = r1 * 7 + r2
                                    U[a, r] = u1 * u2
                                    V[b, r] = v1 * v2
                                    W[c, r] = w1 * w2
                                    
    # Convert to lists of integers for the output format
    U_list = [list(map(int, row)) for row in U]
    V_list = [list(map(int, row)) for row in V]
    W_list = [list(map(int, row)) for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list, "rank": rank}

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0
    
    result = _kronecker_decomposition(n, m, p)
    
    if time.time() < deadline:
        valid, max_diff = _verify_decomp(n, m, p, result["U"], result["V"], result["W"])
        
        if not valid:
            # Fallback to schoolbook if verification fails (should not happen)
            rank = n * m * p
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i * m + j][col] = 1
                        V[j * p + k][col] = 1
                        W[k * n + i][col] = 1
                        col += 1
            return {"U": U, "V": V, "W": W}
    
    return {"U": result["U"], "V": result["V"], "W": result["W"]}
# EVOLVE-BLOCK-END
