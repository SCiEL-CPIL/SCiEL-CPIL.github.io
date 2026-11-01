"""
Strategy: Deterministic construction of the rank-49 Strassen decomposition for 4x4 matrix multiplication,
with a corrected verification step that checks actual matrix multiplication correctness
using random test matrices, rather than the flawed all-ones tensor check in the seed.
This ensures validity under the correct evaluator definition.
"""
import time
import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START

def _strassen_2x2():
    """
    Returns the Strassen decomposition for 2x2 matrix multiplication.
    a[s, i, j]: coefficient of A[i,j] in subproduct s
    b[s, j, k]: coefficient of B[j,k] in subproduct s
    c[i, k, s]: coefficient of M_s in C[i,k]
    """
    a = np.zeros((7, 2, 2))
    b = np.zeros((7, 2, 2))
    c = np.zeros((2, 2, 7))
    
    # m0 = (A00+A11)(B00+B11)
    a[0] = [[1, 0], [0, 1]]
    b[0] = [[1, 0], [0, 1]]
    # m1 = (A10+A11)B00
    a[1] = [[0, 0], [1, 1]]
    b[1] = [[1, 0], [0, 0]]
    # m2 = A00(B10-B11)
    a[2] = [[1, 0], [0, 0]]
    b[2] = [[0, 0], [1, -1]]
    # m3 = A11(B01-B00)
    a[3] = [[0, 0], [0, 1]]
    b[3] = [[-1, 1], [0, 0]]
    # m4 = (A00+A01)B11
    a[4] = [[1, 1], [0, 0]]
    b[4] = [[0, 0], [0, 1]]
    # m5 = (A10-A00)(B00+B01)
    a[5] = [[-1, 0], [1, 0]]
    b[5] = [[1, 1], [0, 0]]
    # m6 = (A01-A11)(B10+B11)
    a[6] = [[0, 1], [0, -1]]
    b[6] = [[0, 0], [1, 1]]
    
    # C00 = m0 + m3 - m4 + m6
    c[0, 0] = [1, 0, 0, 1, -1, 0, 1]
    # C01 = m2 + m4
    c[0, 1] = [0, 0, 1, 0, 1, 0, 0]
    # C10 = m1 + m3
    c[1, 0] = [0, 1, 0, 1, 0, 0, 0]
    # C11 = m0 - m1 + m2 + m5
    c[1, 1] = [1, -1, 1, 0, 0, 1, 0]
    
    return a, b, c

def _build_strassen_4x4():
    """
    Builds the rank-49 decomposition for 4x4 matrix multiplication
    by taking the tensor product of the rank-7 Strassen decomposition for 2x2.
    """
    a2, b2, c2 = _strassen_2x2()
    
    n, m, p = 4, 4, 4
    rank = 49
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for s1 in range(7):
        for s2 in range(7):
            col = 7 * s1 + s2
            
            # U[i, j, r] = a2[s1, i1, j1] * a2[s2, i2, j2]
            for i1 in range(2):
                for j1 in range(2):
                    for i2 in range(2):
                        for j2 in range(2):
                            i = 2 * i1 + i2
                            j = 2 * j1 + j2
                            U[i * m + j, col] = a2[s1, i1, j1] * a2[s2, i2, j2]
            
            # V[j, k, r] = b2[s1, j1, k1] * b2[s2, j2, k2]
            for j1 in range(2):
                for k1 in range(2):
                    for j2 in range(2):
                        for k2 in range(2):
                            j = 2 * j1 + j2
                            k = 2 * k1 + k2
                            V[j * p + k, col] = b2[s1, j1, k1] * b2[s2, j2, k2]
            
            # W[k, i, r] = c2[i1, k1, s1] * c2[i2, k2, s2]
            for i1 in range(2):
                for k1 in range(2):
                    for i2 in range(2):
                        for k2 in range(2):
                            i = 2 * i1 + i2
                            k = 2 * k1 + k2
                            W[k * n + i, col] = c2[i1, k1, s1] * c2[i2, k2, s2]
    
    return U, V, W

def _verify_matrix_mult(U, V, W):
    """
    Verify that the decomposition exactly computes 4x4 matrix multiplication
    by testing with random matrices.
    """
    n, m, p = 4, 4, 4
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    
    rank = U.shape[1]
    
    # Reshape for easier indexing
    # U[i*m+j, r] -> U4[i, j, r]
    U4 = U.reshape(n, m, rank)
    # V[j*p+k, r] -> V4[j, k, r]
    V4 = V.reshape(m, p, rank)
    # W[k*n+i, r] -> W4[k, i, r]
    W4 = W.reshape(p, n, rank)
    
    # Test with a few random matrices
    rng = np.random.default_rng(42)
    for _ in range(10):
        A = rng.integers(-5, 5, size=(n, m)).astype(float)
        B = rng.integers(-5, 5, size=(m, p)).astype(float)
        C_target = A @ B
        
        # Compute C_recon using the decomposition
        # C[i, k] = sum_r W[k, i, r] * (sum_j U[i, j, r] * A[i, j]) * (sum_l V[j, l, r] * B[j, l])
        # Note: The summation indices in the product terms must match the tensor definition.
        # M_r = (sum_{i,j} U[i,j,r] A[i,j]) * (sum_{j,k} V[j,k,r] B[j,k])
        # This requires summing over all i,j for U and all j,k for V.
        
        term_u = np.einsum('ijr,ij->r', U4, A)
        term_v = np.einsum('jkr,jk->r', V4, B)
        
        # C_recon[i, k] = sum_r W[k, i, r] * term_u[r] * term_v[r]
        C_recon = np.einsum('kir,r,r->ik', W4, term_u, term_v)
        
        if not np.allclose(C_recon, C_target, atol=1e-9):
            return False
            
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 135  # Safety margin
    
    # Build the deterministic rank-49 solution
    U, V, W = _build_strassen_4x4()
    
    # Verify with correct matrix multiplication check
    if _verify_matrix_mult(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to schoolbook rank-64 if verification fails (should not happen)
        n, m, p = 4, 4, 4
        rank = 64
        U_fb = np.zeros((n * m, rank))
        V_fb = np.zeros((m * p, rank))
        W_fb = np.zeros((p * n, rank))
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j, col] = 1.0
                    V_fb[j * p + k, col] = 1.0
                    W_fb[k * n + i, col] = 1.0
                    col += 1
        
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}

# EVOLVE-BLOCK-END
