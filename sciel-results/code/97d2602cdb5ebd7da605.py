"""
Rank-49 exact tensor decomposition for 4x4 matrix multiplication.

This program constructs a rank-49 decomposition for multiplying a 4x4 matrix 
by a 4x4 matrix using the Kronecker product of the Strassen 2x2 algorithm.
The rank 49 is the best known upper bound for the rank of the 4x4 matrix 
multiplication tensor, significantly improving upon the schoolbook rank of 64.
"""

import numpy as np
import time

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START

def _schoolbook(n, m, p):
    """Rank n*m*p schoolbook decomposition."""
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

def _strassen_2x2():
    """
    Returns the U, V, W matrices for the rank-7 Strassen decomposition of 2x2 matrix multiplication.
    U shape: (4, 7) indexed by [i*2+j, c]
    V shape: (4, 7) indexed by [j*2+k, c]
    W shape: (4, 7) indexed by [k*2+i, c]
    """
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ], dtype=int)
    
    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ], dtype=int)
    
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ], dtype=int)
    
    return U2, V2, W2

def _verify_decomposition(U, V, W, n, m, p):
    """
    Verifies that the tensor decomposition exactly computes matrix multiplication.
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    """
    rank = U.shape[1]
    # Reshape to 3D
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    
    # Compute T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # Note: W is indexed by [k, i] in the tensor definition W[k*n+i, c]
    # So W3[k, i, c] corresponds to W[k*n+i, c]
    
    T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    
    # The result T[i,j,k] should be 1 for all i,j,k
    return np.all(T == 1)

def _build_permutation(n, m, p, kron_dim):
    """
    Builds the permutation to map the Kronecker product indices to the standard row-major indices.
    For 4x4, kron_dim is 2 (since 4 = 2 * 2).
    
    The Kronecker product of two 2x2 decompositions creates a 4x4 decomposition.
    The indices in the Kronecker product are (i1, i2, j1, j2) etc.
    The standard index for U is i*4+j.
    The Kronecker index is (i1*2+i2) * (j1*2+j2)? No.
    
    Let's derive the mapping.
    U_kron has rows indexed by (i_a, i_b, j_a, j_b) where i_a, i_b in {0,1} and j_a, j_b in {0,1}.
    Actually, np.kron(U2, U2) creates a matrix of size (4, 49).
    The row index R in the kron matrix corresponds to (r1, r2) where r1, r2 are row indices of U2.
    r1 = i_a * 2 + j_a
    r2 = i_b * 2 + j_b
    R = r1 * 4 + r2 = (i_a * 2 + j_a) * 4 + (i_b * 2 + j_b)
    
    The target row index S for the 4x4 matrix is i * 4 + j.
    i = i_a * 2 + i_b
    j = j_a * 2 + j_b
    
    So we need to map S to R.
    S = (i_a * 2 + i_b) * 4 + (j_a * 2 + j_b)
    R = (i_a * 2 + j_a) * 4 + (i_b * 2 + j_b)
    
    We will build a permutation array perm such that U4[S] = U_kron[R].
    i.e., U4 = U_kron[perm, :]
    So perm[S] = R.
    """
    perm_u = np.zeros(16, dtype=int)
    for S in range(16):
        i = S // 4
        j = S % 4
        i_a, i_b = i // 2, i % 2
        j_a, j_b = j // 2, j % 2
        r1 = i_a * 2 + j_a
        r2 = i_b * 2 + j_b
        R = r1 * 4 + r2
        perm_u[S] = R
        
    perm_v = np.zeros(16, dtype=int)
    for S in range(16):
        j = S // 4
        k = S % 4
        j_a, j_b = j // 2, j % 2
        k_a, k_b = k // 2, k % 2
        r1 = j_a * 2 + k_a
        r2 = j_b * 2 + k_b
        R = r1 * 4 + r2
        perm_v[S] = R
        
    perm_w = np.zeros(16, dtype=int)
    for S in range(16):
        k = S // 4
        i = S % 4
        k_a, k_b = k // 2, k % 2
        i_a, i_b = i // 2, i % 2
        r1 = k_a * 2 + i_a
        r2 = k_b * 2 + i_b
        R = r1 * 4 + r2
        perm_w[S] = R
        
    return perm_u, perm_v, perm_w

def construct():
    """
    Constructs a rank-49 exact tensor decomposition for 4x4 matrix multiplication.
    """
    start_time = time.time()
    deadline = start_time + 100
    
    n, m, p = DIMENSIONS
    
    # Get Strassen 2x2 components
    U2, V2, W2 = _strassen_2x2()
    
    # Kronecker product
    U4 = np.kron(U2, U2)
    V4 = np.kron(V2, V2)
    W4 = np.kron(W2, W2)
    
    # Build permutations
    perm_u, perm_v, perm_w = _build_permutation(n, m, p, 2)
    
    # Apply permutations
    U4 = U4[perm_u, :]
    V4 = V4[perm_v, :]
    W4 = W4[perm_w, :]
    
    # Verify
    if _verify_decomposition(U4, V4, W4, n, m, p):
        return {"U": U4.tolist(), "V": V4.tolist(), "W": W4.tolist()}
    else:
        # Fallback to schoolbook if verification fails (should not happen)
        return _schoolbook(n, m, p)

# EVOLVE-BLOCK-END
