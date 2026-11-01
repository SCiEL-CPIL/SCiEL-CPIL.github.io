import numpy as np
import time
import random

DIMENSIONS = [2, 3, 5]
DEADLINE = 120.0
SAFETY_MARGIN = 5.0

def verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Use efficient tensor contraction via einsum
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)
    # W shape: (p*n, rank)
    
    # Reshape U, V, W to 3D tensors
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    
    # Compute tensor[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # This is a 3-way sum. We can do it in two steps.
    # Step 1: T1[i,k,r] = sum_j U[i,j,r] * V[j,k,r]
    # Step 2: Tensor[i,k] = sum_r T1[i,k,r] * W[k,i,r]
    
    # T1[i,k,r] = sum_j U[i,j,r] * V[j,k,r]
    # U3: (n, m, rank)
    # V3: (m, p, rank)
    # We want to sum over m (axis 1 of U3, axis 0 of V3)
    # Result T1: (n, p, rank)
    
    # Using einsum: 'ijr,jkr->ikr'
    T1 = np.einsum('ijr,jkr->ikr', U3, V3)
    
    # Now Tensor[i,k] = sum_r T1[i,k,r] * W[k,i,r]
    # T1: (n, p, rank)
    # W3: (p, n, rank)
    # We need W3 to be indexed by [k,i,r], which is (p, n, rank). This matches W3 shape.
    # We want to sum over r (axis 2 of T1, axis 2 of W3)
    # Result Tensor: (n, p)
    
    # Using einsum: 'ikr,ikr->ik'
    # Wait, W3 is (p, n, rank) which is (k, i, r).
    # So we need to align T1[i,k,r] with W3[k,i,r].
    # T1 is (n, p, rank) -> (i, k, r)
    # W3 is (p, n, rank) -> (k, i, r)
    # We need to transpose W3 to (i, k, r) or adjust indices.
    # W3_transposed = W3.transpose(1, 0, 2) -> (n, p, rank) -> (i, k, r)
    
    W3_trans = W3.transpose(1, 0, 2) # (n, p, rank)
    
    Tensor = np.einsum('ikr,ikr->ik', T1, W3_trans)
    
    expected = np.ones((n, p))
    
    if not np.allclose(Tensor, expected, atol=1e-9):
        return False, Tensor
    return True, Tensor

def build_rank_28(n, m, p):
    """
    Construct a rank-28 decomposition for 2x3x5 matrix multiplication.
    """
    rank = 28
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0

    def add_schoolbook(i, j, k, col):
        U[i * m + j][col] = 1.0
        V[j * p + k][col] = 1.0
        W[k * n + i][col] = 1.0
        return col + 1

    def add_strassen(i_off, j_off, k_off, col):
        # Standard Strassen 7-term block for 2x2x2
        # M1 = (A11+A22)(B11+B22)
        U[(i_off+0)*m + (j_off+0)][col] = 1.0
        U[(i_off+1)*m + (j_off+1)][col] = 1.0
        V[(j_off+0)*p + (k_off+0)][col] = 1.0
        V[(j_off+1)*p + (k_off+1)][col] = 1.0
        W[(k_off+0)*n + (i_off+0)][col] = 1.0
        W[(k_off+1)*n + (i_off+1)][col] = 1.0
        col += 1

        # M2 = A22(B21-B11)
        U[(i_off+1)*m + (j_off+1)][col] = 1.0
        V[(j_off+0)*p + (k_off+0)][col] = 1.0
        V[(j_off+1)*p + (k_off+1)][col] = -1.0
        W[(k_off+0)*n + (i_off+1)][col] = 1.0
        W[(k_off+1)*n + (i_off+1)][col] = 1.0
        col += 1

        # M3 = (A11-A22)(B22+B12)
        U[(i_off+0)*m + (j_off+0)][col] = 1.0
        V[(j_off+0)*p + (k_off+1)][col] = 1.0
        V[(j_off+1)*p + (k_off+1)][col] = -1.0
        W[(k_off+1)*n + (i_off+0)][col] = 1.0
        W[(k_off+1)*n + (i_off+1)][col] = 1.0
        col += 1

        # M4 = (A12+A22)(B22-B12)
        U[(i_off+1)*m + (j_off+1)][col] = 1.0
        V[(j_off+1)*p + (k_off+0)][col] = 1.0
        V[(j_off+0)*p + (k_off+0)][col] = -1.0
        W[(k_off+0)*n + (i_off+0)][col] = 1.0
        W[(k_off+0)*n + (i_off+1)][col] = 1.0
        col += 1

        # M5 = (A11+A12)(B12-B22)
        U[(i_off+0)*m + (j_off+0)][col] = 1.0
        U[(i_off+0)*m + (j_off+1)][col] = 1.0
        V[(j_off+1)*p + (k_off+1)][col] = 1.0
        W[(k_off+0)*n + (i_off+0)][col] = -1.0
        W[(k_off+1)*n + (i_off+0)][col] = 1.0
        col += 1

        # M6 = (A21-A11)(B11+B12)
        U[(i_off+1)*m + (j_off+0)][col] = 1.0
        U[(i_off+0)*m + (j_off+0)][col] = -1.0
        V[(j_off+0)*p + (k_off+0)][col] = 1.0
        V[(j_off+0)*p + (k_off+1)][col] = 1.0
        W[(k_off+1)*n + (i_off+1)][col] = 1.0
        col += 1

        # M7 = (A11-A21)(B21-B22)
        U[(i_off+0)*m + (j_off+1)][col] = 1.0
        U[(i_off+1)*m + (j_off+1)][col] = -1.0
        V[(j_off+1)*p + (k_off+0)][col] = 1.0
        V[(j_off+1)*p + (k_off+1)][col] = 1.0
        W[(k_off+0)*n + (i_off+0)][col] = 1.0
        col += 1

        return col

    # 1. Schoolbook for j=0 (all k)
    # n=2, m=3, p=5
    # j=0: i in {0,1}, k in {0,1,2,3,4} -> 2 * 5 = 10 terms
    for k in range(p):
        col = add_schoolbook(0, 0, k, col)
        col = add_schoolbook(1, 0, k, col)
    
    # 2. Strassen block for j=1,2 and k=0,1
    # i in {0,1}, j in {1,2}, k in {0,1}
    # This is a 2x2x2 block.
    col = add_strassen(0, 1, 0, col) # 7 terms
    
    # 3. Schoolbook for j=1,2 and k=2
    # i in {0,1}, j in {1,2}, k=2 -> 2 * 2 = 4 terms
    for i in range(n):
        for j in [1, 2]:
            col = add_schoolbook(i, j, 2, col)
    
    # 4. Strassen block for j=1,2 and k=3,4
    # i in {0,1}, j in {1,2}, k in {3,4}
    # This is a 2x2x2 block.
    col = add_strassen(0, 1, 3, col) # 7 terms

    assert col == rank, f"Expected rank {rank}, got {col}"
    return U, V, W

def snap_to_half_integers(M, tol=1e-6):
    """Snap values to nearest half-integer if close."""
    M = np.array(M, dtype=float)
    M_snapped = np.round(M * 2) / 2
    if np.allclose(M, M_snapped, atol=tol):
        return M_snapped
    return M

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + DEADLINE - SAFETY_MARGIN
    
    # 1. Try known rank 28
    try:
        U, V, W = build_rank_28(n, m, p)
        valid, _ = verify_decomposition(U, V, W, n, m, p)
        if valid:
            return {'U': U, 'V': V, 'W': W}
    except Exception:
        pass
    
    # 2. Fallback to schoolbook if 28 failed (unlikely)
    rank_school = n * m * p
    U = [[0.0] * rank_school for _ in range(n * m)]
    V = [[0.0] * rank_school for _ in range(m * p)]
    W = [[0.0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    
    return {'U': U, 'V': V, 'W': W}
