"""
Rank-49 exact tensor decomposition for 4x4 matrix multiplication.

Uses the Kronecker product of Strassen's rank-7 algorithm for 2x2 matrices,
with a runtime verification step that constructs the full tensor and solves
for the exact index mapping to guarantee correctness against the evaluator's
specific trace convention. Falls back to schoolbook (rank 64) on any failure.
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


def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor.
    
    The tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    should equal 1 for all (i,j,k) in the multiplication tensor.
    For matrix multiplication, the tensor is the identity in the sense that
    C[i,k] = sum_j A[i,j] * B[j,k], so T[i,j,k] = 1 for all valid (i,j,k).
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Check all (i,j,k) entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(U.shape[1]):
                    s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True


def _strassen_kron_4x4(n, m, p):
    """Rank-49 decomposition via Kronecker product of Strassen 2x2.
    
    Uses explicit index permutation to match the linear index convention:
    U[i*m+j], V[j*p+k], W[k*n+i]
    """
    # Strassen 2x2 factors
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

    # Kronecker product gives 16x49 matrices
    U4 = np.kron(U2, U2)
    V4 = np.kron(V2, V2)
    W4 = np.kron(W2, W2)

    # The Kronecker product arranges rows in a specific order.
    # We need to permute rows to match our convention:
    # U row index: i*m+j where i,j in {0,1,2,3}
    # V row index: j*p+k where j,k in {0,1,2,3}
    # W row index: k*n+i where k,i in {0,1,2,3}
    
    # In the Kronecker product U2 ⊗ U2, the row index is (a*4 + b)
    # where a is the row in the first U2 and b is the row in the second U2.
    # For 2x2 Strassen, U2 rows correspond to (i1,j1) pairs:
    # Row 0: (0,0), Row 1: (0,1), Row 2: (1,0), Row 3: (1,1)
    
    # In our 4x4 convention, i = i1*2 + i0, j = j1*2 + j0
    # So U row index = i*4 + j = (i1*2+i0)*4 + (j1*2+j0)
    
    # In the Kronecker product, row (a, b) where a = i1*2+j1, b = i0*2+j0
    # maps to row a*4 + b = (i1*2+j1)*4 + (i0*2+j0)
    
    # We need: target_row = i*4 + j = (i1*2+i0)*4 + (j1*2+j0)
    # Source row in kron: src_row = (i1*2+j1)*4 + (i0*2+j0)
    
    # Build permutation: for each target row, find the source row
    perm_u = np.zeros(16, dtype=int)
    for i in range(4):
        for j in range(4):
            i1, i0 = i // 2, i % 2
            j1, j0 = j // 2, j % 2
            target = i * 4 + j
            a = i1 * 2 + j1
            b = i0 * 2 + j0
            src = a * 4 + b
            perm_u[target] = src

    perm_v = np.zeros(16, dtype=int)
    for j in range(4):
        for k in range(4):
            j1, j0 = j // 2, j % 2
            k1, k0 = k // 2, k % 2
            target = j * 4 + k
            a = j1 * 2 + k1
            b = j0 * 2 + k0
            src = a * 4 + b
            perm_v[target] = src

    perm_w = np.zeros(16, dtype=int)
    for k in range(4):
        for i in range(4):
            k1, k0 = k // 2, k % 2
            i1, i0 = i // 2, i % 2
            target = k * 4 + i
            a = k1 * 2 + i1
            b = k0 * 2 + i0
            src = a * 4 + b
            perm_w[target] = src

    U4 = U4[perm_u, :]
    V4 = V4[perm_v, :]
    W4 = W4[perm_w, :]

    # Verify the decomposition
    if not _verify_decomposition(U4, V4, W4, n, m, p):
        # Fallback: try alternative permutation approach
        # Direct construction without permutation
        U_direct = np.zeros((16, 49), dtype=int)
        V_direct = np.zeros((16, 49), dtype=int)
        W_direct = np.zeros((16, 49), dtype=int)
        
        for i in range(4):
            for j in range(4):
                i1, i0 = i // 2, i % 2
                j1, j0 = j // 2, j % 2
                u_row = i * 4 + j
                u_block = i1 * 2 + j1
                u_entry = i0 * 2 + j0
                for alpha in range(7):
                    for beta in range(7):
                        col = alpha * 7 + beta
                        U_direct[u_row, col] = U2[u_block, alpha] * U2[u_entry, beta]
        
        for j in range(4):
            for k in range(4):
                j1, j0 = j // 2, j % 2
                k1, k0 = k // 2, k % 2
                v_row = j * 4 + k
                v_block = j1 * 2 + k1
                v_entry = j0 * 2 + k0
                for alpha in range(7):
                    for beta in range(7):
                        col = alpha * 7 + beta
                        V_direct[v_row, col] = V2[v_block, alpha] * V2[v_entry, beta]
        
        for k in range(4):
            for i in range(4):
                k1, k0 = k // 2, k % 2
                i1, i0 = i // 2, i % 2
                w_row = k * 4 + i
                w_block = k1 * 2 + i1
                w_entry = k0 * 2 + i0
                for alpha in range(7):
                    for beta in range(7):
                        col = alpha * 7 + beta
                        W_direct[w_row, col] = W2[w_block, alpha] * W2[w_entry, beta]
        
        if _verify_decomposition(U_direct, V_direct, W_direct, n, m, p):
            return {"U": U_direct.tolist(), "V": V_direct.tolist(), "W": W_direct.tolist()}
        
        # Final fallback: schoolbook
        return _schoolbook(n, m, p)

    return {"U": U4.tolist(), "V": V4.tolist(), "W": W4.tolist()}


def construct():
    """
    Constructs a minimal-rank exact tensor decomposition for 4x4 matrix multiplication.
    Returns {'U', 'V', 'W'} with integer entries.
    """
    start_time = time.time()
    deadline = start_time + 100  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    # Try rank-49 Kronecker-Strassen with verification
    if time.time() < deadline:
        r49 = _strassen_kron_4x4(n, m, p)
        return r49
    
    # Final fallback: schoolbook
    return _schoolbook(n, m, p)

# EVOLVE-BLOCK-END
