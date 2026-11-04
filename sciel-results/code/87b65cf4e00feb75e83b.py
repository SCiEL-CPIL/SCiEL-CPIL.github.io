"""Optimized matrix multiplication tensor decomposition for 4x5 * 5x6."""
DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _make_schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _make_strassen_2x2_block(n, m, p, i1, i2, j1, j2, k1, k2):
    """
    Create a 2x2x2 Strassen-like block decomposition.
    Returns (U, V, W) as dense arrays of size (n*m, 7), (m*p, 7), (p*n, 7).
    """
    rank = 7
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Strassen 7-term formulas for 2x2x2:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B21 - B22)
    # M4 = A22(B12 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    #
    # C11 = M1 + M4 - M5 + M6
    # C12 = M2 + M5
    # C21 = M3 + M4
    # C22 = M1 - M2 + M3 + M7
    
    # M1: (A[i1,j1] + A[i2,j2]) * (B[j1,k1] + B[j2,k2])
    U[i1*m + j1, 0] = 1.0
    U[i2*m + j2, 0] = 1.0
    V[j1*p + k1, 0] = 1.0
    V[j2*p + k2, 0] = 1.0
    W[k1*n + i1, 0] = 1.0  # C11
    W[k2*n + i2, 0] = 1.0  # C22
    
    # M2: (A[i2,j1] + A[i2,j2]) * B[j1,k1]
    U[i2*m + j1, 1] = 1.0
    U[i2*m + j2, 1] = 1.0
    V[j1*p + k1, 1] = 1.0
    W[k2*n + i1, 1] = 1.0  # C12
    
    # M3: A[i1,j1] * (B[j2,k1] - B[j2,k2])
    U[i1*m + j1, 2] = 1.0
    V[j2*p + k1, 2] = 1.0
    V[j2*p + k2, 2] = -1.0
    W[k1*n + i2, 2] = 1.0  # C21
    
    # M4: A[i2,j2] * (B[j1,k2] - B[j1,k1])
    U[i2*m + j2, 3] = 1.0
    V[j1*p + k2, 3] = 1.0
    V[j1*p + k1, 3] = -1.0
    W[k1*n + i1, 3] = 1.0  # C11
    W[k1*n + i2, 3] = 1.0  # C21
    
    # M5: (A[i1,j1] + A[i1,j2]) * B[j2,k2]
    U[i1*m + j1, 4] = 1.0
    U[i1*m + j2, 4] = 1.0
    V[j2*p + k2, 4] = 1.0
    W[k1*n + i1, 4] = -1.0  # C11
    W[k2*n + i1, 4] = 1.0   # C12
    
    # M6: (A[i2,j1] - A[i1,j1]) * (B[j1,k1] + B[j1,k2])
    U[i2*m + j1, 5] = 1.0
    U[i1*m + j1, 5] = -1.0
    V[j1*p + k1, 5] = 1.0
    V[j1*p + k2, 5] = 1.0
    W[k1*n + i1, 5] = 1.0   # C11
    
    # M7: (A[i1,j2] - A[i2,j2]) * (B[j2,k1] + B[j2,k2])
    U[i1*m + j2, 6] = 1.0
    U[i2*m + j2, 6] = -1.0
    V[j2*p + k1, 6] = 1.0
    V[j2*p + k2, 6] = 1.0
    W[k2*n + i2, 6] = 1.0   # C22
    
    return U, V, W

def _verify_full(U, V, W, n, m, p, tol=1e-9):
    """Full verification of the decomposition."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    r = U.shape[1]
    if r != V.shape[1] or r != W.shape[1]:
        return False
    
    # Full tensor check: T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    # should be 1 for all (i,j,k), and for any (i',j',k') with (i',k')!=(i,k),
    # sum_r U[i*m+j,r]*V[j'*p+k',r]*W[k'*n+i',r] = 0 when (i',k')!=(i,k)
    
    # More precisely: define S[i,k,j,i2,k2] = sum_r U[i2*m+j,r]*V[j*p+k2,r]*W[k*n+i,r]
    # We need S[i,k,j,i,k] = 1 and S[i,k,j,i2,k2] = 0 for (i2,k2)!=(i,k)
    
    # Efficient: for each j, compute the matrix M_j where M_j[i,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
    # This should be the identity matrix of size n*p... no wait.
    
    # Actually the condition is:
    # For each (j, i2, k2): the vector v where v[i,k] = sum_r U[i2*m+j,r]*V[j*p+k2,r]*W[k*n+i,r]
    # should be 1 only at (i=i2, k=k2) and 0 elsewhere.
    
    # Let's just do it directly for all combinations
    for i2 in range(n):
        for j in range(m):
            for k2 in range(p):
                # Compute contribution to each (i, k)
                for i in range(n):
                    for k in range(p):
                        val = np.sum(U[i2*m+j, :] * V[j*p+k2, :] * W[k*n+i, :])
                        expected = 1.0 if (i == i2 and k == k2) else 0.0
                        if abs(val - expected) > tol:
                            return False
    return True

def _verify_fast(U, V, W, n, m, p, tol=1e-9):
    """Fast verification using matrix operations."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    r = U.shape[1]
    if r != V.shape[1] or r != W.shape[1]:
        return False
    
    U_mat = U.reshape(n, m, -1)
    V_mat = V.reshape(m, p, -1)
    W_mat = W.reshape(p, n, -1)
    
    # Diagonal check
    diag = np.einsum('ijr,jkr,kir->ijk', U_mat, V_mat, W_mat, optimize=True)
    if not np.allclose(diag, 1.0, atol=tol):
        return False
    
    # Off-diagonal: check that for each (i,j,k), the "cross terms" vanish
    # sum_r U[i,j,r]*V[j,k,r]*W[k',i',r] = 0 for (i',k') != (i,k)
    # This is equivalent to: for each j, the tensor S_j[i,k,i',k'] = sum_r U[i,j,r]*V[j,k,r]*W[k',i',r]
    # should have S_j[i,k,i,k] = 1 and S_j[i,k,i',k'] = 0 for (i',k')!=(i,k)
    
    # For each j, form A_j[i,k,r] = U[i,j,r]*V[j,k,r] (shape n x p x r)
    # Then S_j[i,k,i',k'] = sum_r A_j[i,k,r] * W[k',i',r]
    # Let B_j[i',k',r] = W[k',i',r] reshaped to (p,n,r) -> (p*n, r)
    # S_j = einsum('ikr,rpn->ikpn', A_j, W_mat[j]) where W_mat[j] is (p,n,r)
    
    for j in range(m):
        A_j = U_mat[:, j, :] * V_mat[j, :, :]  # (n, p, r)
        # S_j[i,k,i',k'] = sum_r A_j[i,k,r] * W_mat[j, k', i', r]
        S_j = np.einsum('ikr,kpr->ikip', A_j, W_mat[j], optimize=True)  # (n,p,n,p)
        # Check S_j[i,k,i,k] = 1 and rest = 0
        # Extract diagonal
        idx = np.arange(n)
        diag_vals = S_j[idx, idx, idx, idx]  # This is wrong, need S_j[i,k,i,k]
        # Actually S_j[i,k,i',k'] with i'=i, k'=k: S_j[i,k,i,k]
        # Let's check element by element
        for i in range(n):
            for k in range(p):
                if abs(S_j[i, k, i, k] - 1.0) > tol:
                    return False
                for ip in range(n):
                    for kp in range(p):
                        if ip == i and kp == k:
                            continue
                        if abs(S_j[i, k, ip, kp]) > tol:
                            return False
    return True

def _build_optimized_decomposition(n, m, p):
    """
    Build decomposition using 2x2x2 Strassen blocks where possible,
    schoolbook for remaining strips.
    """
    # Partition: i into 2+2, j into 2+2+1, k into 2+2+2
    i_blocks = [(0, 1), (2, 3)]
    j_blocks = [(0, 1), (2, 3)]
    k_blocks = [(0, 1), (2, 3), (4, 5)]
    
    U_list = []
    V_list = []
    W_list = []
    
    covered = set()
    
    for (i1, i2) in i_blocks:
        for (j1, j2) in j_blocks:
            for (k1, k2) in k_blocks:
                U_block, V_block, W_block = _make_strassen_2x2_block(n, m, p, i1, i2, j1, j2, k1, k2)
                U_list.append(U_block)
                V_list.append(V_block)
                W_list.append(W_block)
                for i in [i1, i2]:
                    for j in [j1, j2]:
                        for k in [k1, k2]:
                            covered.add((i, j, k))
    
    # Schoolbook for uncovered
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    U_col = np.zeros(n * m)
                    V_col = np.zeros(m * p)
                    W_col = np.zeros(p * n)
                    U_col[i * m + j] = 1.0
                    V_col[j * p + k] = 1.0
                    W_col[k * n + i] = 1.0
                    U_list.append(U_col.reshape(-1, 1))
                    V_list.append(V_col.reshape(-1, 1))
                    W_list.append(W_col.reshape(-1, 1))
    
    U = np.hstack(U_list)
    V = np.hstack(V_list)
    W = np.hstack(W_list)
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin
    
    # Build Strassen-based decomposition
    U_strassen, V_strassen, W_strassen = _build_optimized_decomposition(n, m, p)
    rank_strassen = U_strassen.shape[1]
    
    # Verify
    if _verify_fast(U_strassen, V_strassen, W_strassen, n, m, p, tol=1e-9):
        U_best = U_strassen
        V_best = V_strassen
        W_best = W_strassen
    else:
        # Fall back to schoolbook
        U_best, V_best, W_best = _make_schoolbook(n, m, p)
    
    def to_list(mat):
        return mat.tolist()
    
    return {"U": to_list(U_best), "V": to_list(V_best), "W": to_list(W_best)}
# EVOLVE-BLOCK-END
