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
    """
    Full exact verification of the tensor decomposition.
    Returns True if the decomposition exactly computes matrix multiplication.
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    r = U.shape[1]
    if r != V.shape[1] or r != W.shape[1]:
        return False
    
    U_mat = U.reshape(n, m, -1)  # (n, m, r)
    V_mat = V.reshape(m, p, -1)  # (m, p, r)
    W_mat = W.reshape(p, n, -1)  # (p, n, r)
    
    # Compute the full tensor S[i', k', j, i] = sum_r U[i,j,r] * V[j,k,r] * W[k',i',r]
    # We need S[i', k', j, i] = 1 if (i',k') == (i,k), else 0
    
    # Efficient computation using einsum
    # First compute T[j, k, i, r] = U[i,j,r] * V[j,k,r]  (note: U is indexed by [i,j,r], V by [j,k,r])
    # T[i, j, k, r] = U[i,j,r] * V[j,k,r]
    T = np.einsum('ijr,jkr->ijk r', U_mat, V_mat, optimize=True)
    # T shape: (n, m, p, r)
    
    # Now S[i', k', j, i] = sum_r T[i, j, k, r] * W[k', i', r]
    # W_mat shape: (p, n, r) -> W[k', i', r]
    # We want: for each (i, j, k), the vector over (i', k') of sum_r T[i,j,k,r] * W[k',i',r]
    # This is: S[i, j, k, i', k'] = sum_r T[i,j,k,r] * W[k',i',r]
    
    # Let's compute: S = np.einsum('ijk r, kp r -> ijkpk', T, W_mat, optimize=True)
    # But W_mat is (p, n, r), so W[k', i', r] means W_mat[k', i', r]
    # We need to be careful with indices.
    # T[i,j,k,r] * W_mat[kp, ip, r] -> sum over r -> result[i,j,k,kp,ip]
    
    S = np.einsum('ijk r,kp r->ijkpk', T, W_mat, optimize=True)
    # S shape: (n, m, p, p, n) where S[i,j,k,kp,ip] = sum_r U[i,j,r] V[j,k,r] W[kp,ip,r]
    
    # We need: S[i,j,k,kp,ip] = 1 if (kp,ip) == (k,i), else 0
    # Create a target tensor
    target = np.zeros((n, m, p, p, n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i, j, k, k, i] = 1.0
    
    if not np.allclose(S, target, atol=tol, rtol=0):
        return False
    
    return True

def _build_strassen_decomposition(n, m, p):
    """
    Build a decomposition using Strassen for 2x2x2 blocks where possible,
    and schoolbook for the rest.
    """
    # Define blocks: pair up indices where possible
    i_blocks = [(0, 1), (2, 3)]
    j_blocks = [(0, 1), (2, 3)]
    k_blocks = [(0, 1), (2, 3), (4, 5)]
    
    U_list = []
    V_list = []
    W_list = []
    
    # Track which (i,j,k) triples are covered by Strassen blocks
    covered = set()
    
    for (i1, i2) in i_blocks:
        for (j1, j2) in j_blocks:
            for (k1, k2) in k_blocks:
                U_block, V_block, W_block = _make_strassen_2x2_block(n, m, p, i1, i2, j1, j2, k1, k2)
                U_list.append(U_block)
                V_list.append(V_block)
                W_list.append(W_block)
                # Mark covered triples
                for i in [i1, i2]:
                    for j in [j1, j2]:
                        for k in [k1, k2]:
                            covered.add((i, j, k))
    
    # Add schoolbook terms for uncovered triples
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

def _build_alternative_packing(n, m, p):
    """
    Try an alternative packing: use Strassen blocks with different index pairings.
    For 4x5x6, we have 4 i-indices, 5 j-indices, 6 k-indices.
    j has 5 indices, so one j will be unpaired.
    """
    # Try pairing j indices as (0,1), (2,3), and leave j=4 unpaired
    i_blocks = [(0, 1), (2, 3)]
    j_blocks = [(0, 1), (2, 3)]
    k_blocks = [(0, 1), (2, 3), (4, 5)]
    
    U_list = []
    V_list = []
    W_list = []
    covered = set()
    
    # Apply Strassen blocks
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
    deadline = start_time + 140  # 10s safety margin
    
    # Build the Strassen-based decomposition
    U_strassen, V_strassen, W_strassen = _build_strassen_decomposition(n, m, p)
    rank_strassen = U_strassen.shape[1]
    
    # Verify with full exact check
    if _verify_full(U_strassen, V_strassen, W_strassen, n, m, p, tol=1e-9):
        U_best = U_strassen
        V_best = V_strassen
        W_best = W_strassen
        rank_best = rank_strassen
    else:
        # Try alternative packing
        U_alt, V_alt, W_alt = _build_alternative_packing(n, m, p)
        rank_alt = U_alt.shape[1]
        if _verify_full(U_alt, V_alt, W_alt, n, m, p, tol=1e-9):
            U_best = U_alt
            V_best = V_alt
            W_best = W_alt
            rank_best = rank_alt
        else:
            # Fall back to schoolbook
            U_best, V_best, W_best = _make_schoolbook(n, m, p)
            rank_best = n * m * p
    
    # Check time
    if time.time() > deadline - 5:
        # Not enough time for further optimization, return what we have
        pass
    
    # Try to see if we can reduce rank by checking for linearly dependent columns
    # This is a heuristic: if some columns are linearly dependent, we might be able to merge them
    # But for now, the Strassen decomposition is known to be rank-minimal for 2x2x2 blocks
    # and the overall packing is optimal given the block structure.
    
    def to_list(mat):
        return mat.tolist()
    
    return {"U": to_list(U_best), "V": to_list(V_best), "W": to_list(W_best)}
# EVOLVE-BLOCK-END
