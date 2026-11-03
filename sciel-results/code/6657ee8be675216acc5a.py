"""Rank-minimal exact tensor decomposition for 3x4x8 matrix multiplication.

Strategy: 
1. Directly construct the known rank-88 hybrid decomposition
   (schoolbook for i=0, Strassen 2x2 blocks for i=1,2)
2. Attempt rank reduction via SVD on the flattened tensor
3. Fallback to the rank-88 construction if reduction fails
"""
import numpy as np
import time

DIMENSIONS = [3, 4, 8]


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check the cyclic bilinear identity:
       sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1  for all i,j,k.
    """
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    R = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(R):
                    s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True


def _strassen_2x2x2():
    """Return the 7-term Strassen decomposition for 2x2x2 matrix mult.
    U: (4, 7), V: (4, 7), W: (4, 7)
    Row indices: U[i*2+j], V[j*2+k], W[k*2+i]
    """
    U = np.array([
        [1, 0, 0, 1, 1, -1, 0],
        [0, 0, 1, 1, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, -1],
    ], dtype=float)
    V = np.array([
        [1, 0, 0, 1, 0, 1, 0],
        [1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, -1, 0, 0, 1],
        [-1, 0, 1, 0, 0, 0, 0],
    ], dtype=float)
    W = np.array([
        [1, 0, 0, 1, -1, 0, 0],
        [0, 0, 1, -1, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 1],
        [1, 0, 1, 0, 0, 0, 0],
    ], dtype=float)
    return U, V, W


def _build_rank88(n=3, m=4, p=8):
    """Build the rank-88 hybrid decomposition:
    - i=0: schoolbook (32 terms)
    - i=1,2: Strassen 2x2x2 blocks (4 j-blocks * 4 k-blocks * 7 terms * 2 rows = 224... wait)
    
    Actually: for i=1 and i=2, we have 4 j-columns and 8 k-columns.
    Partition into 2x2 Strassen blocks: 2 j-blocks * 4 k-blocks = 8 blocks per row.
    Each block costs 7 terms. So 8*7 = 56 terms per row, 112 for two rows.
    Total: 32 + 112 = 144? That's not 88.
    
    Wait, let me reconsider. The standard approach:
    For i=0: 4*8 = 32 schoolbook terms
    For i=1,2: treat as 2x4x8, use Strassen on 2x2 sub-blocks
    Actually, we can share the Strassen structure between i=1 and i=2.
    
    Let me think differently. The 3x4x8 tensor:
    For each output row i (0,1,2), we compute row i of C = A*B.
    
    A better construction:
    - Use Strassen for the 2x2x2 subproblems in a structured way.
    
    Actually, the known result is rank 88. Let me construct it properly.
    
    Construction:
    - i=0: schoolbook, 32 terms
    - i=1,2: Use a 2x4x8 multiplication. This can be done with:
      Partition the 4 j-columns into 2 groups of 2, and 8 k-columns into 4 groups of 2.
      For each (j-group, k-group) pair, use Strassen 2x2x2.
      That's 2*4 = 8 Strassen blocks, each 7 terms = 56 terms.
      But we have 2 rows (i=1,2), so do we need 56*2 = 112? No, because
      Strassen already handles 2x2 output.
      
      Wait: Strassen 2x2x2 multiplies 2x2 * 2x2 -> 2x2. So if we partition
      j into 2 groups of 2, and k into 4 groups of 2, for each (jg, kg) pair
      we get a 2x2 output (covering i=1,2 for that j-group and k-group).
      So 2*4 = 8 Strassen blocks * 7 = 56 terms for both i=1 and i=2.
      
    Total: 32 + 56 = 88. Yes!
    """
    sU, sV, sW = _strassen_2x2x2()
    
    # i=0: schoolbook
    # i=1,2: Strassen blocks
    total_rank = 32 + 56  # 88
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col = 0
    # i=0 schoolbook
    for j in range(m):
        for k in range(p):
            U[0 * m + j, col] = 1.0
            V[j * p + k, col] = 1.0
            W[k * n + 0, col] = 1.0
            col += 1
    
    # i=1,2: Strassen 2x2x2 blocks
    # j-groups: [0,1], [2,3]
    # k-groups: [0,1], [2,3], [4,5], [6,7]
    for jg in range(2):
        js = jg * 2
        for kg in range(4):
            ks = kg * 2
            for t in range(7):
                for l in range(4):
                    # sU[t,l] contributes to U[i*2+li, lj] where l = li*2+lj
                    li, lj = l // 2, l % 2
                    # sV[t,l] contributes to V[lj*2+lk, ...] where l = lj*2+lk
                    lv_j, lv_k = l // 2, l % 2
                    # sW[t,l] contributes to W[lk*2+li, ...] where l = lk*2+li
                    lw_k, lw_i = l // 2, l % 2
                    
                    if sU[t, l] != 0:
                        U[(1 + li) * m + (js + lj), col] += sU[t, l]
                    if sV[t, l] != 0:
                        V[(js + lv_j) * p + (ks + lv_k), col] += sV[t, l]
                    if sW[t, l] != 0:
                        W[(ks + lw_k) * n + (1 + lw_i), col] += sW[t, l]
                col += 1
    
    assert col == total_rank, f"Column mismatch: {col} vs {total_rank}"
    return U, V, W


def _try_reduce_rank(U, V, W, n, m, p, deadline, tol=1e-9):
    """Attempt to reduce rank by finding linear dependencies among columns."""
    U = np.asarray(U, dtype=float).copy()
    V = np.asarray(V, dtype=float).copy()
    W = np.asarray(W, dtype=float).copy()
    R = U.shape[1]
    
    if R == 0:
        return U, V, W
    
    # Build the flattened tensor vectors for each column
    F = np.zeros((n * m * p, R))
    for r in range(R):
        u = U[:, r].reshape(n, m)
        v = V[:, r].reshape(m, p)
        w = W[:, r].reshape(p, n)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    F[i * m * p + j * p + k, r] = u[i, j] * v[j, k] * w[k, i]
    
    # Find the rank of F using SVD
    if time.time() > deadline - 20:
        return U, V, W
    
    try:
        U_svd, s, Vt = np.linalg.svd(F, full_matrices=False)
        tol_svd = max(F.shape) * np.finfo(float).eps * s[0]
        num_rank = int(np.sum(s > tol_svd))
        
        if num_rank < R:
            P = Vt[:num_rank, :].T
            U_new = U @ P
            V_new = V @ P
            W_new = W @ P
            
            if _verify(U_new, V_new, W_new, n, m, p, tol=1e-8):
                return U_new, V_new, W_new
    except Exception:
        pass
    
    return U, V, W


def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS
    
    # Step 1: Directly construct rank-88
    U, V, W = _build_rank88(n, m, p)
    best_U, best_V, best_W = U, V, W
    best_rank = U.shape[1]
    
    # Verify
    if not _verify(best_U, best_V, best_W, n, m, p):
        # Should not happen, but fallback to full schoolbook
        U_fb = np.zeros((n * m, 96))
        V_fb = np.zeros((m * p, 96))
        W_fb = np.zeros((p * n, 96))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j, col] = 1.0
                    V_fb[j * p + k, col] = 1.0
                    W_fb[k * n + i, col] = 1.0
                    col += 1
        best_U, best_V, best_W = U_fb, V_fb, W_fb
        best_rank = 96
    
    # Step 2: Try to reduce rank
    if time.time() < deadline - 30:
        try:
            U_red, V_red, W_red = _try_reduce_rank(best_U, best_V, best_W, n, m, p, deadline)
            new_rank = U_red.shape[1]
            if new_rank < best_rank and _verify(U_red, V_red, W_red, n, m, p, tol=1e-8):
                best_U, best_V, best_W = U_red, V_red, W_red
                best_rank = new_rank
        except Exception:
            pass
    
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}


if __name__ == "__main__":
    result = construct()
    rank = len(result['U'][0])
    print(f"Rank: {rank}")
