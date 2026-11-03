"""Rank-44 exact decomposition for 2x3 * 3x8 matrix multiplication.

Uses Strassen (rank 7) on 2x2x2 subblocks and schoolbook (rank 2) on
2x1x8 subblocks.  A deadline-guarded smooth min-norm polish is attempted
to see whether a lower rank is achievable; if not, the verified rank-44
solution is returned.
"""
import time
import numpy as np

_DIMS = (2, 3, 8)
_DEADLINE = 140.0          # wall-clock seconds (evaluator kills at 150 s)
_MARGIN = 5.0              # safety margin before the kill
_DEADLINE = _DEADLINE - _MARGIN


def _matmul_tensor(n, m, p):
    """Return the (n*m, m*p, p*n) multiplication tensor T[i*m+j, j*p+k, k*n+i]."""
    T = np.zeros((n * m, m * p, p * n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1
    return T


def _strassen_2x2x2():
    """Return (U,V,W) each of shape (4,7) for the 2x2x2 multiplication tensor."""
    U = np.zeros((4, 7), dtype=np.int64)
    V = np.zeros((4, 7), dtype=np.int64)
    W = np.zeros((4, 7), dtype=np.int64)
    # Strassen's 7 products:
    # M1 = (a11+a22)*(b11+b22)
    # M2 = (a21+a22)*b11
    # M3 = a11*(b12-b22)
    # M4 = a22*(b21-b11)
    # M5 = (a11+a12)*b22
    # M6 = (a21-a11)*(b11+b12)
    # M7 = (a12-a22)*(b21+b22)
    # Output c11 = M1 - M4 + M5 + M7
    #          c12 = M3 + M5
    #          c21 = M2 + M4
    #          c22 = M1 - M3 + M2 + M6
    # U rows: (a11, a12, a21, a22), V rows: (b11, b12, b21, b22)
    # W rows: (c11, c12, c21, c22)
    # M1: a11+a22, b11+b22
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    # M2: a21+a22, b11
    U[1, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = 1
    # M3: a11, b12-b22
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = -1
    # M4: a22, b21-b11
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = -1; W[2, 3] = 1
    # M5: a11+a12, b22
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = 1; W[1, 4] = 1
    # M6: a21-a11, b11+b12
    U[1, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    # M7: a12-a22, b21+b22
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    return U, V, W


def _build_rank44():
    """Build the rank-44 decomposition.

    Decomposition strategy:
    - 2x3 * 3x8:  split the 3x3 middle into a 2x2 block (rows 0-1, cols 0-1)
      and a 1x3 block (row 2, cols 0-2).
    - The 2x2 block: 2x2 * 2x2 * 2x2 -> Strassen rank 7.
      This handles A[0:2, 0:2] * B[0:2, 0:2] contributing to C[0:2, 0:2].
    - The 1x3 block: 2x1 * 1x3 * 3x2 -> schoolbook rank 2*1*3 = 6.
      This handles A[0:2, 2] * B[2, 0:3] contributing to C[0:2, 0:3].
    - Also need A[0:2, 0:2] * B[0:2, 2] -> 2x2 * 2x1 * 1x2 = 2*2*1 = 4.
    - And A[0:2, 2] * B[2, 0:2] -> 2x1 * 1x2 * 2x2 = 2*1*2 = 4.
    
    Wait, let me think more carefully.
    A is 2x3: rows i in {0,1}, cols j in {0,1,2}
    B is 3x8: rows j in {0,1,2}, cols k in {0,...,7}
    C is 2x8: rows i in {0,1}, cols k in {0,...,7}
    C[i,k] = sum_j A[i,j] * B[j,k]

    Split j into {0,1} (group 1) and {2} (group 2).
    
    Group 1 (j in {0,1}): A[:, 0:2] is 2x2, B[0:2, :] is 2x8.
    Product: 2x2 * 2x8. Split k into 4 groups of 2: k in {0,1}, {2,3}, {4,5}, {6,7}.
    Each sub-block: 2x2 * 2x2 -> Strassen rank 7.
    Total: 4 * 7 = 28.
    
    Group 2 (j = 2): A[:, 2] is 2x1, B[2, :] is 1x8.
    Product: 2x1 * 1x8. Schoolbook: 2*1*8 = 16.
    Total: 16.
    
    Grand total: 28 + 16 = 44.
    """
    n, m, p = _DIMS  # 2, 3, 8
    # U shape: (n*m, rank) = (6, rank)
    # V shape: (m*p, rank) = (24, rank)
    # W shape: (p*n, rank) = (16, rank)
    
    U = np.zeros((n * m, 44), dtype=np.int64)
    V = np.zeros((m * p, 44), dtype=np.int64)
    W = np.zeros((p * n, 44), dtype=np.int64)
    
    col = 0
    # Group 1: j in {0,1}, split k into 4 pairs
    for k0 in range(0, p, 2):
        k1 = k0 + 1
        sU, sV, sW = _strassen_2x2x2()
        # A indices: i in {0,1}, j in {0,1} -> row i*m + j = i*3 + j
        # B indices: j in {0,1}, k in {k0, k1} -> row j*p + k = j*8 + k
        # C indices: k in {k0, k1}, i in {0,1} -> row k*n + i = k*2 + i
        
        # Map Strassen's 2x2 indices to actual indices
        # Strassen A rows: a11->(i=0,j=0), a12->(i=0,j=1), a21->(i=1,j=0), a22->(i=1,j=1)
        # Strassen B rows: b11->(j=0,k=k0), b12->(j=0,k=k1), b21->(j=1,k=k0), b22->(j=1,k=k1)
        # Strassen C rows: c11->(k=k0,i=0), c12->(k=k1,i=0), c21->(k=k0,i=1), c22->(k=k1,i=1)
        
        a_map = [0 * m + 0, 0 * m + 1, 1 * m + 0, 1 * m + 1]  # (0,0),(0,1),(1,0),(1,1)
        b_map = [0 * p + k0, 0 * p + k1, 1 * p + k0, 1 * p + k1]  # (0,k0),(0,k1),(1,k0),(1,k1)
        c_map = [k0 * n + 0, k1 * n + 0, k0 * n + 1, k1 * n + 1]  # (k0,0),(k1,0),(k0,1),(k1,1)
        
        for c in range(7):
            for r in range(4):
                if sU[r, c] != 0:
                    U[a_map[r], col] += sU[r, c]
                if sV[r, c] != 0:
                    V[b_map[r], col] += sV[r, c]
                if sW[r, c] != 0:
                    W[c_map[r], col] += sW[r, c]
            col += 1
    
    # Group 2: j = 2, k in {0,...,7}
    # A[:, 2] is 2x1, B[2, :] is 1x8
    # Schoolbook: for each i in {0,1}, k in {0,...,7}:
    for i in range(n):
        for k in range(p):
            U[i * m + 2, col] = 1       # A[i, 2]
            V[2 * p + k, col] = 1       # B[2, k]
            W[k * n + i, col] = 1       # C[k, i] -> wait, C[i,k] so index is k*n + i
            col += 1
    
    return U, V, W


def _verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly reproduces the matmul tensor."""
    T = _matmul_tensor(n, m, p)
    rank = U.shape[1]
    # Compute sum over c of U[:,c] outer V[:,c] outer W[:,c]
    # This is expensive for large rank, but rank=44 is fine.
    result = np.zeros_like(T)
    for c in range(rank):
        result += np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])
    return np.array_equal(result, T)


def _try_polish(U, V, W, n, m, p, max_time):
    """Attempt to find a lower-rank decomposition via smooth min-norm refinement.
    
    Uses a soft-min/epigraph approach: minimize the sum of squared deviations
    from the tensor with a soft constraint on rank. This is a heuristic and
    may not find a lower rank, but it's worth trying within the time budget.
    """
    T = _matmul_tensor(n, m, p)
    nm, mp, pn = n * m, m * p, p * n
    
    # Try ranks from 43 down to 30 (theoretical lower bound for 2x3x8 is at least
    # max(2*3, 3*8, 8*2) = 24, but in practice rank is likely 40+)
    start_time = time.time()
    best_rank = 44
    best_sol = (U, V, W)
    
    for target_rank in range(43, 23, -1):
        if time.time() - start_time > max_time:
            break
        # Initialize with random + noise
        rng = np.random.default_rng(target_rank)
        U_t = rng.normal(size=(nm, target_rank))
        V_t = rng.normal(size=(mp, target_rank))
        W_t = rng.normal(size=(pn, target_rank))
        
        # Smooth objective: sum of (T_ijk - sum_c U_ic V_jc W_kc)^2
        # Use gradient descent with a few iterations
        lr = 0.01
        for step in range(200):
            if time.time() - start_time > max_time:
                break
            # Compute current tensor
            current = np.einsum('ic,jc,kc->ijk', U_t, V_t, W_t)
            diff = current - T
            # Gradient w.r.t. U: diff @ (V^T @ W)
            grad_U = np.einsum('ijk,jc,kc->ic', diff, V_t, W_t)
            grad_V = np.einsum('ijk,ic,kc->jc', diff, U_t, W_t)
            grad_W = np.einsum('ijk,ic,jc->kc', diff, U_t, V_t)
            U_t -= lr * grad_U
            V_t -= lr * grad_V
            W_t -= lr * grad_W
        
        # Check if close to zero
        final_current = np.einsum('ic,jc,kc->ijk', U_t, V_t, W_t)
        err = np.max(np.abs(final_current - T))
        if err < 1e-6:
            # Round to nearest integer/half-integer
            U_t = np.round(2 * U_t) / 2
            V_t = np.round(2 * V_t) / 2
            W_t = np.round(2 * W_t) / 2
            if _verify(U_t, V_t, W_t, n, m, p):
                best_rank = target_rank
                best_sol = (U_t, V_t, W_t)
                break
    
    return best_rank, best_sol


def construct():
    n, m, p = _DIMS
    start = time.time()
    
    # Build the rank-44 solution
    U, V, W = _build_rank44()
    
    # Verify
    if not _verify(U, V, W, n, m, p):
        # Fallback to schoolbook if something went wrong
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=np.int64)
        V = np.zeros((m * p, rank), dtype=np.int64)
        W = np.zeros((p * n, rank), dtype=np.int64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to find a lower rank within the time budget
    remaining = _DEADLINE - (time.time() - start)
    if remaining > 10:
        best_rank, best_sol = _try_polish(U, V, W, n, m, p, remaining - 5)
        if best_rank < 44:
            U, V, W = best_sol
    
    # Convert to lists (the interface expects lists per the seed program)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
