import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 2, 10]
_DEADLINE = 120.0

def _verify(U, V, W, n, m, p):
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    recon = np.einsum('ir,jr,kr->ijk', U, V, W)
    return bool(np.allclose(recon, T, atol=1e-9, rtol=0.0))

_STRASSEN = [
    ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
    ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
    ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),
    ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
    ([(0, 1), (1, 1)], [(3, 1)], [(1, 1), (0, -1)]),
    ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
    ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
]

def _emit_strassen(U, V, W, n, p, base_col, c0, c1):
    for t in range(7):
        col = base_col + t
        u_side, v_side, w_side = _STRASSEN[t]
        for u_row, coeff in u_side:
            U[u_row][col] += coeff
        for v_koff, coeff in v_side:
            j = v_koff // 2
            k = c0 if v_koff % 2 == 0 else c1
            V[j * p + k][col] += coeff
        for w_koff, coeff in w_side:
            i = w_koff // 2
            k = c0 if w_koff % 2 == 0 else c1
            W[k * n + i][col] += coeff

def _find_rank12_2x2x4():
    """
    Attempt to find a rank-12 decomposition for a 2x2x4 block.
    """
    T = np.zeros((4, 8, 8))
    for i in range(2):
        for j in range(2):
            for k in range(4):
                T[i, j * 4 + k, k * 2 + i] = 1.0

    # We use a randomized search with alternating least squares.
    # We initialize U, V, W with random values and iterate.
    # We then round to the nearest half-integer and verify.
    
    np.random.seed(0)
    best_error = float('inf')
    best_solution = None
    
    for attempt in range(50):
        U = np.random.randn(4, 12)
        V = np.random.randn(8, 12)
        W = np.random.randn(8, 12)
        
        for iter in range(200):
            # Update U
            # T = sum_r u_r (v_r w_r^T)^T
            # T_reshaped = T.reshape(4, 64)
            # A = sum_r (v_r w_r^T).reshape(64)
            # U = T_reshaped A^T (A A^T)^-1
            A = np.zeros((64, 12))
            for r in range(12):
                A[:, r] = (V[:, r] * W[:, r][:, None]).flatten()
            try:
                U = np.linalg.lstsq(A.T, T.reshape(4, 64).T, rcond=None)[0].T
            except:
                continue
            
            # Update V
            B = np.zeros((64, 12))
            for r in range(12):
                B[:, r] = (U[:, r] * W[:, r][:, None]).flatten()
            try:
                V = np.linalg.lstsq(B.T, T.reshape(8, 32).T, rcond=None)[0].T
            except:
                continue
            
            # Update W
            C = np.zeros((32, 12))
            for r in range(12):
                C[:, r] = (U[:, r] * V[:, r][:, None]).flatten()
            try:
                W = np.linalg.lstsq(C.T, T.reshape(8, 32).T, rcond=None)[0].T
            except:
                continue
        
        # Round to half-integers
        U_rounded = np.round(U * 2) / 2
        V_rounded = np.round(V * 2) / 2
        W_rounded = np.round(W * 2) / 2
        
        recon = np.einsum('ir,jr,kr->ijk', U_rounded, V_rounded, W_rounded)
        error = np.linalg.norm(recon - T)
        
        if error < 1e-6:
            return U_rounded, V_rounded, W_rounded
        
        if error < best_error:
            best_error = error
            best_solution = (U_rounded, V_rounded, W_rounded)
    
    return None

def construct():
    t0 = time.time()
    n, m, p = DIMENSIONS  # 2, 2, 10

    # Try to find a rank-12 2x2x4 block
    rank12_block = _find_rank12_2x2x4()
    
    if rank12_block is not None:
        U12, V12, W12 = rank12_block
        rank = 12 + 12 + 7  # 31
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        # Emit first 2x2x4 block
        for r in range(12):
            for i in range(4):
                U[i][r] += U12[i, r]
            for j in range(8):
                # Map local V index to global
                # Local: 0-3 are row 0 of B, 4-7 are row 1 of B
                # Global: j*p + k
                if j < 4:
                    row = 0
                    col_idx = j
                else:
                    row = 1
                    col_idx = j - 4
                V[row * p + col_idx][r] += V12[j, r]
            for k in range(8):
                # Map local W index to global
                # Local: 0-3 are row 0 of C, 4-7 are row 1 of C
                # Global: k*n + i
                if k < 4:
                    row = 0
                    col_idx = k
                else:
                    row = 1
                    col_idx = k - 4
                W[col_idx * n + row][r] += W12[k, r]
        
        # Emit second 2x2x4 block
        for r in range(12):
            col = 12 + r
            for i in range(4):
                U[i][col] += U12[i, r]
            for j in range(8):
                if j < 4:
                    row = 0
                    col_idx = j + 4
                else:
                    row = 1
                    col_idx = j - 4 + 4
                V[row * p + col_idx][col] += V12[j, r]
            for k in range(8):
                if k < 4:
                    row = 0
                    col_idx = k + 4
                else:
                    row = 1
                    col_idx = k - 4 + 4
                W[col_idx * n + row][col] += W12[k, r]
        
        # Emit Strassen block
        _emit_strassen(U, V, W, n, p, 24, 8, 9)
        
        if _verify(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}

    # Fallback: safe rank-35 baseline
    rank = 35
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    for b in range(5):
        _emit_strassen(U, V, W, n, p, b * 7, 2 * b, 2 * b + 1)

    if not _verify(U, V, W, n, m, p):
        raise RuntimeError("rank-35 fallback failed verification")
    return {"U": U, "V": V, "W": W}
