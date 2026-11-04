import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 4, 8]

def _exact_rank_and_pivots(M):
    """
    Compute the exact rank of an integer matrix M and return the pivot columns.
    Uses Fraction to avoid floating point errors.
    """
    if M.size == 0:
        return 0, []
    
    M_frac = np.array([[Fraction(x) for x in row] for row in M], dtype=object)
    m, n = M_frac.shape
    rank = 0
    pivots = []
    
    for col in range(n):
        if rank >= m:
            break
        # Find a non-zero pivot in this column
        pivot_row = -1
        for row in range(rank, m):
            if M_frac[row, col] != 0:
                pivot_row = row
                break
        
        if pivot_row == -1:
            continue
        
        # Swap rows
        if pivot_row != rank:
            M_frac[[rank, pivot_row]] = M_frac[[pivot_row, rank]]
        
        pivots.append(col)
        
        # Eliminate below
        pivot_val = M_frac[rank, col]
        for row in range(rank + 1, m):
            if M_frac[row, col] != 0:
                factor = M_frac[row, col] / pivot_val
                for c in range(col, n):
                    M_frac[row, c] -= factor * M_frac[rank, c]
        
        rank += 1
        
    return rank, pivots

def _qr_with_pivoting(A):
    """QR decomposition with column pivoting."""
    m, n = A.shape
    Q = np.eye(m)
    R = A.copy()
    P = list(range(n))
    
    for k in range(min(m, n)):
        # Find column with largest norm in remaining columns
        if k < n:
            col_norms = np.linalg.norm(R[k:, k:], axis=0)
            max_col = np.argmax(col_norms) + k
            if max_col != k:
                R[:, [k, max_col]] = R[:, [max_col, k]]
                Q[:, [k, max_col]] = Q[:, [max_col, k]]
                P[k], P[max_col] = P[max_col], P[k]
        
        # Householder reflection
        x = R[k:, k]
        if np.linalg.norm(x) > 0:
            v = x.copy()
            v[0] += np.sign(v[0]) * np.linalg.norm(v)
            v = v / np.linalg.norm(v)
            R[k:, k:] -= 2 * np.outer(v, v @ R[k:, k:])
            Q[k:, :] -= 2 * np.outer(v, v @ Q[k:, :])
    
    return Q, R, P

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition correctly computes matrix multiplication."""
    test_matrices_A = [
        np.eye(n, m),
        np.ones((n, m)),
        np.arange(n * m, dtype=float).reshape(n, m) / (n * m),
        np.random.RandomState(42).randn(n, m),
        np.random.RandomState(123).randn(n, m),
    ]
    test_matrices_B = [
        np.eye(m, p),
        np.ones((m, p)),
        np.arange(m * p, dtype=float).reshape(m, p) / (m * p),
        np.random.RandomState(42).randn(m, p),
        np.random.RandomState(123).randn(m, p),
    ]
    
    for A in test_matrices_A:
        for B in test_matrices_B:
            C_expected = A @ B
            C_computed = np.zeros((n, p))
            for c in range(U.shape[1]):
                sa = np.dot(U[:, c], A.flatten())
                sb = np.dot(V[:, c], B.flatten())
                C_computed += sa * sb * W[:, c].reshape(n, p)
            
            if not np.allclose(C_computed, C_expected, atol=1e-9):
                return False
    
    return True

def construct():
    deadline = time.time() + 110
    n, m, p = DIMENSIONS
    
    # Build Strassen 2x2x2 blocks
    n_blocks_i = 1
    n_blocks_j = 2
    n_blocks_k = 4
    total_blocks = n_blocks_i * n_blocks_j * n_blocks_k
    rank = total_blocks * 7  # 56
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for bi in range(n_blocks_i):
        for bj in range(n_blocks_j):
            for bk in range(n_blocks_k):
                i0 = 2 * bi
                i1 = 2 * bi + 1
                j0 = 2 * bj
                j1 = 2 * bj + 1
                k0 = 2 * bk
                k1 = 2 * bk + 1
                
                u_a = i0 * m + j0
                u_b = i0 * m + j1
                u_c = i1 * m + j0
                u_d = i1 * m + j1
                
                v_e = j0 * p + k0
                v_f = j0 * p + k1
                v_g = j1 * p + k0
                v_h = j1 * p + k1
                
                w_c00 = k0 * n + i0
                w_c01 = k1 * n + i0
                w_c10 = k0 * n + i1
                w_c11 = k1 * n + i1
                
                # M1 = (a+d)(e+h)
                U[u_a][col] += 1; U[u_d][col] += 1
                V[v_e][col] += 1; V[v_h][col] += 1
                W[w_c00][col] += 1; W[w_c11][col] += 1
                col += 1
                
                # M2 = (c+d)e
                U[u_c][col] += 1; U[u_d][col] += 1
                V[v_e][col] += 1
                W[w_c10][col] += 1; W[w_c11][col] -= 1
                col += 1
                
                # M3 = a(f-h)
                U[u_a][col] += 1
                V[v_f][col] += 1; V[v_h][col] -= 1
                W[w_c01][col] += 1; W[w_c11][col] += 1
                col += 1
                
                # M4 = d(g-e)
                U[u_d][col] += 1
                V[v_g][col] += 1; V[v_e][col] -= 1
                W[w_c00][col] += 1; W[w_c10][col] += 1
                col += 1
                
                # M5 = (a+b)h
                U[u_a][col] += 1; U[u_b][col] += 1
                V[v_h][col] += 1
                W[w_c00][col] -= 1; W[w_c01][col] += 1
                col += 1
                
                # M6 = (c-a)(e+f)
                U[u_c][col] += 1; U[u_a][col] -= 1
                V[v_e][col] += 1; V[v_f][col] += 1
                W[w_c11][col] += 1
                col += 1
                
                # M7 = (b-d)(g+h)
                U[u_b][col] += 1; U[u_d][col] -= 1
                V[v_g][col] += 1; V[v_h][col] += 1
                W[w_c00][col] += 1
                col += 1
    
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Stacked matrix for dependency check
    stacked = np.vstack([U_np, V_np, W_np])
    
    # Use QR with column pivoting to find independent columns
    try:
        Q, R, P = _qr_with_pivoting(stacked)
        tol = 1e-10
        r = np.sum(np.abs(np.diag(R)) > tol)
        
        if r < rank:
            # Get the pivot columns (first r columns after permutation)
            pivot_cols = P[:r]
            
            U_reduced = U_np[:, pivot_cols]
            V_reduced = V_np[:, pivot_cols]
            W_reduced = W_np[:, pivot_cols]
            
            # Verify with multiple test cases
            if _verify_decomposition(U_reduced, V_reduced, W_reduced, n, m, p):
                return {"U": U_reduced.tolist(), "V": V_reduced.tolist(), "W": W_reduced.tolist()}
    except Exception:
        pass
    
    # Try exact integer rank reduction (lattice-like approach)
    try:
        # Convert to integers (they are already integers)
        U_int = np.array(U, dtype=int)
        V_int = np.array(V, dtype=int)
        W_int = np.array(W, dtype=int)
        stacked_int = np.vstack([U_int, V_int, W_int])
        
        r_exact, pivots_exact = _exact_rank_and_pivots(stacked_int)
        
        if r_exact < rank:
            U_reduced_exact = U_int[:, pivots_exact]
            V_reduced_exact = V_int[:, pivots_exact]
            W_reduced_exact = W_int[:, pivots_exact]
            
            if _verify_decomposition(U_reduced_exact, V_reduced_exact, W_reduced_exact, n, m, p):
                return {"U": U_reduced_exact.tolist(), "V": V_reduced_exact.tolist(), "W": W_reduced_exact.tolist()}
    except Exception:
        pass
    
    # Fallback: return the full decomposition
    return {"U": U, "V": V, "W": W}
