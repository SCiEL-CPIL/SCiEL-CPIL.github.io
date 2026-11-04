import time
import numpy as np

DIMENSIONS = [2, 4, 8]

def _build_rank56():
    """Build the standard rank-56 Strassen decomposition for 2x4 * 4x8."""
    n, m, p = 2, 4, 8
    U = np.zeros((n * m, 56), dtype=np.int8)
    V = np.zeros((m * p, 56), dtype=np.int8)
    W = np.zeros((p * n, 56), dtype=np.int8)
    
    col = 0
    for bi in range(1):
        for bj in range(2):
            for bk in range(4):
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
                U[u_a, col] += 1; U[u_d, col] += 1
                V[v_e, col] += 1; V[v_h, col] += 1
                W[w_c00, col] += 1; W[w_c11, col] += 1
                col += 1
                
                # M2 = (c+d)e
                U[u_c, col] += 1; U[u_d, col] += 1
                V[v_e, col] += 1
                W[w_c10, col] += 1; W[w_c11, col] -= 1
                col += 1
                
                # M3 = a(f-h)
                U[u_a, col] += 1
                V[v_f, col] += 1; V[v_h, col] -= 1
                W[w_c01, col] += 1; W[w_c11, col] += 1
                col += 1
                
                # M4 = d(g-e)
                U[u_d, col] += 1
                V[v_g, col] += 1; V[v_e, col] -= 1
                W[w_c00, col] += 1; W[w_c10, col] += 1
                col += 1
                
                # M5 = (a+b)h
                U[u_a, col] += 1; U[u_b, col] += 1
                V[v_h, col] += 1
                W[w_c00, col] -= 1; W[w_c01, col] += 1
                col += 1
                
                # M6 = (c-a)(e+f)
                U[u_c, col] += 1; U[u_a, col] -= 1
                V[v_e, col] += 1; V[v_f, col] += 1
                W[w_c11, col] += 1
                col += 1
                
                # M7 = (b-d)(g+h)
                U[u_b, col] += 1; U[u_d, col] -= 1
                V[v_g, col] += 1; V[v_h, col] += 1
                W[w_c00, col] += 1
                col += 1
    
    return U, V, W

def _verify_decomposition(U, V, W, n=2, m=4, p=8):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(U.shape[1]):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _prune_columns(U, V, W, n=2, m=4, p=8):
    """Try to find and remove linearly dependent columns to reduce rank."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    
    rank = U.shape[1]
    if rank <= 1:
        return U, V, W
    
    n, m, p = 2, 4, 8
    dim = n * m * m * p * p * n  # 4096
    
    # Build the matrix whose columns are the flattened outer products
    M = np.zeros((dim, rank), dtype=np.float64)
    for r in range(rank):
        outer = np.outer(np.outer(U[:, r], V[:, r]), W[:, r]).flatten()
        M[:, r] = outer
    
    # Find the rank of M and identify which columns are in the span of others
    try:
        U_svd, S, Vt = np.linalg.svd(M, full_matrices=False)
        tol = 1e-10
        effective_rank = np.sum(S > tol * S[0])
        
        if effective_rank < rank:
            selected = []
            basis = []
            for r in range(rank):
                col = M[:, r].copy()
                if len(basis) == 0:
                    selected.append(r)
                    basis.append(col.copy())
                else:
                    residual = col.copy()
                    for b in basis:
                        proj = np.dot(residual, b) / np.dot(b, b)
                        residual -= proj * b
                    if np.linalg.norm(residual) > tol:
                        selected.append(r)
                        basis.append(residual.copy())
            
            if len(selected) < rank:
                U_new = U[:, selected]
                V_new = V[:, selected]
                W_new = W[:, selected]
                return U_new, V_new, W_new
    except Exception:
        pass
    
    return U, V, W

def construct():
    deadline = time.time() + 110
    n, m, p = DIMENSIONS
    
    # Build the rank-56 decomposition
    U, V, W = _build_rank56()
    
    # Try to prune columns
    U, V, W = _prune_columns(U, V, W, n, m, p)
    
    # Verify the decomposition is still exact
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fall back to the original rank-56
        U, V, W = _build_rank56()
    
    rank = U.shape[1]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
