"""Tensor decomposition for 2x3 matrix multiplication (A@B, B is 3x10). Column elimination search."""
import time
import numpy as np

def _build_rank55():
    n, m, p = 2, 3, 10
    rank = 55
    U = np.zeros((n*m, rank), dtype=int)  # 6 x 55, indexed by (i*m+j)
    V = np.zeros((m*p, rank), dtype=int)  # 30 x 55, indexed by (j*p+k)
    W = np.zeros((p*n, rank), dtype=int)  # 20 x 55, indexed by (k*n+i)
    col = 0
    for t in range(5):
        k0 = 2 * t
        k1 = 2 * t + 1
        # Term 1: (A00+A11)*(B0k0+B1k1)
        U[0, col] += 1; U[4, col] += 1
        V[0*p+k0, col] += 1; V[1*p+k1, col] += 1
        W[k0*n+0, col] += 1; W[k1*n+1, col] += 1
        col += 1
        # Term 2: (A10+A11)*B0k0
        U[3, col] += 1; U[4, col] += 1
        V[0*p+k0, col] += 1
        W[k0*n+1, col] += 1; W[k1*n+1, col] -= 1
        col += 1
        # Term 3: A00*(B0k1-B1k1)
        U[0, col] += 1
        V[0*p+k1, col] += 1; V[1*p+k1, col] -= 1
        W[k1*n+0, col] += 1; W[k1*n+1, col] += 1
        col += 1
        # Term 4: A11*(B1k0-B0k0)
        U[4, col] += 1
        V[1*p+k0, col] += 1; V[0*p+k0, col] -= 1
        W[k0*n+0, col] += 1; W[k0*n+1, col] += 1
        col += 1
        # Term 5: (A00+A01)*B1k1
        U[0, col] += 1; U[1, col] += 1
        V[1*p+k1, col] += 1
        W[k0*n+0, col] -= 1; W[k1*n+0, col] += 1
        col += 1
        # Term 6: (A10-A00)*(B0k0+B0k1)
        U[3, col] += 1; U[0, col] -= 1
        V[0*p+k0, col] += 1; V[0*p+k1, col] += 1
        W[k1*n+1, col] += 1
        col += 1
        # Term 7: (A01-A11)*(B1k0+B1k1)
        U[1, col] += 1; U[4, col] -= 1
        V[1*p+k0, col] += 1; V[1*p+k1, col] += 1
        W[k0*n+0, col] += 1
        col += 1
        # Terms 8-11: direct for row 2 of B (j=2)
        U[2, col] += 1; V[2*p+k0, col] += 1; W[k0*n+0, col] += 1
        col += 1
        U[2, col] += 1; V[2*p+k1, col] += 1; W[k1*n+0, col] += 1
        col += 1
        U[5, col] += 1; V[2*p+k0, col] += 1; W[k0*n+1, col] += 1
        col += 1
        U[5, col] += 1; V[2*p+k1, col] += 1; W[k1*n+1, col] += 1
        col += 1
    return U, V, W

def _verify(U, V, W):
    n, m, p = 2, 3, 10
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = np.dot(U[i*m+j, :], np.multiply(V[j*p+k, :], W[k*n+i, :]))
                if abs(total - 1) > 1e-8:
                    return False
    return True

def construct():
    _t0 = time.time()
    deadline = _t0 + 100
    
    U, V, W = _build_rank55()
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = U.shape[1]
    
    # Verify base
    assert _verify(best_U, best_V, best_W), "Base rank-55 decomposition failed verification"
    
    # Column elimination search
    for iteration in range(20):
        if time.time() > deadline:
            break
        eliminated = False
        for col_idx in range(best_U.shape[1]):
            if time.time() > deadline:
                break
            rank = best_U.shape[1]
            if rank <= 1:
                break
            
            other_cols = [r for r in range(rank) if r != col_idx]
            A_mat = np.vstack([
                best_U[:, other_cols].T,
                best_V[:, other_cols].T,
                best_W[:, other_cols].T
            ])
            b_vec = np.concatenate([
                best_U[:, col_idx],
                best_V[:, col_idx],
                best_W[:, col_idx]
            ])
            
            try:
                sol, residuals, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                if len(residuals) == 0 or residuals[0] < 1e-10:
                    A_mat_sol = A_mat @ sol
                    if np.max(np.abs(A_mat_sol - b_vec)) < 1e-10:
                        new_U = np.delete(best_U, col_idx, axis=1)
                        new_V = np.delete(best_V, col_idx, axis=1)
                        new_W = np.delete(best_W, col_idx, axis=1)
                        for idx, r in enumerate(other_cols):
                            new_U[:, r] += sol[idx] * best_U[:, r]
                            new_V[:, r] += sol[idx] * best_V[:, r]
                            new_W[:, r] += sol[idx] * best_W[:, r]
                        if _verify(new_U, new_V, new_W):
                            best_U, best_V, best_W = new_U, new_V, new_W
                            best_rank = best_U.shape[1]
                            eliminated = True
                            break
            except:
                pass
        if not eliminated:
            break
    
    def to_list(mat):
        lst = []
        for row in mat:
            lst_row = [round(val * 2) / 2 for val in row]
            lst.append(lst_row)
        return lst
    
    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
