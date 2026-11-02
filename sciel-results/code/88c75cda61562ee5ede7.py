"""Matrix multiplication 4,4,5 - Strassen block decomposition with proper rank reduction."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 130
    n, m, p = DIMENSIONS  # 4, 4, 5
    nm, mp, pn = n * m, m * p, p * n  # 16, 20, 20

    def time_left():
        return deadline - time.time()

    # Build decomposition using Strassen for 2x2x2 blocks and schoolbook for 2x2x1
    # Partition: rows {0,1} and {2,3}, cols {0,1}, {2,3}, {4}
    # C[0:2, 0:2] = A[0:2, 0:2] * B[0:2, 0:2] + A[0:2, 2:4] * B[2:4, 0:2]
    # etc.
    
    terms_U = []  # list of dicts: {index: coeff}
    terms_V = []
    terms_W = []

    def u_idx(i, j):
        return i * m + j

    def v_idx(j, k):
        return j * p + k

    def w_idx(k, i):
        return k * n + i

    def add_strassen_2x2x2(i0, j0, k0):
        """Add Strassen terms for A[i0:i0+2, j0:j0+2] * B[j0:j0+2, k0:k0+2] -> C[i0:i0+2, k0:k0+2]"""
        # M1 = (a+d)(e+h)
        u = {u_idx(i0, j0): 1, u_idx(i0+1, j0+1): 1}
        v = {v_idx(j0, k0): 1, v_idx(j0+1, k0+1): 1}
        w = {w_idx(k0, i0): 1, w_idx(k0+1, i0+1): 1}
        terms_U.append(u); terms_V.append(v); terms_W.append(w)
        
        # M2 = (c+d)e
        u = {u_idx(i0+1, j0): 1, u_idx(i0+1, j0+1): 1}
        v = {v_idx(j0, k0): 1}
        w = {w_idx(k0, i0+1): 1, w_idx(k0+1, i0+1): -1}
        terms_U.append(u); terms_V.append(v); terms_W.append(w)
        
        # M3 = a(f-h)
        u = {u_idx(i0, j0): 1}
        v = {v_idx(j0, k0+1): 1, v_idx(j0+1, k0+1): -1}
        w = {w_idx(k0+1, i0): 1, w_idx(k0+1, i0+1): 1}
        terms_U.append(u); terms_V.append(v); terms_W.append(w)
        
        # M4 = d(g-e)
        u = {u_idx(i0+1, j0+1): 1}
        v = {v_idx(j0+1, k0): 1, v_idx(j0, k0): -1}
        w = {w_idx(k0, i0): 1, w_idx(k0, i0+1): 1}
        terms_U.append(u); terms_V.append(v); terms_W.append(w)
        
        # M5 = (a+b)h
        u = {u_idx(i0, j0): 1, u_idx(i0, j0+1): 1}
        v = {v_idx(j0+1, k0+1): 1}
        w = {w_idx(k0, i0): -1, w_idx(k0+1, i0): 1}
        terms_U.append(u); terms_V.append(v); terms_W.append(w)
        
        # M6 = (c-a)(e+f)
        u = {u_idx(i0+1, j0): 1, u_idx(i0, j0): -1}
        v = {v_idx(j0, k0): 1, v_idx(j0, k0+1): 1}
        w = {w_idx(k0+1, i0+1): 1}
        terms_U.append(u); terms_V.append(v); terms_W.append(w)
        
        # M7 = (b-d)(g+h)
        u = {u_idx(i0, j0+1): 1, u_idx(i0+1, j0+1): -1}
        v = {v_idx(j0+1, k0): 1, v_idx(j0+1, k0+1): 1}
        w = {w_idx(k0, i0): 1}
        terms_U.append(u); terms_V.append(v); terms_W.append(w)

    def add_schoolbook_2x2x1(i0, j0, k0):
        """Add schoolbook terms for A[i0:i0+2, j0:j0+2] * B[j0:j0+2, k0] -> C[i0:i0+2, k0]"""
        for di in range(2):
            for dj in range(2):
                u = {u_idx(i0+di, j0+dj): 1}
                v = {v_idx(j0+dj, k0): 1}
                w = {w_idx(k0, i0+di): 1}
                terms_U.append(u); terms_V.append(v); terms_W.append(w)

    # Build the decomposition:
    # C has shape 4x5, so columns 0-4
    # Rows 0-3, split into {0,1} and {2,3}
    # Cols 0-4, split into {0,1}, {2,3}, {4}
    
    # C[0:2, 0:2] = A[0:2, 0:2]*B[0:2, 0:2] + A[0:2, 2:4]*B[2:4, 0:2]
    add_strassen_2x2x2(0, 0, 0)
    add_strassen_2x2x2(0, 2, 0)
    
    # C[0:2, 2:4] = A[0:2, 0:2]*B[0:2, 2:4] + A[0:2, 2:4]*B[2:4, 2:4]
    add_strassen_2x2x2(0, 0, 2)
    add_strassen_2x2x2(0, 2, 2)
    
    # C[0:2, 4] = A[0:2, 0:2]*B[0:2, 4] + A[0:2, 2:4]*B[2:4, 4]
    add_schoolbook_2x2x1(0, 0, 4)
    add_schoolbook_2x2x1(0, 2, 4)
    
    # C[2:4, 0:2] = A[2:4, 0:2]*B[0:2, 0:2] + A[2:4, 2:4]*B[2:4, 0:2]
    add_strassen_2x2x2(2, 0, 0)
    add_strassen_2x2x2(2, 2, 0)
    
    # C[2:4, 2:4] = A[2:4, 0:2]*B[0:2, 2:4] + A[2:4, 2:4]*B[2:4, 2:4]
    add_strassen_2x2x2(2, 0, 2)
    add_strassen_2x2x2(2, 2, 2)
    
    # C[2:4, 4] = A[2:4, 0:2]*B[0:2, 4] + A[2:4, 2:4]*B[2:4, 4]
    add_schoolbook_2x2x1(2, 0, 4)
    add_schoolbook_2x2x1(2, 2, 4)

    rank = len(terms_U)
    
    # Convert to matrices
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    for c in range(rank):
        for idx, coeff in terms_U[c].items():
            U[idx, c] = coeff
        for idx, coeff in terms_V[c].items():
            V[idx, c] = coeff
        for idx, coeff in terms_W[c].items():
            W[idx, c] = coeff

    # Verify correctness
    def verify(U, V, W):
        rng = np.random.default_rng(123)
        A = rng.random((n, m))
        B = rng.random((m, p))
        C_expected = A @ B
        C_recon = np.zeros((n, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    C_recon[i, k] += U[u_idx(i, j), :] @ V[v_idx(j, k), :] * W[w_idx(k, i), :]
        # More efficient: use einsum-like approach
        # Actually, let's just check a few random entries
        for trial in range(10):
            A_t = rng.random((n, m))
            B_t = rng.random((m, p))
            C_exp = A_t @ B_t
            C_rec = np.zeros((n, p))
            for i in range(n):
                for k in range(p):
                    for j in range(m):
                        for r in range(U.shape[1]):
                            C_rec[i, k] += U[u_idx(i, j), r] * V[v_idx(j, k), r] * W[w_idx(k, i), r]
            if not np.allclose(C_exp, C_rec, atol=1e-10):
                return False
        return True

    if not verify(U, V, W):
        # Fallback to schoolbook
        rank_sb = n * m * p
        U = np.zeros((nm, rank_sb))
        V = np.zeros((mp, rank_sb))
        W = np.zeros((pn, rank_sb))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[u_idx(i, j), col] = 1
                    V[v_idx(j, k), col] = 1
                    W[w_idx(k, i), col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Attempt rank reduction: try to find linear dependencies among the rank-1 terms
    # that preserve the tensor structure
    # Strategy: for each pair of terms, check if one can be expressed as a linear combo
    # of others while maintaining the decomposition
    
    # A more effective approach: use the fact that Strassen's 7 terms for 2x2x2 
    # are already minimal, but across different blocks there may be sharing.
    
    # Try: remove one term at a time and check if the remaining still decompose correctly
    # This is expensive but we have time budget
    
    best_rank = rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try removing each term and checking validity
    for remove_idx in range(rank):
        if time_left() < 30:
            break
        U_test = np.delete(U, remove_idx, axis=1)
        V_test = np.delete(V, remove_idx, axis=1)
        W_test = np.delete(W, remove_idx, axis=1)
        if verify(U_test, V_test, W_test):
            U, V, W = U_test, V_test, W_test
            rank = rank - 1
            best_rank = rank
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            # Restart the removal process from the beginning
            remove_idx = 0
            continue
        remove_idx += 1

    # Try removing pairs of terms
    for i in range(rank):
        if time_left() < 20:
            break
        for j in range(i + 1, rank):
            if time_left() < 15:
                break
            U_test = np.delete(U, [i, j], axis=1)
            V_test = np.delete(V, [i, j], axis=1)
            W_test = np.delete(W, [i, j], axis=1)
            if verify(U_test, V_test, W_test):
                U, V, W = U_test, V_test, W_test
                rank = rank - 2
                best_rank = rank
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                i = 0
                break

    if time_left() > 5:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    else:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
