import time
import numpy as np

DIMENSIONS = [3, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115
    
    # Build rank-44 solution
    U44, V44, W44 = build_rank44(n, m, p)
    
    # Try to find rank-43 by removing one column and re-solving
    # We try all 44 columns, but with a time limit
    # We also try to find rank-42 by removing two columns if time permits
    
    # First, try rank 43 - try all 44 columns
    for col in range(44):
        if time.time() > deadline - 15:
            break
        result = try_reduce_to_rank(U44, V44, W44, [col], n, m, p, deadline)
        if result is not None:
            U, V, W = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
    # If rank 43 failed, try rank 42 (remove 2 columns)
    # We only try a few pairs to save time
    import itertools
    pairs = list(itertools.combinations(range(44), 2))
    # Shuffle to avoid bias
    np.random.seed(42)
    np.random.shuffle(pairs)
    
    for pair in pairs[:30]: # Try first 30 pairs
        if time.time() > deadline - 5:
            break
        result = try_reduce_to_rank(U44, V44, W44, list(pair), n, m, p, deadline)
        if result is not None:
            U, V, W = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W44.tolist()}
            
    # Return rank-44
    return {"U": U44.tolist(), "V": V44.tolist(), "W": W44.tolist()}

def build_rank44(n, m, p):
    rank = 44
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0

    def add_strassen_222(i_off, j_off, k_off):
        nonlocal col
        # 7 terms for 2x2x2 Strassen
        # M1 = (A11+A22)(B11+B22)
        U[(i_off+0)*m + (j_off+0), col] = 1
        U[(i_off+1)*m + (j_off+1), col] = 1
        V[(j_off+0)*p + (k_off+0), col] = 1
        V[(j_off+1)*p + (k_off+1), col] = 1
        W[(k_off+0)*n + (i_off+0), col] = 1
        W[(k_off+1)*n + (i_off+1), col] = 1
        col += 1

        # M2 = (A21+A22)B11
        U[(i_off+1)*m + (j_off+0), col] = 1
        U[(i_off+1)*m + (j_off+1), col] = 1
        V[(j_off+0)*p + (k_off+0), col] = 1
        W[(k_off+0)*n + (i_off+1), col] = 1
        W[(k_off+1)*n + (i_off+1), col] = -1
        col += 1

        # M3 = A11(B21-B22)
        U[(i_off+0)*m + (j_off+0), col] = 1
        V[(j_off+0)*p + (k_off+1), col] = 1
        V[(j_off+1)*p + (k_off+1), col] = -1
        W[(k_off+1)*n + (i_off+0), col] = 1
        W[(k_off+1)*n + (i_off+1), col] = 1
        col += 1

        # M4 = A22(B11-B12)
        U[(i_off+1)*m + (j_off+1), col] = 1
        V[(j_off+1)*p + (k_off+0), col] = 1
        V[(j_off+0)*p + (k_off+0), col] = -1
        W[(k_off+0)*n + (i_off+0), col] = 1
        W[(k_off+0)*n + (i_off+1), col] = 1
        col += 1

        # M5 = (A11+A12)B22
        U[(i_off+0)*m + (j_off+0), col] = 1
        U[(i_off+0)*m + (j_off+1), col] = 1
        V[(j_off+1)*p + (k_off+1), col] = 1
        W[(k_off+0)*n + (i_off+0), col] = -1
        W[(k_off+1)*n + (i_off+0), col] = 1
        col += 1

        # M6 = (A21-A11)(B11+B12)
        U[(i_off+1)*m + (j_off+0), col] = 1
        U[(i_off+0)*m + (j_off+0), col] = -1
        V[(j_off+0)*p + (k_off+0), col] = 1
        V[(j_off+0)*p + (k_off+1), col] = 1
        W[(k_off+1)*n + (i_off+1), col] = 1
        col += 1

        # M7 = (A12-A22)(B21+B22)
        U[(i_off+0)*m + (j_off+1), col] = 1
        U[(i_off+1)*m + (j_off+1), col] = -1
        V[(j_off+1)*p + (k_off+0), col] = 1
        V[(j_off+1)*p + (k_off+1), col] = 1
        W[(k_off+0)*n + (i_off+0), col] = 1
        col += 1

    def add_122(i_off, j_off, k_off):
        nonlocal col
        # 4 terms for 1x2x2 schoolbook
        U[(i_off+0)*m + (j_off+0), col] = 1
        V[(j_off+0)*p + (k_off+0), col] = 1
        W[(k_off+0)*n + (i_off+0), col] = 1
        col += 1

        U[(i_off+0)*m + (j_off+1), col] = 1
        V[(j_off+1)*p + (k_off+0), col] = 1
        W[(k_off+0)*n + (i_off+0), col] = 1
        col += 1

        U[(i_off+0)*m + (j_off+0), col] = 1
        V[(j_off+0)*p + (k_off+1), col] = 1
        W[(k_off+1)*n + (i_off+0), col] = 1
        col += 1

        U[(i_off+0)*m + (j_off+1), col] = 1
        V[(j_off+1)*p + (k_off+1), col] = 1
        W[(k_off+1)*n + (i_off+0), col] = 1
        col += 1

    add_strassen_222(0, 0, 0)
    add_strassen_222(0, 2, 0)
    add_strassen_222(0, 0, 2)
    add_strassen_222(0, 2, 2)
    add_122(2, 0, 0)
    add_122(2, 2, 0)
    add_122(2, 0, 2)
    add_122(2, 2, 2)

    return U, V, W

def check(U, V, W, n, m, p):
    for i in range(n):
        for j in range(m):
            a = i * m + j
            for k in range(p):
                b = j * p + k
                c = k * n + i
                val = np.dot(U[a], V[b] * W[c])
                if abs(val - 1) > 1e-6:
                    return False
    return True

def vectorized_check(U, V, W, n, m, p, triples):
    """Vectorized check: returns array of values for all triples."""
    a_idx = triples[:, 0]
    b_idx = triples[:, 1]
    c_idx = triples[:, 2]
    # U[a, r] * V[b, r] * W[c, r] summed over r
    UVW = U[a_idx] * V[b_idx] * W[c_idx]
    return UVW.sum(axis=1)

def try_reduce_to_rank(U, V, W, remove_cols, n, m, p, deadline):
    R = U.shape[1] - len(remove_cols)
    if R <= 0:
        return None
        
    U_new = np.delete(U, remove_cols, axis=1).copy()
    V_new = np.delete(V, remove_cols, axis=1).copy()
    W_new = np.delete(W, remove_cols, axis=1).copy()

    # Precompute index arrays for speed
    triples = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                triples.append((i * m + j, j * p + k, k * n + i))
    triples = np.array(triples)
    num_triples = len(triples)

    def fast_check():
        vals = vectorized_check(U_new, V_new, W_new, n, m, p, triples)
        return np.all(np.abs(vals - 1) < 1e-6)

    def get_residuals():
        return vectorized_check(U_new, V_new, W_new, n, m, p, triples) - 1

    # ALS iterations
    for it in range(80):
        if time.time() > deadline - 3:
            break

        # Update U
        for i in range(n):
            for j in range(m):
                M = np.zeros((p, R))
                for k in range(p):
                    M[k, :] = V_new[j * p + k, :] * W_new[k * n + i, :]
                try:
                    sol, *_ = np.linalg.lstsq(M.T, np.ones(p), rcond=None)
                    U_new[i * m + j, :] = sol
                except:
                    pass

        # Update V
        for j in range(m):
            for k in range(p):
                M = np.zeros((n, R))
                for i in range(n):
                    M[i, :] = U_new[i * m + j, :] * W_new[k * n + i, :]
                try:
                    sol, *_ = np.linalg.lstsq(M.T, np.ones(n), rcond=None)
                    V_new[j * p + k, :] = sol
                except:
                    pass

        # Update W
        for k in range(p):
            for i in range(n):
                M = np.zeros((m, R))
                for j in range(m):
                    M[j, :] = U_new[i * m + j, :] * V_new[j * p + k, :]
                try:
                    sol, *_ = np.linalg.lstsq(M.T, np.ones(m), rcond=None)
                    W_new[k * n + i, :] = sol
                except:
                    pass

    # Round to half-integers
    U_new = np.round(U_new * 2) / 2
    V_new = np.round(V_new * 2) / 2
    W_new = np.round(W_new * 2) / 2

    if fast_check():
        return U_new, V_new, W_new

    # Local search using vectorized residual computation
    best_residual = np.inf
    for ls_iter in range(500):
        if time.time() > deadline - 1:
            break
        
        residuals = get_residuals()
        max_res = np.max(np.abs(residuals))
        if max_res < 1e-6:
            return U_new, V_new, W_new
        
        if max_res < best_residual:
            best_residual = max_res
        
        # Find the most violated triple
        worst_idx = np.argmax(np.abs(residuals))
        a, b, c = triples[worst_idx]
        
        # Try adjusting U[a, r]
        for r in range(R):
            vu = V_new[b, r] * W_new[c, r]
            if abs(vu) < 1e-9:
                continue
            target = U_new[a, r] - residuals[worst_idx] / vu
            for delta in [0.5, -0.5]:
                U_new[a, r] = U_new[a, r] + delta
                if fast_check():
                    return U_new, V_new, W_new
                U_new[a, r] -= delta
        
        # Try adjusting V[b, r]
        for r in range(R):
            vv = U_new[a, r] * W_new[c, r]
            if abs(vv) < 1e-9:
                continue
            for delta in [0.5, -0.5]:
                V_new[b, r] = V_new[b, r] + delta
                if fast_check():
                    return U_new, V_new, W_new
                V_new[b, r] -= delta
        
        # Try adjusting W[c, r]
        for r in range(R):
            vw = U_new[a, r] * V_new[b, r]
            if abs(vw) < 1e-9:
                continue
            for delta in [0.5, -0.5]:
                W_new[c, r] = W_new[c, r] + delta
                if fast_check():
                    return U_new, V_new, W_new
                W_new[c, r] -= delta
        
        # Also try other violated triples (top 5)
        top_violated = np.argsort(np.abs(residuals))[-5:][::-1]
        for idx in top_violated:
            if idx == worst_idx:
                continue
            a2, b2, c2 = triples[idx]
            for r in range(R):
                vu = V_new[b2, r] * W_new[c2, r]
                if abs(vu) > 1e-9:
                    for delta in [0.5, -0.5]:
                        U_new[a2, r] = U_new[a2, r] + delta
                        if fast_check():
                            return U_new, V_new, W_new
                        U_new[a2, r] -= delta
                vv = U_new[a2, r] * W_new[c2, r]
                if abs(vv) > 1e-9:
                    for delta in [0.5, -0.5]:
                        V_new[b2, r] = V_new[b2, r] + delta
                        if fast_check():
                            return U_new, V_new, W_new
                        V_new[b2, r] -= delta
                vw = U_new[a2, r] * V_new[b2, r]
                if abs(vw) > 1e-9:
                    for delta in [0.5, -0.5]:
                        W_new[c2, r] = W_new[c2, r] + delta
                        if fast_check():
                            return U_new, V_new, W_new
                        W_new[c2, r] -= delta
    
    return None
