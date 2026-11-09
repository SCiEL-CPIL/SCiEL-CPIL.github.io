import time
import numpy as np

DIMENSIONS = [3, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115
    
    # Build rank-44 solution
    U44, V44, W44 = build_rank44(n, m, p)
    
    # Precompute triples for fast checking
    triples = np.array([(i * m + j, j * p + k, k * n + i) 
                        for i in range(n) for j in range(m) for k in range(p)])
    
    # Identify representative columns to try for reduction
    # The rank-44 solution consists of 4 Strassen blocks (7 cols each) and 4 Trivial blocks (4 cols each)
    # Strassen blocks start at 0, 7, 14, 21
    # Trivial blocks start at 28, 29, 30, 31? No, add_122 adds 4 cols.
    # 4 * 7 = 28. So Strassen cols are 0-27.
    # 4 * 4 = 16. So Trivial cols are 28-43.
    
    # Due to symmetry, we try one representative from each "structural" type.
    # Strassen columns 0-6 are structurally distinct.
    # Trivial columns 0-3 are structurally distinct.
    # Blocks are likely equivalent, so we only need to try columns from the first block.
    # However, to be safe, we can try a few from different blocks if time permits.
    
    candidate_cols = list(range(7))  # Strassen types 0-6
    candidate_cols.extend(range(28, 32)) # Trivial types 0-3 (cols 28, 29, 30, 31)
    
    # Shuffle to randomize? Or keep ordered. Ordered is fine.
    
    for col in candidate_cols:
        if time.time() > deadline - 10:
            break
        result = try_reduce_to_rank(U44, V44, W44, col, n, m, p, deadline, triples)
        if result is not None:
            U, V, W = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # If rank-43 not found, try a few more random columns if time permits
    remaining_cols = [c for c in range(44) if c not in candidate_cols]
    np.random.shuffle(remaining_cols)
    for col in remaining_cols[:5]:
        if time.time() > deadline - 5:
            break
        result = try_reduce_to_rank(U44, V44, W44, col, n, m, p, deadline, triples)
        if result is not None:
            U, V, W = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
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
        U[(i_off+0)*m + (j_off+0), col] = 1
        U[(i_off+1)*m + (j_off+1), col] = 1
        V[(j_off+0)*p + (k_off+0), col] = 1
        V[(j_off+1)*p + (k_off+1), col] = 1
        W[(k_off+0)*n + (i_off+0), col] = 1
        W[(k_off+1)*n + (i_off+1), col] = 1
        col += 1

        U[(i_off+1)*m + (j_off+0), col] = 1
        U[(i_off+1)*m + (j_off+1), col] = 1
        V[(j_off+0)*p + (k_off+0), col] = 1
        W[(k_off+0)*n + (i_off+1), col] = 1
        W[(k_off+1)*n + (i_off+1), col] = -1
        col += 1

        U[(i_off+0)*m + (j_off+0), col] = 1
        V[(j_off+0)*p + (k_off+1), col] = 1
        V[(j_off+1)*p + (k_off+1), col] = -1
        W[(k_off+1)*n + (i_off+0), col] = 1
        W[(k_off+1)*n + (i_off+1), col] = 1
        col += 1

        U[(i_off+1)*m + (j_off+1), col] = 1
        V[(j_off+1)*p + (k_off+0), col] = 1
        V[(j_off+0)*p + (k_off+0), col] = -1
        W[(k_off+0)*n + (i_off+0), col] = 1
        W[(k_off+0)*n + (i_off+1), col] = 1
        col += 1

        U[(i_off+0)*m + (j_off+0), col] = 1
        U[(i_off+0)*m + (j_off+1), col] = 1
        V[(j_off+1)*p + (k_off+1), col] = 1
        W[(k_off+0)*n + (i_off+0), col] = -1
        W[(k_off+1)*n + (i_off+0), col] = 1
        col += 1

        U[(i_off+1)*m + (j_off+0), col] = 1
        U[(i_off+0)*m + (j_off+0), col] = -1
        V[(j_off+0)*p + (k_off+0), col] = 1
        V[(j_off+0)*p + (k_off+1), col] = 1
        W[(k_off+1)*n + (i_off+1), col] = 1
        col += 1

        U[(i_off+0)*m + (j_off+1), col] = 1
        U[(i_off+1)*m + (j_off+1), col] = -1
        V[(j_off+1)*p + (k_off+0), col] = 1
        V[(j_off+1)*p + (k_off+1), col] = 1
        W[(k_off+0)*n + (i_off+0), col] = 1
        col += 1

    def add_122(i_off, j_off, k_off):
        nonlocal col
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

def check(U, V, W, n, m, p, triples):
    for idx in range(len(triples)):
        a, b, c = triples[idx]
        val = np.dot(U[a], V[b] * W[c])
        if abs(val - 1) > 1e-6:
            return False
    return True

def try_reduce_to_rank(U, V, W, remove_col, n, m, p, deadline, triples):
    R = U.shape[1] - 1  # 43
    U43 = np.delete(U, remove_col, axis=1).copy()
    V43 = np.delete(V, remove_col, axis=1).copy()
    W43 = np.delete(W, remove_col, axis=1).copy()

    # ALS iterations
    for it in range(40):
        if time.time() > deadline - 5:
            break

        # Update U
        for i in range(n):
            for j in range(m):
                M = np.zeros((p, R))
                for k in range(p):
                    M[k, :] = V43[j * p + k, :] * W43[k * n + i, :]
                try:
                    sol, *_ = np.linalg.lstsq(M.T, np.ones(p), rcond=None)
                    U43[i * m + j, :] = sol
                except:
                    pass

        # Update V
        for j in range(m):
            for k in range(p):
                M = np.zeros((n, R))
                for i in range(n):
                    M[i, :] = U43[i * m + j, :] * W43[k * n + i, :]
                try:
                    sol, *_ = np.linalg.lstsq(M.T, np.ones(n), rcond=None)
                    V43[j * p + k, :] = sol
                except:
                    pass

        # Update W
        for k in range(p):
            for i in range(n):
                M = np.zeros((m, R))
                for j in range(m):
                    M[j, :] = U43[i * m + j, :] * V43[j * p + k, :]
                try:
                    sol, *_ = np.linalg.lstsq(M.T, np.ones(m), rcond=None)
                    W43[k * n + i, :] = sol
                except:
                    pass

    # Round to half-integers
    U43 = np.round(U43 * 2) / 2
    V43 = np.round(V43 * 2) / 2
    W43 = np.round(W43 * 2) / 2

    if check(U43, V43, W43, n, m, p, triples):
        return U43, V43, W43

    # Local search
    for ls_iter in range(100):
        if time.time() > deadline - 1:
            break
        
        # Find violated entries
        violated = set()
        for idx in range(len(triples)):
            a, b, c = triples[idx]
            val = np.dot(U43[a], V43[b] * W43[c])
            if abs(val - 1) > 1e-6:
                violated.add(a)
                violated.add(b)
                violated.add(c)
        
        if not violated:
            return U43, V43, W43
        
        # Try flipping entries in violated positions
        improved = False
        for a in violated:
            if a < n * m:
                for r in range(R):
                    old = U43[a, r]
                    for delta in [0.5, -0.5]:
                        U43[a, r] = old + delta
                        if check(U43, V43, W43, n, m, p, triples):
                            return U43, V43, W43
                        U43[a, r] = old
            elif a < n * m + m * p:
                v_idx = a - n * m
                for r in range(R):
                    old = V43[v_idx, r]
                    for delta in [0.5, -0.5]:
                        V43[v_idx, r] = old + delta
                        if check(U43, V43, W43, n, m, p, triples):
                            return U43, V43, W43
                        V43[v_idx, r] = old
            else:
                w_idx = a - n * m - m * p
                for r in range(R):
                    old = W43[w_idx, r]
                    for delta in [0.5, -0.5]:
                        W43[w_idx, r] = old + delta
                        if check(U43, V43, W43, n, m, p, triples):
                            return U43, V43, W43
                        W43[w_idx, r] = old
    
    return None
