import time
import numpy as np
import itertools

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115

    U_cols = []
    V_cols = []
    W_cols = []

    def add_term(U_entry, V_entry, W_entry):
        U_col = [0] * (n * m)
        for idx, val in U_entry.items():
            U_col[idx] = val
        V_col = [0] * (m * p)
        for idx, val in V_entry.items():
            V_col[idx] = val
        W_col = [0] * (p * n)
        for idx, val in W_entry.items():
            W_col[idx] = val
        U_cols.append(U_col)
        V_cols.append(V_col)
        W_cols.append(W_col)

    # Rank-50 construction (same as current program)
    for block_start in range(0, 8, 2):
        k0 = block_start
        k1 = block_start + 1
        
        add_term(
            {0: 1, 4: 1},
            {k0: 1, p+k1: 1},
            {k0*n: 1, k1*n+1: 1}
        )
        add_term(
            {3: 1, 4: 1},
            {k0: 1},
            {k0*n+1: 1, k1*n+1: -1}
        )
        add_term(
            {0: 1},
            {k1: 1, p+k1: -1},
            {k1*n: 1, k1*n+1: 1}
        )
        add_term(
            {4: 1},
            {p+k0: 1, k0: -1},
            {k0*n: 1, k0*n+1: 1}
        )
        add_term(
            {0: 1, 1: 1},
            {p+k1: 1},
            {k0*n: -1, k1*n: 1}
        )
        add_term(
            {3: 1, 0: -1},
            {k0: 1, k1: 1},
            {k1*n+1: 1}
        )
        add_term(
            {1: 1, 4: -1},
            {p+k0: 1, p+k1: 1},
            {k0*n: 1}
        )
    
    k = 8
    add_term({0: 1}, {k: 1}, {k*n: 1})
    add_term({1: 1}, {p+k: 1}, {k*n: 1})
    add_term({3: 1}, {k: 1}, {k*n+1: 1})
    add_term({4: 1}, {p+k: 1}, {k*n+1: 1})
    
    for i in range(n):
        for kk in range(p):
            add_term(
                {i*m+2: 1},
                {2*p+kk: 1},
                {kk*n+i: 1}
            )
            
    rank = len(U_cols)

    def verify(U_mat, V_mat, W_mat, r):
        """Verify that U @ V^T @ W^T decomposes the tensor."""
        for i in range(n):
            for kk in range(m):
                for j in range(p):
                    a = i * m + kk
                    b = kk * p + j
                    c = j * n + i
                    val = sum(U_mat[a, s] * V_mat[b, s] * W_mat[c, s] for s in range(r))
                    if abs(val - 1) > 1e-8:
                        return False
        return True

    def try_remove(indices_to_remove):
        """Try to remove the given indices by solving for scaling factors."""
        if time.time() > deadline - 5:
            return None
        if not indices_to_remove:
            return None
        
        U_mat = np.array(U_cols, dtype=float).T
        V_mat = np.array(V_cols, dtype=float).T
        W_mat = np.array(W_cols, dtype=float).T
        
        keep = [s for s in range(rank) if s not in indices_to_remove]
        r_new = len(keep)
        
        # Build system: for each tensor entry, sum of kept terms = sum of removed terms
        equations = []
        for i in range(n):
            for kk in range(m):
                for j in range(p):
                    a = i * m + kk
                    b = kk * p + j
                    c = j * n + i
                    lhs = sum(U_mat[a, r] * V_mat[b, r] * W_mat[c, r] for r in indices_to_remove)
                    if abs(lhs) > 1e-12:
                        row = np.array([U_mat[a, s] * V_mat[b, s] * W_mat[c, s] for s in keep])
                        equations.append((row, lhs))
        
        if not equations:
            return None
            
        A_sys = np.array([eq[0] for eq in equations])
        b_sys = np.array([eq[1] for eq in equations])
        
        try:
            d, residuals, rank_sys, sv = np.linalg.lstsq(A_sys, b_sys, rcond=None)
            if not np.allclose(A_sys @ d, b_sys, atol=1e-8):
                return None
            
            # Check if d is half-integer (d*2 should be integer)
            if not np.allclose(d * 2, np.round(d * 2)):
                return None
            
            # Scale U by d
            U_new = np.array([U_cols[s] for s in keep], dtype=float).T
            V_new = np.array([V_cols[s] for s in keep], dtype=float).T
            W_new = np.array([W_cols[s] for s in keep], dtype=float).T
            
            for s in range(r_new):
                U_new[:, s] *= d[s]
            
            if not verify(U_new, V_new, W_new, r_new):
                return None
            
            U_final = [[U_new[i, s] for s in range(r_new)] for i in range(n*m)]
            V_final = [[V_new[i, s] for s in range(r_new)] for i in range(m*p)]
            W_final = [[W_new[i, s] for s in range(r_new)] for i in range(p*n)]
            return (U_final, V_final, W_final)
        except:
            pass
        return None

    # Try removing 1 term
    for r_remove in range(rank):
        if time.time() > deadline - 15:
            break
        res = try_remove([r_remove])
        if res:
            U_cols, V_cols, W_cols = res
            rank = len(U_cols)
            break

    # Try removing 2 terms
    if rank > 45 and time.time() < deadline - 15:
        for r1, r2 in itertools.combinations(range(rank), 2):
            if time.time() > deadline - 10:
                break
            res = try_remove([r1, r2])
            if res:
                U_cols, V_cols, W_cols = res
                rank = len(U_cols)
                break

    # Try removing 3 terms (if still above 45)
    if rank > 45 and time.time() < deadline - 10:
        for combo in itertools.combinations(range(rank), 3):
            if time.time() > deadline - 5:
                break
            res = try_remove(list(combo))
            if res:
                U_cols, V_cols, W_cols = res
                rank = len(U_cols)
                break

    rank = len(U_cols)
    U = [[U_cols[r][i] for r in range(rank)] for i in range(n*m)]
    V = [[V_cols[r][j] for r in range(rank)] for j in range(m*p)]
    W = [[W_cols[r][k] for r in range(rank)] for k in range(p*n)]
    
    return {"U": U, "V": V, "W": W}
