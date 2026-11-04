import time
import sys
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 8]

def construct():
    t0 = time.time()
    deadline = t0 + 140.0

    def _elapsed_ok():
        return (time.time() - t0) < deadline

    n, m, p = 2, 3, 8
    nm, mp, pn = n * m, m * p, p * n

    # Schoolbook fallback (Rank 48)
    def _schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    U[i * m + a][col] = 1
                    V[a * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def _validate(candidate):
        U = candidate["U"]
        V = candidate["V"]
        W = candidate["W"]
        rank = len(U[0])
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    acc = Fraction(0)
                    u_row = U[i * m + a]
                    v_row = V[a * p + k]
                    w_row = W[k * n + i]
                    for c in range(rank):
                        if u_row[c] == 0 or v_row[c] == 0 or w_row[c] == 0:
                            continue
                        acc += Fraction(u_row[c]) * Fraction(v_row[c]) * Fraction(w_row[c])
                    if acc != 1:
                        return False
        return True

    # Rank-44 fallback
    def _rank44():
        S_U = [
            [1, 0, 0, 1],
            [0, 0, 1, 1],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [1, 1, 0, 0],
            [-1, 0, 1, 0],
            [0, 1, 0, -1]
        ]
        S_V = [
            [1, 0, 0, 1],
            [1, 0, 0, 0],
            [0, 1, 0, -1],
            [-1, 0, 1, 0],
            [0, 0, 0, 1],
            [1, 1, 0, 0],
            [0, 0, 1, 1]
        ]
        S_W = [
            [1, 0, 0, 1],
            [0, 1, 0, -1],
            [0, 0, 1, 1],
            [1, 1, 0, 0],
            [-1, 0, 1, 0],
            [0, 0, 0, 1],
            [1, 0, 0, 0]
        ]
        rank = 44
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for block_idx in range(4):
            k0 = block_idx * 2
            for r in range(7):
                for li in range(2):
                    for la in range(2):
                        u_val = S_U[r][li * 2 + la]
                        if u_val != 0:
                            U[li * m + la][col] = u_val
                for la in range(2):
                    for lk in range(2):
                        v_val = S_V[r][la * 2 + lk]
                        if v_val != 0:
                            V[la * p + (k0 + lk)][col] = v_val
                for lk in range(2):
                    for li in range(2):
                        w_val = S_W[r][lk * 2 + li]
                        if w_val != 0:
                            W[(k0 + lk) * n + li][col] = w_val
                col += 1
        for i in range(2):
            for k in range(8):
                U[i * m + 2][col] = 1
                V[2 * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
        return {"U": U, "V": V, "W": W}

    best_solution = _rank44()
    best_rank = 44

    if not _validate(best_solution):
        best_solution = _schoolbook()
        best_rank = 48

    if not _elapsed_ok():
        return best_solution

    # Try to find rank 36
    # Search for rank-9 decomposition of 2x3x2
    def _search_rank9_232():
        # U: 6x9, V: 6x9, W: 4x9
        # Flatten to vector
        dim_u = 6 * 9
        dim_v = 6 * 9
        dim_w = 4 * 9
        total_dim = dim_u + dim_v + dim_w
        
        def _loss(x):
            U = np.array(x[:dim_u]).reshape(6, 9)
            V = np.array(x[dim_u:dim_u+dim_v]).reshape(6, 9)
            W = np.array(x[dim_u+dim_v:]).reshape(4, 9)
            
            errors = []
            for i in range(2):
                for a in range(3):
                    for k in range(2):
                        # Sum over r
                        s = 0.0
                        for r in range(9):
                            s += U[i*3+a, r] * V[a*2+k, r] * W[k*2+i, r]
                        errors.append(s - 1.0)
            return np.array(errors)
            
        # Random restarts
        np.random.seed(42)
        for restart in range(50):
            x0 = np.random.randn(total_dim) * 0.1
            try:
                from scipy.optimize import least_squares
                res = least_squares(_loss, x0, method='lm', max_nfev=1000)
                if res.cost < 1e-6:
                    # Found a solution, try to round
                    U = np.array(res.x[:dim_u]).reshape(6, 9)
                    V = np.array(res.x[dim_u:dim_u+dim_v]).reshape(6, 9)
                    W = np.array(res.x[dim_u+dim_v:]).reshape(4, 9)
                    
                    # Round to nearest 0.5
                    U_r = np.round(U * 2) / 2
                    V_r = np.round(V * 2) / 2
                    W_r = np.round(W * 2) / 2
                    
                    # Verify
                    valid = True
                    for i in range(2):
                        for a in range(3):
                            for k in range(2):
                                s = 0.0
                                for r in range(9):
                                    s += U_r[i*3+a, r] * V_r[a*2+k, r] * W_r[k*2+i, r]
                                if abs(s - 1.0) > 1e-9:
                                    valid = False
                                    break
                            if not valid: break
                        if not valid: break
                    if valid:
                        # Return the rounded solution as lists of Fractions
                        U_list = [[Fraction(int(U_r[i, r] * 2)) / 2 for r in range(9)] for i in range(6)]
                        V_list = [[Fraction(int(V_r[i, r] * 2)) / 2 for r in range(9)] for i in range(6)]
                        W_list = [[Fraction(int(W_r[i, r] * 2)) / 2 for r in range(9)] for i in range(4)]
                        return U_list, V_list, W_list
            except Exception:
                pass
        return None

    if _elapsed_ok():
        res_232 = _search_rank9_232()
        if res_232 is not None:
            U_block, V_block, W_block = res_232
            rank = 36
            U = [[0] * rank for _ in range(nm)]
            V = [[0] * rank for _ in range(mp)]
            W = [[0] * rank for _ in range(pn)]
            
            # Assemble 4 blocks
            for block_idx in range(4):
                k0 = block_idx * 2
                col_offset = block_idx * 9
                for r in range(9):
                    col = col_offset + r
                    # U: indices (i, a) where a in 0,1,2
                    # U_block is 6x9. Row i*3+a
                    for i in range(2):
                        for a in range(3):
                            val = U_block[i*3+a][r]
                            if val != 0:
                                U[i*m+a][col] = val
                    # V: indices (a, k) where a in 0,1,2, k in 0,1
                    # V_block is 6x9. Row a*2+k
                    for a in range(3):
                        for lk in range(2):
                            val = V_block[a*2+lk][r]
                            if val != 0:
                                V[a*p + (k0+lk)][col] = val
                    # W: indices (k, i) where k in 0,1, i in 0,1
                    # W_block is 4x9. Row k*2+i
                    for lk in range(2):
                        for i in range(2):
                            val = W_block[lk*2+i][r]
                            if val != 0:
                                W[(k0+lk)*n + i][col] = val
            
            candidate = {"U": U, "V": V, "W": W}
            if _validate(candidate):
                best_solution = candidate
                best_rank = 36
                return best_solution

    return best_solution
