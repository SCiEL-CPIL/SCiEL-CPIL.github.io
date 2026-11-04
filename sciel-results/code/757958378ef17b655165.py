import time
import numpy as np
from itertools import product

def _strassen_28():
    """Return (U, V, W) as lists of lists, rank 28, 4 independent Strassen blocks."""
    n, m, p = 2, 4, 4
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    def add_block(col_offset, X_rows, Y_rows, Z_rows):
        U[X_rows[0]][col_offset + 0] += 1
        U[X_rows[3]][col_offset + 0] += 1
        V[Y_rows[0]][col_offset + 0] += 1
        V[Y_rows[3]][col_offset + 0] += 1
        W[Z_rows[0]][col_offset + 0] += 1
        W[Z_rows[3]][col_offset + 0] += 1

        U[X_rows[2]][col_offset + 1] += 1
        U[X_rows[3]][col_offset + 1] += 1
        V[Y_rows[0]][col_offset + 1] += 1
        W[Z_rows[2]][col_offset + 1] += 1
        W[Z_rows[3]][col_offset + 1] -= 1

        U[X_rows[0]][col_offset + 2] += 1
        V[Y_rows[1]][col_offset + 2] += 1
        V[Y_rows[3]][col_offset + 2] -= 1
        W[Z_rows[1]][col_offset + 2] += 1
        W[Z_rows[3]][col_offset + 2] += 1

        U[X_rows[3]][col_offset + 3] += 1
        V[Y_rows[2]][col_offset + 3] += 1
        V[Y_rows[0]][col_offset + 3] -= 1
        W[Z_rows[0]][col_offset + 3] += 1
        W[Z_rows[2]][col_offset + 3] += 1

        U[X_rows[0]][col_offset + 4] += 1
        U[X_rows[1]][col_offset + 4] += 1
        V[Y_rows[3]][col_offset + 4] += 1
        W[Z_rows[0]][col_offset + 4] -= 1
        W[Z_rows[1]][col_offset + 4] += 1

        U[X_rows[2]][col_offset + 5] += 1
        U[X_rows[0]][col_offset + 5] -= 1
        V[Y_rows[0]][col_offset + 5] += 1
        V[Y_rows[1]][col_offset + 5] += 1
        W[Z_rows[3]][col_offset + 5] += 1

        U[X_rows[1]][col_offset + 6] += 1
        U[X_rows[3]][col_offset + 6] -= 1
        V[Y_rows[2]][col_offset + 6] += 1
        V[Y_rows[3]][col_offset + 6] += 1
        W[Z_rows[0]][col_offset + 6] += 1

    X1 = [0, 1, 4, 5];  Y1 = [0, 1, 4, 5];   Z1 = [0, 2, 1, 3]
    X2 = [2, 3, 6, 7];  Y2 = [8, 9, 12, 13]; Z2 = [0, 2, 1, 3]
    X3 = [0, 1, 4, 5];  Y3 = [2, 3, 6, 7];   Z3 = [4, 6, 5, 7]
    X4 = [2, 3, 6, 7];  Y4 = [10, 11, 14, 15]; Z4 = [4, 6, 5, 7]

    add_block(0,  X1, Y1, Z1)
    add_block(7,  X2, Y2, Z2)
    add_block(14, X3, Y3, Z3)
    add_block(21, X4, Y4, Z4)
    return U, V, W

def _verify_decomp(U, V, W, n=2, m=4, p=4):
    rank = len(U[0])
    def target(i, j, k):
        if i % m != j // m: return 0
        if j % m != k // n: return 0
        if k % n != i // m: return 0
        return 1
    
    for i in range(n * m):
        for j in range(m * p):
            for k in range(p * n):
                s = 0
                for r in range(rank):
                    s += U[i][r] * V[j][r] * W[k][r]
                if s != target(i, j, k):
                    return False
    return True

def _try_rank_search(deadline):
    n, m, p = 2, 4, 4
    nm, mp, pn = n * m, m * p, p * n
    
    def target_tensor():
        T = np.zeros((nm, mp, pn), dtype=int)
        for i in range(nm):
            for j in range(mp):
                if i % m != j // m: continue
                for k in range(pn):
                    if j % m != k // n: continue
                    if k % n != i // m: continue
                    T[i, j, k] = 1
        return T
    
    T = target_tensor()
    best_rank = 28
    best_decomp = None
    
    # Short search for rank < 28
    for r in range(24, 28):
        if time.time() > deadline - 5:
            break
        for trial in range(200):
            if time.time() > deadline - 5:
                break
            U_cand = np.random.randint(-2, 3, (nm, r))
            V_cand = np.random.randint(-2, 3, (mp, r))
            W_cand = np.random.randint(-2, 3, (pn, r))
            
            ok = True
            for i in range(nm):
                for j in range(mp):
                    P = U_cand[i] * V_cand[j]
                    for k in range(pn):
                        s = np.dot(P, W_cand[k])
                        expected = 0
                        if i % m == j // m and j % m == k // n and k % n == i // m:
                            expected = 1
                        if s != expected:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break
            
            if ok:
                best_rank = r
                best_decomp = (U_cand.tolist(), V_cand.tolist(), W_cand.tolist())
                return best_rank, best_decomp
    
    return best_rank, best_decomp

def construct():
    t0 = time.time()
    deadline = t0 + 110.0
    
    try:
        rank_found, decomp_found = _try_rank_search(deadline)
        if decomp_found is not None:
            U, V, W = decomp_found
            if _verify_decomp(U, V, W):
                return {"U": U, "V": V, "W": W}
    except Exception:
        pass
    
    U, V, W = _strassen_28()
    return {"U": U, "V": V, "W": W}
