import numpy as np
import time

def construct():
    n, m, p = 2, 3, 5
    start_time = time.time()
    deadline = start_time + 115
    
    nm = n * m  # 6
    mp = m * p  # 15
    pn = p * n  # 10
    
    # Build target tensor T6[i,j,jp,ik]
    T6 = np.zeros((n, m, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for jp in range(mp):
                j2 = jp // p
                k2 = jp % p
                for ik in range(pn):
                    k_out = ik // n
                    i_out = ik % n
                    if i == i_out and j == j2 and k2 == k_out:
                        T6[i, j, jp, ik] = 1.0
    
    def check_solution(U, V, W, tol=1e-8):
        result = np.zeros((n, m, mp, pn), dtype=np.float64)
        r = U.shape[1]
        for r_idx in range(r):
            u_col = U[:, r_idx].reshape(n, m)
            v_col = V[:, r_idx]
            w_col = W[:, r_idx]
            for i in range(n):
                for j in range(m):
                    result[i, j, :, :] += u_col[i, j] * np.outer(v_col, w_col)
        return np.max(np.abs(result - T6))
    
    def solve_for_W(U, V):
        W = np.zeros((pn, U.shape[1]), dtype=np.float64)
        r = U.shape[1]
        for ik in range(pn):
            k_out = ik // n
            i_out = ik % n
            A = np.zeros((n * m * mp, r), dtype=np.float64)
            b = np.zeros(n * m * mp, dtype=np.float64)
            idx = 0
            for i in range(n):
                for j in range(m):
                    for jp in range(mp):
                        j2 = jp // p
                        k2 = jp % p
                        delta = 1.0 if (i == i_out and j == j2 and k2 == k_out) else 0.0
                        A[idx, :] = U[i*m+j, :] * V[jp, :]
                        b[idx] = delta
                        idx += 1
            W[ik, :], _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        return W
    
    def solve_for_U(V, W):
        U = np.zeros((nm, V.shape[1]), dtype=np.float64)
        r = V.shape[1]
        for ij in range(nm):
            i_out = ij // m
            j_out = ij % m
            A = np.zeros((m * mp * pn, r), dtype=np.float64)
            b = np.zeros(m * mp * pn, dtype=np.float64)
            idx = 0
            for j in range(m):
                for jp in range(mp):
                    j2 = jp // p
                    k2 = jp % p
                    for ik in range(pn):
                        k_out = ik // n
                        i2 = ik % n
                        delta = 1.0 if (i_out == i2 and j_out == j2 and k2 == k_out) else 0.0
                        A[idx, :] = V[jp, :] * W[ik, :]
                        b[idx] = delta
                        idx += 1
            U[ij, :], _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        return U
    
    def solve_for_V(U, W):
        V = np.zeros((mp, U.shape[1]), dtype=np.float64)
        r = U.shape[1]
        for jp in range(mp):
            j_out = jp // p
            k_out = jp % p
            A = np.zeros((nm * m * pn, r), dtype=np.float64)
            b = np.zeros(nm * m * pn, dtype=np.float64)
            idx = 0
            for i in range(n):
                for j in range(m):
                    for ik in range(pn):
                        k_out2 = ik // n
                        i2 = ik % n
                        delta = 1.0 if (i == i2 and j == j_out and k_out == k_out2) else 0.0
                        A[idx, :] = U[i*m+j, :] * W[ik, :]
                        b[idx] = delta
                        idx += 1
            V[jp, :], _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        return V
    
    best_result = None
    best_rank = n * m * p  # 30
    
    # Try ranks from 25 to 29
    for r in range(25, 30):
        if time.time() > deadline - 10:
            break
        for trial in range(15):
            if time.time() > deadline - 10:
                break
            np.random.seed(trial * 100 + r)
            U = np.random.randn(nm, r)
            V = np.random.randn(mp, r)
            W = np.random.randn(pn, r)
            
            for it in range(30):
                if time.time() > deadline - 10:
                    break
                W = solve_for_W(U, V)
                U = solve_for_U(V, W)
                V = solve_for_V(U, W)
            
            err = check_solution(U, V, W)
            if err < 1e-6:
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fall back to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
