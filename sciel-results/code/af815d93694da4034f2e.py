"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5
    start_time = time.time()
    deadline = start_time + 135.0  # safety margin before 150s kill
    
    schoolbook_rank = n * m * p  # 80
    
    def build_schoolbook():
        rank = schoolbook_rank
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    def verify(U, V, W, tol=1e-9):
        n, m, p = DIMENSIONS
        rank = len(U[0])
        if len(U) != n * m or len(V) != m * p or len(W) != p * n:
            return False
        if any(len(row) != rank for row in U):
            return False
        if any(len(row) != rank for row in V):
            return False
        if any(len(row) != rank for row in W):
            return False
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    def build_strassen_2x2x2():
        rank = 7
        U = [[0] * rank for _ in range(4)]
        V = [[0] * rank for _ in range(4)]
        W = [[0] * rank for _ in range(4)]
        U[0][0]=1; U[3][0]=1; V[0][0]=1; V[3][0]=1; W[0][0]=1; W[2][0]=1; W[1][0]=1; W[3][0]=1
        U[2][1]=1; U[3][1]=1; V[0][1]=1; W[1][1]=1; W[3][1]=1
        U[0][2]=1; V[1][2]=1; V[3][2]=-1; W[2][2]=1
        U[3][3]=1; V[2][3]=1; V[0][3]=-1; W[1][3]=1; W[3][3]=1
        U[0][4]=1; U[1][4]=1; V[3][4]=1; W[2][4]=1; W[3][4]=1
        U[2][5]=1; U[0][5]=-1; V[0][5]=1; V[1][5]=1; W[3][5]=1
        U[1][6]=1; U[3][6]=-1; V[2][6]=1; V[3][6]=1; W[0][6]=1; W[2][6]=1
        return {"U": U, "V": V, "W": W}
    
    def build_4x4x4_strassen():
        s2 = build_strassen_2x2x2()
        U2 = np.array(s2["U"], dtype=float)
        V2 = np.array(s2["V"], dtype=float)
        W2 = np.array(s2["W"], dtype=float)
        rank = 49
        U = np.zeros((16, rank), dtype=float)
        V = np.zeros((16, rank), dtype=float)
        W = np.zeros((16, rank), dtype=float)
        for c_block in range(7):
            for c_inner in range(7):
                c = c_block * 7 + c_inner
                for i in range(4):
                    for j in range(4):
                        bi, bj = i // 2, j // 2
                        ri, rj = i % 2, j % 2
                        U[i*4+j, c] = U2[bi*2+bj, c_block] * U2[ri*2+rj, c_inner]
                for j in range(4):
                    for k in range(4):
                        bj, bk = j // 2, k // 2
                        rj, rk = j % 2, k % 2
                        V[j*4+k, c] = V2[bj*2+bk, c_block] * V2[rj*2+rk, c_inner]
                for k in range(4):
                    for i in range(4):
                        bk, bi = k // 2, i // 2
                        rk, ri = k % 2, i % 2
                        W[k*4+i, c] = W2[bk*2+bi, c_block] * W2[rk*2+ri, c_inner]
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    def build_4x4x5_strassen():
        s4 = build_4x4x4_strassen()
        U4 = np.array(s4["U"], dtype=float)
        V4 = np.array(s4["V"], dtype=float)
        W4 = np.array(s4["W"], dtype=float)
        rank = 49 + 16
        U = np.zeros((16, rank), dtype=float)
        V = np.zeros((20, rank), dtype=float)
        W = np.zeros((20, rank), dtype=float)
        U[:, :49] = U4
        V[:16, :49] = V4
        W[:16, :49] = W4
        for i in range(4):
            for j in range(4):
                c = 49 + i*4 + j
                U[i*4+j, c] = 1
                V[j*5+4, c] = 1
                W[4*4+i, c] = 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    def _als_decomposition(rank, n, m, p, deadline, n_restarts=10):
        """Alternating Least Squares for tensor decomposition."""
        best_err = float('inf')
        best_factors = None
        
        for restart in range(n_restarts):
            if time.time() > deadline - 5:
                break
            np.random.seed(restart * 1000 + 42)
            U = np.random.randn(n * m, rank)
            V = np.random.randn(m * p, rank)
            W = np.random.randn(p * n, rank)
            for c in range(rank):
                nu = np.linalg.norm(U[:, c])
                nv = np.linalg.norm(V[:, c])
                nw = np.linalg.norm(W[:, c])
                if nu > 1e-10: U[:, c] /= nu
                if nv > 1e-10: V[:, c] /= nv
                if nw > 1e-10: W[:, c] /= nw
            
            for it in range(100):
                if time.time() > deadline - 5:
                    break
                # Update U
                for i in range(n):
                    for j in range(m):
                        M = V[j*p:(j+1)*p, :] * W[np.arange(p)*n+i, :]
                        try:
                            x, _, _, _ = np.linalg.lstsq(M.T, np.ones(p), rcond=None)
                            U[i*m+j, :] = x
                        except: pass
                # Update V
                for j in range(m):
                    for k in range(p):
                        M = U[np.arange(n)*m+j, :] * W[k*n+np.arange(n), :]
                        try:
                            x, _, _, _ = np.linalg.lstsq(M.T, np.ones(n), rcond=None)
                            V[j*p+k, :] = x
                        except: pass
                # Update W
                for k in range(p):
                    for i in range(n):
                        M = U[i*m+np.arange(m), :] * V[np.arange(m)*p+k, :]
                        try:
                            x, _, _, _ = np.linalg.lstsq(M.T, np.ones(m), rcond=None)
                            W[k*n+i, :] = x
                        except: pass
                
                max_err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                            err = abs(s - 1.0)
                            if err > max_err: max_err = err
                if max_err < best_err:
                    best_err = max_err
                    best_factors = (U.copy(), V.copy(), W.copy())
                if max_err < 1e-8: break
        
        if best_factors is not None and best_err < 1e-6:
            U, V, W = best_factors
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        return None
    
    best_result = build_schoolbook()
    best_rank = schoolbook_rank
    
    if time.time() < deadline - 25:
        try:
            result65 = build_4x4x5_strassen()
            if verify(result65["U"], result65["V"], result65["W"], tol=1e-6):
                best_result = result65
                best_rank = 65
        except Exception:
            pass
    
    candidate_ranks = [35, 34, 33, 32, 30, 28, 25, 22, 20]
    for target_rank in candidate_ranks:
        if time.time() > deadline - 10: break
        if target_rank >= best_rank: continue
        result = _als_decomposition(target_rank, n, m, p, deadline, n_restarts=5)
        if result is not None:
            if verify(result["U"], result["V"], result["W"], tol=1e-6):
                best_result = result
                best_rank = target_rank
                break
    
    if best_rank > 40 and time.time() < deadline - 8:
        for target_rank in [38, 36, 35, 34, 33, 32, 30]:
            if time.time() > deadline - 3: break
            if target_rank >= best_rank: continue
            result = _als_decomposition(target_rank, n, m, p, deadline, n_restarts=15)
            if result is not None:
                if verify(result["U"], result["V"], result["W"], tol=1e-6):
                    best_result = result
                    best_rank = target_rank
                    break
    
    return best_result
# EVOLVE-BLOCK-END
