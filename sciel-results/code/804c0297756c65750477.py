import numpy as np
import random

def construct():
    n, m, p = 3, 5, 7
    nm, mp, pn = n * m, m * p, p * n
    
    def check(U, V, W):
        # U: (nm, R), V: (mp, R), W: (pn, R)
        R = U.shape[1]
        # Check all i, j, k
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    s = 0
                    for r in range(R):
                        s += U[u_idx, r] * V[v_idx, r] * W[w_idx, r]
                    if s != 1:
                        return False
        return True

    def error(U, V, W):
        R = U.shape[1]
        err = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    s = 0
                    for r in range(R):
                        s += U[u_idx, r] * V[v_idx, r] * W[w_idx, r]
                    err += (1 - s) ** 2
        return err

    # Search for rank < 105
    for rank in range(35, 105):
        # Try a few random initializations
        for attempt in range(20):
            U = np.random.randint(-1, 2, size=(nm, rank))
            V = np.random.randint(-1, 2, size=(mp, rank))
            W = np.random.randint(-1, 2, size=(pn, rank))
            
            if check(U, V, W):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
            # Local search
            current_err = error(U, V, W)
            improved = True
            while improved and current_err > 0:
                improved = False
                # Try changing one entry in U
                for i in range(nm):
                    for r in range(rank):
                        old_val = U[i, r]
                        for new_val in [-1, 0, 1]:
                            if new_val == old_val: continue
                            U[i, r] = new_val
                            new_err = error(U, V, W)
                            if new_err < current_err:
                                current_err = new_err
                                improved = True
                            else:
                                U[i, r] = old_val
                # Try changing one entry in V
                for i in range(mp):
                    for r in range(rank):
                        old_val = V[i, r]
                        for new_val in [-1, 0, 1]:
                            if new_val == old_val: continue
                            V[i, r] = new_val
                            new_err = error(U, V, W)
                            if new_err < current_err:
                                current_err = new_err
                                improved = True
                            else:
                                V[i, r] = old_val
                # Try changing one entry in W
                for i in range(pn):
                    for r in range(rank):
                        old_val = W[i, r]
                        for new_val in [-1, 0, 1]:
                            if new_val == old_val: continue
                            W[i, r] = new_val
                            new_err = error(U, V, W)
                            if new_err < current_err:
                                current_err = new_err
                                improved = True
                            else:
                                W[i, r] = old_val
                
                if current_err == 0:
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback to schoolbook
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
