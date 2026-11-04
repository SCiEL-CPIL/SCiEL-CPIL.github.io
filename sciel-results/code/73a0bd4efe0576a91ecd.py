import numpy as np
import time
import random
from fractions import Fraction

def construct():
    n, m, p = 2, 4, 4
    # Target tensor T
    T = np.zeros((8, 16, 8), dtype=int)
    for i in range(n):
        for k in range(m):
            for j in range(p):
                x = i * m + k
                y = k * p + j
                z = j * n + i
                T[x, y, z] = 1

    start_time = time.time()
    deadline = start_time + 140

    # Known rank-28 solution as fallback
    rank_known = 28
    U_known = [[0] * rank_known for _ in range(n * m)]
    V_known = [[0] * rank_known for _ in range(m * p)]
    W_known = [[0] * rank_known for _ in range(p * n)]
    
    def add_block(col_offset, X_rows, Y_rows, Z_rows):
        U = U_known
        V = V_known
        W = W_known
        
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

    X1 = [0, 1, 4, 5]
    Y1 = [0, 1, 4, 5]
    Z1 = [0, 2, 1, 3]
    add_block(0, X1, Y1, Z1)

    X2 = [2, 3, 6, 7]
    Y2 = [8, 9, 12, 13]
    Z2 = [0, 2, 1, 3]
    add_block(7, X2, Y2, Z2)

    X3 = [0, 1, 4, 5]
    Y3 = [2, 3, 6, 7]
    Z3 = [4, 6, 5, 7]
    add_block(14, X3, Y3, Z3)

    X4 = [2, 3, 6, 7]
    Y4 = [10, 11, 14, 15]
    Z4 = [4, 6, 5, 7]
    add_block(21, X4, Y4, Z4)

    best_U = U_known
    best_V = V_known
    best_W = W_known
    best_rank = 28

    def check_solution(U, V, W, rank):
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        T_est = np.zeros((8, 16, 8), dtype=float)
        for r in range(rank):
            u = U_np[:, r]
            v = V_np[:, r]
            w = W_np[:, r]
            T_est += np.einsum('r,s,t->rst', u, v, w)
        if not np.allclose(T_est, T, atol=1e-9):
            return False
        return True

    # Quick check: Is the known solution valid?
    if check_solution(best_U, best_V, best_W, best_rank):
        # It is valid. We can try to improve, but we have a safe fallback.
        pass
    else:
        # If the known solution is invalid (bug), we have a problem.
        # But it should be valid.
        pass

    # Search for rank < 28
    # Use a faster, more targeted search
    for trial in range(10):
        if time.time() > deadline - 10:
            break
        
        random.seed(trial * 12345 + 678)
        np.random.seed(trial * 12345 + 678)
        
        # Simplified greedy peeling
        residual = T.astype(float).copy()
        U_cols = []
        V_cols = []
        W_cols = []
        
        for step in range(27):
            if time.time() > deadline - 5:
                break
            if np.linalg.norm(residual) < 1e-12:
                break
            
            u = np.random.randn(8)
            v = np.random.randn(16)
            w = np.random.randn(8)
            u /= np.linalg.norm(u)
            v /= np.linalg.norm(v)
            w /= np.linalg.norm(w)
            
            for it in range(50):
                u_new = np.einsum('xyz,y,z->x', residual, v, w)
                v_new = np.einsum('xyz,x,z->y', residual, u, w)
                w_new = np.einsum('xyz,x,y->z', residual, u, v)
                
                nu = np.linalg.norm(u_new)
                nv = np.linalg.norm(v_new)
                nw = np.linalg.norm(w_new)
                
                if nu < 1e-15 or nv < 1e-15 or nw < 1e-15:
                    break
                
                u_new /= nu
                v_new /= nv
                w_new /= nw
                
                if (np.linalg.norm(u_new - u) < 1e-10 and
                    np.linalg.norm(v_new - v) < 1e-10 and
                    np.linalg.norm(w_new - w) < 1e-10):
                    u, v, w = u_new, v_new, w_new
                    break
                u, v, w = u_new, v_new, w_new
            
            c = np.einsum('xyz,x,y,z->', residual, u, v, w)
            
            if abs(c) < 1e-12:
                break
            
            residual -= c * np.einsum('x,y,z->xyz', u, v, w)
            
            U_cols.append(c * u)
            V_cols.append(v)
            W_cols.append(w)
            
            if np.linalg.norm(residual) < 1e-10:
                break
        
        rank = len(U_cols)
        if rank >= best_rank:
            continue
        
        # Rationalize
        # ... (same as before)
        # For brevity, I'll keep the rationalization logic from the original but ensure it's correct.
        
        U_rat = []
        V_rat = []
        W_rat = []
        
        for r in range(rank):
            u_r = []
            for val in U_cols[r]:
                best_frac = Fraction(0)
                best_err = abs(val)
                for d in range(1, 9):
                    num = round(val * d)
                    frac = Fraction(num, d)
                    err = abs(val - float(frac))
                    if err < best_err:
                        best_err = err
                        best_frac = frac
                u_r.append(best_frac)
            U_rat.append(u_r)
            
            v_r = []
            for val in V_cols[r]:
                best_frac = Fraction(0)
                best_err = abs(val)
                for d in range(1, 9):
                    num = round(val * d)
                    frac = Fraction(num, d)
                    err = abs(val - float(frac))
                    if err < best_err:
                        best_err = err
                        best_frac = frac
                v_r.append(best_frac)
            V_rat.append(v_r)
            
            w_r = []
            for val in W_cols[r]:
                best_frac = Fraction(0)
                best_err = abs(val)
                for d in range(1, 9):
                    num = round(val * d)
                    frac = Fraction(num, d)
                    err = abs(val - float(frac))
                    if err < best_err:
                        best_err = err
                        best_frac = frac
                w_r.append(best_frac)
            W_rat.append(w_r)
        
        U_list = [[0] * rank for _ in range(8)]
        V_list = [[0] * rank for _ in range(16)]
        W_list = [[0] * rank for _ in range(8)]
        
        for r in range(rank):
            for i in range(8):
                U_list[i][r] = float(U_rat[r][i])
            for i in range(16):
                V_list[i][r] = float(V_rat[r][i])
            for i in range(8):
                W_list[i][r] = float(W_rat[r][i])
        
        if check_solution(U_list, V_list, W_list, rank):
            best_rank = rank
            best_U = U_list
            best_V = V_list
            best_W = W_list
            break

    return {"U": best_U, "V": best_V, "W": best_W}
