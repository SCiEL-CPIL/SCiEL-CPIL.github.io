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

    # Greedy peeling: iteratively extract rank-1 components from residual
    def greedy_peel(T_target, max_rank=30):
        """Greedy algorithm: at each step, find best rank-1 approx to residual."""
        residual = T_target.astype(float).copy()
        U_cols = []
        V_cols = []
        W_cols = []
        
        for step in range(max_rank):
            if time.time() > deadline:
                break
            if np.linalg.norm(residual) < 1e-12:
                break
            
            # Find best rank-1 approximation via mode-n flattenings
            # T[x,y,z] = sum_r U[x,r]*V[y,r]*W[z,r]
            # Residual R[x,y,z]. Best rank-1: maximize |R contracted with u,v,w|
            # Use power iteration on the 3-way tensor
            
            # Flatten to matrices for SVD-like approach
            # R as matrix: R_mat[x, y*p+z] = R[x,y,z]
            # Or use the fact that best rank-1 is related to singular vectors
            
            # Power iteration for 3-way tensor
            u = np.random.randn(8)
            v = np.random.randn(16)
            w = np.random.randn(8)
            u /= np.linalg.norm(u)
            v /= np.linalg.norm(v)
            w /= np.linalg.norm(w)
            
            for it in range(100):
                # Update u: u[x] = sum_{y,z} R[x,y,z] * v[y] * w[z]
                u_new = np.einsum('xyz,y,z->x', residual, v, w)
                # Update v: v[y] = sum_{x,z} R[x,y,z] * u[x] * w[z]
                v_new = np.einsum('xyz,x,z->y', residual, u, w)
                # Update w: w[z] = sum_{x,y} R[x,y,z] * u[x] * v[y]
                w_new = np.einsum('xyz,x,y->z', residual, u, v)
                
                nu = np.linalg.norm(u_new)
                nv = np.linalg.norm(v_new)
                nw = np.linalg.norm(w_new)
                
                if nu < 1e-15 or nv < 1e-15 or nw < 1e-15:
                    break
                
                u_new /= nu
                v_new /= nv
                w_new /= nw
                
                # Check convergence
                if (np.linalg.norm(u_new - u) < 1e-10 and
                    np.linalg.norm(v_new - v) < 1e-10 and
                    np.linalg.norm(w_new - w) < 1e-10):
                    u, v, w = u_new, v_new, w_new
                    break
                u, v, w = u_new, v_new, w_new
            
            # Compute the coefficient: c = <R, u⊗v⊗w> / (||u||^2 * ||v||^2 * ||w||^2)
            # Since u,v,w are unit: c = <R, u⊗v⊗w>
            c = np.einsum('xyz,x,y,z->', residual, u, v, w)
            
            if abs(c) < 1e-12:
                break
            
            # Subtract rank-1 component
            residual -= c * np.einsum('x,y,z->xyz', u, v, w)
            
            # Store the component (we'll rationalize later)
            U_cols.append(c * u)
            V_cols.append(v)
            W_cols.append(w)
            
            if np.linalg.norm(residual) < 1e-10:
                break
        
        return U_cols, V_cols, W_cols

    def rationalize_vector(vec, max_denom=8):
        """Convert a float vector to half-integers (denom up to 8)."""
        result = []
        for val in vec:
            best_frac = Fraction(0)
            best_err = abs(val)
            for d in range(1, max_denom + 1):
                num = round(val * d)
                frac = Fraction(num, d)
                err = abs(val - float(frac))
                if err < best_err:
                    best_err = err
                    best_frac = frac
            result.append(best_frac)
        return result

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

    # Try greedy peeling with multiple random initializations
    best_rank = 28
    best_U = None
    best_V = None
    best_W = None

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

    # Try greedy peeling with various initializations
    for trial in range(20):
        if time.time() > deadline:
            break
        
        # Random initialization for power iteration
        random.seed(trial * 12345 + 678)
        np.random.seed(trial * 12345 + 678)
        
        U_cols, V_cols, W_cols = greedy_peel(T, max_rank=30)
        
        if not U_cols:
            continue
        
        rank = len(U_cols)
        if rank >= best_rank:
            continue
        
        # Rationalize: convert to half-integers
        U_rat = []
        V_rat = []
        W_rat = []
        valid = True
        
        for r in range(rank):
            u_r = rationalize_vector(U_cols[r], max_denom=8)
            v_r = rationalize_vector(V_cols[r], max_denom=8)
            w_r = rationalize_vector(W_cols[r], max_denom=8)
            
            for i in range(8):
                U_rat.append(u_r[i])
            for i in range(16):
                V_rat.append(v_r[i])
            for i in range(8):
                W_rat.append(w_r[i])
        
        # Convert to list-of-lists format
        U_list = [[0] * rank for _ in range(8)]
        V_list = [[0] * rank for _ in range(16)]
        W_list = [[0] * rank for _ in range(8)]
        
        for r in range(rank):
            for i in range(8):
                U_list[i][r] = float(U_rat[i * rank + r])
            for i in range(16):
                V_list[i][r] = float(V_rat[i * rank + r])
            for i in range(8):
                W_list[i][r] = float(W_rat[i * rank + r])
        
        if check_solution(U_list, V_list, W_list, rank):
            best_rank = rank
            best_U = U_list
            best_V = V_list
            best_W = W_list
            break
    
    return {"U": best_U, "V": best_V, "W": best_W}
