import time
import numpy as np
from fractions import Fraction
import scipy.linalg

def construct():
    n, m, p = 3, 4, 4
    dim_u = n * m  # 12
    dim_v = m * p  # 16
    dim_w = p * n  # 12

    start_time = time.time()
    deadline = start_time + 105

    def build_rank44():
        rank = 44
        U = [[Fraction(0)] * rank for _ in range(dim_u)]
        V = [[Fraction(0)] * rank for _ in range(dim_v)]
        W = [[Fraction(0)] * rank for _ in range(dim_w)]

        col = 0
        def add_strassen(X_u, Y_v, Z_w, col):
            U[X_u[0]][col] += 1; U[X_u[3]][col] += 1
            V[Y_v[0]][col] += 1; V[Y_v[3]][col] += 1
            W[Z_w[0]][col] += 1; W[Z_w[3]][col] += 1
            col += 1
            U[X_u[2]][col] += 1; U[X_u[3]][col] += 1
            V[Y_v[0]][col] += 1
            W[Z_w[2]][col] += 1; W[Z_w[3]][col] -= 1
            col += 1
            U[X_u[0]][col] += 1
            V[Y_v[1]][col] += 1; V[Y_v[3]][col] -= 1
            W[Z_w[1]][col] += 1; W[Z_w[3]][col] += 1
            col += 1
            U[X_u[3]][col] += 1
            V[Y_v[2]][col] += 1; V[Y_v[0]][col] -= 1
            W[Z_w[0]][col] += 1; W[Z_w[2]][col] += 1
            col += 1
            U[X_u[0]][col] += 1; U[X_u[1]][col] += 1
            V[Y_v[3]][col] += 1
            W[Z_w[0]][col] -= 1; W[Z_w[1]][col] += 1
            col += 1
            U[X_u[2]][col] += 1; U[X_u[0]][col] -= 1
            V[Y_v[0]][col] += 1; V[Y_v[1]][col] += 1
            W[Z_w[3]][col] += 1
            col += 1
            U[X_u[1]][col] += 1; U[X_u[3]][col] -= 1
            V[Y_v[2]][col] += 1; V[Y_v[3]][col] += 1
            W[Z_w[0]][col] += 1
            col += 1
            return col

        # Four 2x2x2 Strassen blocks
        col = add_strassen([0, 1, 4, 5], [0, 1, 4, 5], [0, 3, 1, 4], col)
        col = add_strassen([2, 3, 6, 7], [8, 9, 12, 13], [0, 3, 1, 4], col)
        col = add_strassen([0, 1, 4, 5], [2, 3, 6, 7], [6, 9, 7, 10], col)
        col = add_strassen([2, 3, 6, 7], [10, 11, 14, 15], [6, 9, 7, 10], col)
        
        # Schoolbook for the 3rd row of A
        for j in range(4):
            U[8+j][col] += 1; V[j*4+0][col] += 1; W[2][col] += 1
            col += 1
            U[8+j][col] += 1; V[j*4+1][col] += 1; W[5][col] += 1
            col += 1
        for j in range(4):
            U[8+j][col] += 1; V[j*4+2][col] += 1; W[8][col] += 1
            col += 1
            U[8+j][col] += 1; V[j*4+3][col] += 1; W[11][col] += 1
            col += 1

        return U, V, W, col

    # Precompute expected tensor
    T_expected = np.zeros((dim_u, dim_v, dim_w), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_expected[i*m+j, j*p+k, k*n+i] = 1.0

    def verify_frac(U, V, W):
        U_arr = np.array([[float(x) for x in row] for row in U], dtype=np.float64)
        V_arr = np.array([[float(x) for x in row] for row in V], dtype=np.float64)
        W_arr = np.array([[float(x) for x in row] for row in W], dtype=np.float64)
        
        if U_arr.size == 0:
            return np.allclose(np.zeros_like(T_expected), T_expected, atol=1e-10, rtol=1e-10)
            
        T_actual = np.einsum('ir,jr,kr->ijk', U_arr, V_arr, W_arr)
        return np.allclose(T_actual, T_expected, atol=1e-10, rtol=1e-10)

    U, V, W, actual_rank = build_rank44()
    
    if not verify_frac(U, V, W):
        # Fallback to schoolbook
        rank = 48
        U = [[0.0]*rank for _ in range(dim_u)]
        V = [[0.0]*rank for _ in range(dim_v)]
        W = [[0.0]*rank for _ in range(dim_w)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1.0
                    V[j*p+k][col] = 1.0
                    W[k*n+i][col] = 1.0
                    col += 1
        return {"U": U, "V": V, "W": W}

    # Convert to float arrays for linear algebra
    U_np = np.array([[float(x) for x in row] for row in U], dtype=np.float64)
    V_np = np.array([[float(x) for x in row] for row in V], dtype=np.float64)
    W_np = np.array([[float(x) for x in row] for row in W], dtype=np.float64)
    
    current_U = U_np
    current_V = V_np
    current_W = W_np
    current_rank = actual_rank
    
    # Use SVD to find null space and guide removal
    if actual_rank > 1 and time.time() < deadline:
        try:
            M = np.einsum('ur,vr,wr->uvwr', U_np, V_np, W_np).reshape(-1, actual_rank)
            _, s, vh = np.linalg.svd(M, full_matrices=False)
            tol = 1e-9 * max(s)
            rank_M = int(np.sum(s > tol))
            
            # The null space vectors tell us which combinations are zero
            # We can use the right singular vectors to identify which factors to remove
            # Factors corresponding to small singular values are candidates for removal
            # Sort factors by their singular value (smallest first)
            sorted_indices = np.argsort(s)
            
            # Try removing factors in order of smallest singular value
            for r_idx in sorted_indices:
                if time.time() > deadline:
                    break
                if r_idx >= current_rank:
                    continue
                keep = [i for i in range(current_rank) if i != r_idx]
                
                U_test = current_U[:, keep]
                V_test = current_V[:, keep]
                W_test = current_W[:, keep]
                
                T_test = np.einsum('ir,jr,kr->ijk', U_test, V_test, W_test)
                if np.allclose(T_test, T_expected, atol=1e-10, rtol=1e-10):
                    current_U = U_test
                    current_V = V_test
                    current_W = W_test
                    current_rank -= 1
        except Exception:
            pass
    
    # Continue with greedy removal for any remaining dependencies
    max_removals = 15
    removed = 0
    while removed < max_removals and current_rank > 1:
        if time.time() > deadline:
            break
        found = False
        for r_idx in range(current_rank):
            keep = [i for i in range(current_rank) if i != r_idx]
            
            U_test = current_U[:, keep]
            V_test = current_V[:, keep]
            W_test = current_W[:, keep]
            
            T_test = np.einsum('ir,jr,kr->ijk', U_test, V_test, W_test)
            if np.allclose(T_test, T_expected, atol=1e-10, rtol=1e-10):
                current_U = U_test
                current_V = V_test
                current_W = W_test
                current_rank -= 1
                removed += 1
                found = True
                break
        if not found:
            break
            
    return {"U": current_U.tolist(), "V": current_V.tolist(), "W": current_W.tolist()}
