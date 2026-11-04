import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0

    target_tensor = np.ones((n, m, p))
    
    def compute_residual(U, V, W):
        res = np.zeros((n, m, p))
        r = U.shape[1]
        for c in range(r):
            U_c = U[:, c].reshape(n, m)
            V_c = V[:, c].reshape(m, p)
            W_c = W[:, c].reshape(p, n)
            U_part = U_c[:, :, np.newaxis]
            V_part = V_c[np.newaxis, :, :]
            W_part = W_c.T[:, np.newaxis, :]
            res += U_part * V_part * W_part
        return np.sum((res - target_tensor) ** 2)

    def solve_linear_U(V, W, r):
        # Solve for U given V, W
        # Eq: sum_c U_c[i,j] V_c[j,k] W_c[k,i] = 1
        # Unknowns: U_c[i,j] for c in r, i in n, j in m
        # Total unknowns: r * n * m
        # Equations: n * m * p
        size_u = n * m
        A = np.zeros((n * m * p, r * size_u))
        b = np.ones(n * m * p)
        
        for c in range(r):
            V_c = V[:, c].reshape(m, p)
            W_c = W[:, c].reshape(p, n)
            for i in range(n):
                for j in range(m):
                    u_idx = c * size_u + i * m + j
                    for k in range(p):
                        eq_idx = (i * m + j) * p + k
                        # Coeff of U_c[i,j] in eq (i,j,k) is V_c[j,k] * W_c[k,i]
                        A[eq_idx, u_idx] = V_c[j, k] * W_c[k, i]
        
        # Solve least squares
        U, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        return U.reshape(size_u, r)

    def solve_linear_V(U, W, r):
        # Solve for V given U, W
        # Eq: sum_c U_c[i,j] V_c[j,k] W_c[k,i] = 1
        # Unknowns: V_c[j,k]
        size_v = m * p
        A = np.zeros((n * m * p, r * size_v))
        b = np.ones(n * m * p)
        
        for c in range(r):
            U_c = U[:, c].reshape(n, m)
            W_c = W[:, c].reshape(p, n)
            for j in range(m):
                for k in range(p):
                    v_idx = c * size_v + j * p + k
                    for i in range(n):
                        eq_idx = (i * m + j) * p + k
                        A[eq_idx, v_idx] = U_c[i, j] * W_c[k, i]
        
        V, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        return V.reshape(size_v, r)

    def solve_linear_W(U, V, r):
        # Solve for W given U, V
        # Eq: sum_c U_c[i,j] V_c[j,k] W_c[k,i] = 1
        # Unknowns: W_c[k,i]
        size_w = p * n
        A = np.zeros((n * m * p, r * size_w))
        b = np.ones(n * m * p)
        
        for c in range(r):
            U_c = U[:, c].reshape(n, m)
            V_c = V[:, c].reshape(m, p)
            for k in range(p):
                for i in range(n):
                    w_idx = c * size_w + k * n + i
                    for j in range(m):
                        eq_idx = (i * m + j) * p + k
                        A[eq_idx, w_idx] = U_c[i, j] * V_c[j, k]
        
        W, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        return W.reshape(size_w, r)

    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
        
    fallback = schoolbook()
    best_result = None
    best_rank = n * m * p

    # Try ranks from 1 to 30
    for r in range(1, 31):
        if time.time() > deadline:
            break
        if r >= best_rank:
            continue
            
        size_u = n * m
        size_v = m * p
        size_w = p * n
        
        max_restarts = 10
        found = False
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
                
            # Random initialization
            U = np.random.randn(size_u, r)
            V = np.random.randn(size_v, r)
            W = np.random.randn(size_w, r)
            
            # ALS iterations
            for _ in range(50):
                if time.time() > deadline:
                    break
                U = solve_linear_U(V, W, r)
                V = solve_linear_V(U, W, r)
                W = solve_linear_W(U, V, r)
                
                res_val = compute_residual(U, V, W)
                if res_val < 1e-8:
                    # Snap to half-integers
                    U_sol = np.round(U * 2) / 2
                    V_sol = np.round(V * 2) / 2
                    W_sol = np.round(W * 2) / 2
                    
                    # Verify
                    res_val_sol = compute_residual(U_sol, V_sol, W_sol)
                    if res_val_sol < 1e-9:
                        best_rank = r
                        best_result = {
                            "U": U_sol.tolist(),
                            "V": V_sol.tolist(),
                            "W": W_sol.tolist()
                        }
                        found = True
                        break
            if found:
                break
        
    if best_result:
        return best_result
    else:
        return fallback

# EVOLVE-BLOCK-END
