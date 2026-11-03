import numpy as np
import time

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    # Build target tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T[a, b, c] = 1
    
    def verify(U, V, W):
        # U, V, W are expected to be half-integers
        U_int = U * 2
        V_int = V * 2
        W_int = W * 2
        
        if not np.allclose(U_int, np.round(U_int)):
            return False
        if not np.allclose(V_int, np.round(V_int)):
            return False
        if not np.allclose(W_int, np.round(W_int)):
            return False
            
        U_int = np.round(U_int).astype(np.int64)
        V_int = np.round(V_int).astype(np.int64)
        W_int = np.round(W_int).astype(np.int64)
        
        # Check non-zero entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i*m + j
                    b = j*p + k
                    c = k*n + i
                    val = 0
                    for col in range(U_int.shape[1]):
                        val += U_int[a, col] * V_int[b, col] * W_int[c, col]
                    if val != 8: # 1 * 2^3 = 8
                        return False
        
        # Full check
        T_calc = np.einsum('ak,bk,ck->abc', U_int, V_int, W_int)
        T_target = np.zeros_like(T_calc)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i*m + j
                    b = j*p + k
                    c = k*n + i
                    T_target[a, b, c] = 8
                    
        if not np.array_equal(T_calc, T_target):
            return False
            
        return True

    def solve_rank(r, time_limit, seed=0):
        np.random.seed(seed)
        U = np.random.randn(n*m, r)
        V = np.random.randn(m*p, r)
        W = np.random.randn(p*n, r)
        
        # Precompute indices for T non-zeros
        # Actually, we use the full T matrix in lstsq
        
        for it in range(20):
            if time.time() > time_limit:
                break
                
            # Update U
            for a in range(n*m):
                # M_a[b * p*n + c, :] = V[b, :] * W[c, :]
                V_rep = np.repeat(V, p*n, axis=0)
                W_rep = np.tile(W, (m*p, 1))
                M_a = V_rep * W_rep
                
                t_a = T[a, :, :].flatten()
                U[a, :], residuals, rank, sv = np.linalg.lstsq(M_a, t_a, rcond=None)
                
            # Update V
            for b in range(m*p):
                M_b = np.zeros((n*m * p*n, r))
                U_rep = np.repeat(U, p*n, axis=0)
                W_rep = np.tile(W, (n*m, 1))
                M_b = U_rep * W_rep
                t_b = T[:, b, :].flatten()
                V[b, :], residuals, rank, sv = np.linalg.lstsq(M_b, t_b, rcond=None)
                
            # Update W
            for c in range(p*n):
                M_c = np.zeros((n*m * m*p, r))
                U_rep = np.repeat(U, m*p, axis=0)
                V_rep = np.tile(V, (n*m, 1))
                M_c = U_rep * V_rep
                t_c = T[:, :, c].flatten()
                W[c, :], residuals, rank, sv = np.linalg.lstsq(M_c, t_c, rcond=None)
                
            # Round and check periodically
            if it % 2 == 0:
                U_rounded = np.round(U * 2) / 2.0
                V_rounded = np.round(V * 2) / 2.0
                W_rounded = np.round(W * 2) / 2.0
                if verify(U_rounded, V_rounded, W_rounded):
                    return U_rounded, V_rounded, W_rounded
        
        # Final check
        U_rounded = np.round(U * 2) / 2.0
        V_rounded = np.round(V * 2) / 2.0
        W_rounded = np.round(W * 2) / 2.0
        if verify(U_rounded, V_rounded, W_rounded):
            return U_rounded, V_rounded, W_rounded
        return None

    # Try ranks from 62 down to 45
    start_time = time.time()
    deadline = start_time + 100 # Leave 20s margin
    
    # We know 63 is possible. Try to find < 63.
    for r in range(62, 44, -1):
        if time.time() > deadline:
            break
        # Try a few initializations
        for init in range(3):
            if time.time() > deadline:
                break
            res = solve_rank(r, deadline - 5, seed=init)
            if res is not None:
                U, V, W = res
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W.tolist()
                return {"U": U_list, "V": V_list, "W": W_list}
                
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
