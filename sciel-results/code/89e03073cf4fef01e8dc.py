"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify_decomp(U, V, W, n, m, p, rank):
    """Verify that the decomposition is correct using numpy."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    # Check: sum_lambda W[k,i,lambda] * U[i,l',lambda] * V[l'',k,lambda] == delta[l',l'']
    for i in range(n):
        for k in range(p):
            for lp in range(m):
                for lm in range(m):
                    # Calculate sum over lambda
                    # U[i, lp, lambda], V[lm, k, lambda], W[k, i, lambda]
                    u_vec = U_arr[i*m+lp, :]
                    v_vec = V_arr[lm*p+k, :]
                    w_vec = W_arr[k*n+i, :]
                    s = np.dot(u_vec, v_vec * w_vec)
                    expected = 1.0 if lp == lm else 0.0
                    if abs(s - expected) > 1e-9:
                        return False
    return True

def _add_strassen_2x2x2(U, V, W, col, n, m, p, j0, j1, k0, k1):
    """Add the 7 Strassen products for a 2x2x2 block."""
    # P1 = (A[0,j0]+A[1,j1]) * (B[j0,k0]+B[j1,k1])
    U[0*m+j0][col] += 1; U[1*m+j1][col] += 1
    V[j0*p+k0][col] += 1; V[j1*p+k1][col] += 1
    W[k0*n+0][col] += 1; W[k1*n+1][col] += 1
    col += 1
    # P2 = (A[1,j0]+A[1,j1]) * B[j0,k0]
    U[1*m+j0][col] += 1; U[1*m+j1][col] += 1
    V[j0*p+k0][col] += 1
    W[k0*n+1][col] += 1; W[k1*n+1][col] -= 1
    col += 1
    # P3 = A[0,j0] * (B[j0,k1]-B[j1,k1])
    U[0*m+j0][col] += 1
    V[j0*p+k1][col] += 1; V[j1*p+k1][col] -= 1
    W[k1*n+0][col] += 1; W[k1*n+1][col] += 1
    col += 1
    # P4 = A[1,j1] * (B[j1,k0]-B[j0,k0])
    U[1*m+j1][col] += 1
    V[j1*p+k0][col] += 1; V[j0*p+k0][col] -= 1
    W[k0*n+0][col] += 1; W[k0*n+1][col] += 1
    col += 1
    # P5 = (A[0,j0]+A[0,j1]) * B[j1,k1]
    U[0*m+j0][col] += 1; U[0*m+j1][col] += 1
    V[j1*p+k1][col] += 1
    W[k0*n+0][col] -= 1; W[k1*n+0][col] += 1
    col += 1
    # P6 = (A[1,j0]-A[0,j0]) * (B[j0,k0]+B[j0,k1])
    U[1*m+j0][col] += 1; U[0*m+j0][col] -= 1
    V[j0*p+k0][col] += 1; V[j0*p+k1][col] += 1
    W[k1*n+1][col] += 1
    col += 1
    # P7 = (A[0,j1]-A[1,j1]) * (B[j1,k0]+B[j1,k1])
    U[0*m+j1][col] += 1; U[1*m+j1][col] -= 1
    V[j1*p+k0][col] += 1; V[j1*p+k1][col] += 1
    W[k0*n+0][col] += 1
    col += 1
    return col

def _add_strassen_2x2x1(U, V, W, col, n, m, p, j0, j1, k):
    """Add the 2 Strassen products for a 2x2x1 block."""
    # P1 = (A[0,j0]+A[1,j1]) * B[j0,k]
    U[0*m+j0][col] += 1; U[1*m+j1][col] += 1
    V[j0*p+k][col] += 1
    W[k*n+0][col] += 1; W[k*n+1][col] += 1
    col += 1
    # P2 = A[1,j1] * (B[j1,k]-B[j0,k])
    U[1*m+j1][col] += 1
    V[j1*p+k][col] += 1; V[j0*p+k][col] -= 1
    W[k*n+1][col] -= 1
    col += 1
    return col

def _build_rank54(n, m, p):
    """Build rank-54 decomposition: 6 blocks of 2x2x2 (rank 7) + 12 individual products for j=4."""
    rank = 54
    U = [[0] * rank for _ in range(n*m)]
    V = [[0] * rank for _ in range(m*p)]
    W = [[0] * rank for _ in range(p*n)]
    
    col = 0
    # 6 blocks of 2x2x2
    for j0, j1 in [(0,1), (2,3)]:
        for k0, k1 in [(0,1), (2,3), (4,5)]:
            col = _add_strassen_2x2x2(U, V, W, col, n, m, p, j0, j1, k0, k1)
    
    # j=4: 12 individual products (2 per k, for i=0,1)
    for k in range(p):
        # i=0
        U[0*m+4][col] += 1
        V[4*p+k][col] += 1
        W[k*n+0][col] += 1
        col += 1
        # i=1
        U[1*m+4][col] += 1
        V[4*p+k][col] += 1
        W[k*n+1][col] += 1
        col += 1
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    return U, V, W, rank

def _try_rank_reduction(U, V, W, n, m, p, rank, deadline):
    """Try to reduce the rank by numerical optimization."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    best_U, best_V, best_W = [row[:] for row in U], [row[:] for row in V], [row[:] for row in W]
    best_rank = rank
    
    from scipy.optimize import minimize
    
    for target_rank in range(rank - 1, max(30, rank - 8), -1):
        if time.time() > deadline:
            break
        
        U_init = U_arr[:, :target_rank].copy()
        V_init = V_arr[:, :target_rank].copy()
        W_init = W_arr[:, :target_rank].copy()
        
        rng = np.random.default_rng(42 + target_rank)
        U_init = U_init + rng.normal(0, 0.1, size=U_init.shape)
        V_init = V_init + rng.normal(0, 0.1, size=V_init.shape)
        W_init = W_init + rng.normal(0, 0.1, size=W_init.shape)
        
        def objective(x):
            U_x = x[:n*m*target_rank].reshape(n*m, target_rank)
            V_x = x[n*m*target_rank:n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
            W_x = x[n*m*target_rank + m*p*target_rank:].reshape(p*n, target_rank)
            res = 0.0
            for i in range(n):
                for l in range(m):
                    for k in range(p):
                        s = np.dot(U_x[i*m+l, :], V_x[l*p+k, :] * W_x[k*n+i, :])
                        res += (s - 1.0) ** 2
            return res
        
        x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        try:
            result = minimize(objective, x0, method='Nelder-Mead',
                             options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-10})
            if result.fun < 1e-8:
                U_sol = result.x[:n*m*target_rank].reshape(n*m, target_rank)
                V_sol = result.x[n*m*target_rank:n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
                W_sol = result.x[n*m*target_rank + m*p*target_rank:].reshape(p*n, target_rank)
                if _verify_decomp(U_sol.tolist(), V_sol.tolist(), W_sol.tolist(), n, m, p, target_rank):
                    best_U = U_sol.tolist()
                    best_V = V_sol.tolist()
                    best_W = W_sol.tolist()
                    best_rank = target_rank
                    break
        except Exception:
            continue
    
    return best_U, best_V, best_W, best_rank

def construct():
    n, m, p = DIMENSIONS
    _start_time = time.time()
    _deadline = _start_time + 130
    
    # Build rank-54 decomposition
    U54, V54, W54, rank54 = _build_rank54(n, m, p)
    
    if not _verify_decomp(U54, V54, W54, n, m, p, rank54):
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n*m)]
        V = [[0] * rank for _ in range(m*p)]
        W = [[0] * rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for l in range(m):
                for k in range(p):
                    U[i*m+l][col] = 1
                    V[l*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Return the valid rank-54 decomposition
    # We disable the risky optimization to ensure stability and a valid score
    return {"U": U54, "V": V54, "W": W54}
# EVOLVE-BLOCK-END
