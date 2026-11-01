import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def _add_strassen_block(U, V, W, col, i_off, j_off, k_off):
    """Add Strassen 2x2x2 block at given offsets. Returns new col."""
    n, m, p = 2, 3, 5
    U[i_off * m + j_off][col] = 1
    U[(i_off+1) * m + (j_off+1)][col] = 1
    V[j_off * p + k_off][col] = 1
    V[(j_off+1) * p + (k_off+1)][col] = 1
    W[k_off * n + i_off][col] = 1
    W[(k_off+1) * n + (i_off+1)][col] = 1
    col += 1
    
    U[(i_off+1) * m + j_off][col] = 1
    U[(i_off+1) * m + (j_off+1)][col] = 1
    V[j_off * p + k_off][col] = 1
    W[k_off * n + (i_off+1)][col] = 1
    W[(k_off+1) * n + (i_off+1)][col] = -1
    col += 1
    
    U[i_off * m + j_off][col] = 1
    V[j_off * p + (k_off+1)][col] = 1
    V[(j_off+1) * p + (k_off+1)][col] = -1
    W[(k_off+1) * n + i_off][col] = 1
    W[(k_off+1) * n + (i_off+1)][col] = 1
    col += 1
    
    U[(i_off+1) * m + (j_off+1)][col] = 1
    V[(j_off+1) * p + k_off][col] = 1
    V[j_off * p + k_off][col] = -1
    W[k_off * n + i_off][col] = 1
    W[k_off * n + (i_off+1)][col] = 1
    col += 1
    
    U[i_off * m + j_off][col] = 1
    U[i_off * m + (j_off+1)][col] = 1
    V[(j_off+1) * p + (k_off+1)][col] = 1
    W[k_off * n + i_off][col] = -1
    W[(k_off+1) * n + i_off][col] = 1
    col += 1
    
    U[(i_off+1) * m + j_off][col] = 1
    U[i_off * m + j_off][col] = -1
    V[j_off * p + k_off][col] = 1
    V[j_off * p + (k_off+1)][col] = 1
    W[(k_off+1) * n + (i_off+1)][col] = 1
    col += 1
    
    U[i_off * m + (j_off+1)][col] = 1
    U[(i_off+1) * m + (j_off+1)][col] = -1
    V[(j_off+1) * p + k_off][col] = 1
    V[(j_off+1) * p + (k_off+1)][col] = 1
    W[k_off * n + i_off][col] = 1
    col += 1
    
    return col

def _verify_full(U, V, W, n=2, m=3, p=5):
    """Verify full 2x3x5 decomposition."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = sum(U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] for r in range(rank))
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _try_rank27(max_time=120):
    """Try to find rank-27 decomposition of 2x3x5."""
    start = time.time()
    n, m, p = 2, 3, 5
    rank = 27
    num_U = n * m * rank  # 6*27=162
    num_V = m * p * rank  # 15*27=405
    num_W = p * n * rank  # 10*27=270
    total = num_U + num_V + num_W  # 837
    
    def objective(x):
        U = x[0:num_U].reshape(n, m, rank)
        V = x[num_U:num_U+num_V].reshape(m, p, rank)
        W = x[num_U+num_V:total].reshape(p, n, rank)
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = float(np.dot(U[i,j,:], V[j,k,:] * W[k,i,:]))
                    err += (s - 1.0) ** 2
        return err
    
    best_val = np.inf
    best_x = None
    
    # Start from the rank-28 solution and zero out one column, then optimize
    def make_rank28_init():
        U = np.zeros((n, m, 28))
        V = np.zeros((m, p, 28))
        W = np.zeros((p, n, 28))
        col = 0
        for i in range(n):
            for k in range(p):
                U[i, 0, col] = 1
                V[0, k, col] = 1
                W[k, i, col] = 1
                col += 1
        # Strassen for j={1,2}, k={0,1}
        i_off, j_off, k_off = 0, 1, 0
        U[i_off*m+j_off, col] = 1; U[(i_off+1)*m+(j_off+1), col] = 1
        V[j_off*p+k_off, col] = 1; V[(j_off+1)*p+(k_off+1), col] = 1
        W[k_off*n+i_off, col] = 1; W[(k_off+1)*n+(i_off+1), col] = 1
        col += 1
        U[(i_off+1)*m+j_off, col] = 1; U[(i_off+1)*m+(j_off+1), col] = 1
        V[j_off*p+k_off, col] = 1
        W[k_off*n+(i_off+1), col] = 1; W[(k_off+1)*n+(i_off+1), col] = -1
        col += 1
        U[i_off*m+j_off, col] = 1
        V[j_off*p+(k_off+1), col] = 1; V[(j_off+1)*p+(k_off+1), col] = -1
        W[(k_off+1)*n+i_off, col] = 1; W[(k_off+1)*n+(i_off+1), col] = 1
        col += 1
        U[(i_off+1)*m+(j_off+1), col] = 1
        V[(j_off+1)*p+k_off, col] = 1; V[j_off*p+k_off, col] = -1
        W[k_off*n+i_off, col] = 1; W[k_off*n+(i_off+1), col] = 1
        col += 1
        U[i_off*m+j_off, col] = 1; U[i_off*m+(j_off+1), col] = 1
        V[(j_off+1)*p+(k_off+1), col] = 1
        W[k_off*n+i_off, col] = -1; W[(k_off+1)*n+i_off, col] = 1
        col += 1
        U[(i_off+1)*m+j_off, col] = 1; U[i_off*m+j_off, col] = -1
        V[j_off*p+k_off, col] = 1; V[j_off*p+(k_off+1), col] = 1
        W[(k_off+1)*n+(i_off+1), col] = 1
        col += 1
        U[i_off*m+(j_off+1), col] = 1; U[(i_off+1)*m+(j_off+1), col] = -1
        V[(j_off+1)*p+k_off, col] = 1; V[(j_off+1)*p+(k_off+1), col] = 1
        W[k_off*n+i_off, col] = 1
        col += 1
        # Schoolbook for i in {0,1}, j in {1,2}, k=2
        for i in range(n):
            for j in range(1, 3):
                U[i, j, col] = 1
                V[j, 2, col] = 1
                W[2, i, col] = 1
                col += 1
        # Strassen for j={1,2}, k={3,4}
        i_off, j_off, k_off = 0, 1, 3
        U[i_off*m+j_off, col] = 1; U[(i_off+1)*m+(j_off+1), col] = 1
        V[j_off*p+k_off, col] = 1; V[(j_off+1)*p+(k_off+1), col] = 1
        W[k_off*n+i_off, col] = 1; W[(k_off+1)*n+(i_off+1), col] = 1
        col += 1
        U[(i_off+1)*m+j_off, col] = 1; U[(i_off+1)*m+(j_off+1), col] = 1
        V[j_off*p+k_off, col] = 1
        W[k_off*n+(i_off+1), col] = 1; W[(k_off+1)*n+(i_off+1), col] = -1
        col += 1
        U[i_off*m+j_off, col] = 1
        V[j_off*p+(k_off+1), col] = 1; V[(j_off+1)*p+(k_off+1), col] = -1
        W[(k_off+1)*n+i_off, col] = 1; W[(k_off+1)*n+(i_off+1), col] = 1
        col += 1
        U[(i_off+1)*m+(j_off+1), col] = 1
        V[(j_off+1)*p+k_off, col] = 1; V[j_off*p+k_off, col] = -1
        W[k_off*n+i_off, col] = 1; W[k_off*n+(i_off+1), col] = 1
        col += 1
        U[i_off*m+j_off, col] = 1; U[i_off*m+(j_off+1), col] = 1
        V[(j_off+1)*p+(k_off+1), col] = 1
        W[k_off*n+i_off, col] = -1; W[(k_off+1)*n+i_off, col] = 1
        col += 1
        U[(i_off+1)*m+j_off, col] = 1; U[i_off*m+j_off, col] = -1
        V[j_off*p+k_off, col] = 1; V[j_off*p+(k_off+1), col] = 1
        W[(k_off+1)*n+(i_off+1), col] = 1
        col += 1
        U[i_off*m+(j_off+1), col] = 1; U[(i_off+1)*m+(j_off+1), col] = -1
        V[(j_off+1)*p+k_off, col] = 1; V[(j_off+1)*p+(k_off+1), col] = 1
        W[k_off*n+i_off, col] = 1
        col += 1
        assert col == 28
        return U, V, W
    
    # Create rank-27 init by removing one column from rank-28
    U28, V28, W28 = make_rank28_init()
    
    for restart in range(30):
        if time.time() - start > max_time - 10:
            break
        
        np.random.seed(2000 + restart)
        
        # Remove a random column from rank-28 to get rank-27 init
        col_to_remove = np.random.randint(0, 28)
        keep = [c for c in range(28) if c != col_to_remove]
        
        U_init = np.zeros((n, m, rank))
        V_init = np.zeros((m, p, rank))
        W_init = np.zeros((p, n, rank))
        for new_c, old_c in enumerate(keep):
            U_init[:, :, new_c] = U28[:, :, old_c]
            V_init[:, :, new_c] = V28[:, :, old_c]
            W_init[:, :, new_c] = W28[:, :, old_c]
        
        # Add small noise
        noise_scale = 0.01
        U_init += np.random.randn(n, m, rank) * noise_scale
        V_init += np.random.randn(m, p, rank) * noise_scale
        W_init += np.random.randn(p, n, rank) * noise_scale
        
        x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        try:
            from scipy.optimize import minimize
            res = minimize(objective, x0, method='L-BFGS-B',
                           options={'maxiter': 200, 'ftol': 1e-18, 'gtol': 1e-18})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
            if best_val < 1e-10:
                break
        except Exception:
            continue
    
    if best_x is None or best_val > 1e-8:
        return None
    
    U = best_x[0:num_U].reshape(n, m, rank)
    V = best_x[num_U:num_U+num_V].reshape(m, p, rank)
    W = best_x[num_U+num_V:total].reshape(p, n, rank)
    
    # Convert to list format and try rounding
    U_list = [[0]*rank for _ in range(n*m)]
    V_list = [[0]*rank for _ in range(m*p)]
    W_list = [[0]*rank for _ in range(p*n)]
    
    for i in range(n):
        for j in range(m):
            for r in range(rank):
                U_list[i*m+j][r] = U[i,j,r]
    for j in range(m):
        for k in range(p):
            for r in range(rank):
                V_list[j*p+k][r] = V[j,k,r]
    for k in range(p):
        for i in range(n):
            for r in range(rank):
                W_list[k*n+i][r] = W[k,i,r]
    
    # Try rounding to half-integers
    for mult in [1, 2, 3, 4, 5, 6, 8]:
        U_r = [[round(U_list[i][r] * mult) / mult for r in range(rank)] for i in range(n*m)]
        V_r = [[round(V_list[i][r] * mult) / mult for r in range(rank)] for i in range(m*p)]
        W_r = [[round(W_list[i][r] * mult) / mult for r in range(rank)] for i in range(p*n)]
        if _verify_full(U_r, V_r, W_r):
            return {"U": U_r, "V": V_r, "W": W_r}
    
    return None

def construct():
    n, m, p = 2, 3, 5
    deadline = time.time() + 145
    
    # Try to find rank-27 solution
    if time.time() < deadline - 5:
        t_left = min(130, deadline - time.time() - 5)
        result27 = _try_rank27(max_time=t_left)
        if result27 is not None:
            return result27
    
    # Fallback: rank-28 solution
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for k in range(p):
            U[i * m + 0][col] = 1
            V[0 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
    
    col = _add_strassen_block(U, V, W, col, 0, 1, 0)
    
    for i in range(n):
        for j in range(1, 3):
            U[i * m + j][col] = 1
            V[j * p + 2][col] = 1
            W[2 * n + i][col] = 1
            col += 1
    
    col = _add_strassen_block(U, V, W, col, 0, 1, 3)
    
    assert col == rank, f"Column mismatch: {col} != {rank}"
    return {"U": U, "V": V, "W": W}
