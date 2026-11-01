import time
import numpy as np

def _build_strassen_decomposition(n, m, p, num_blocks, last_col_schoolbook=True):
    """Build Strassen-based decomposition for n x m * m x p."""
    rank = num_blocks * 7
    if last_col_schoolbook:
        rank += m * n  # schoolbook for remaining columns
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    for block in range(num_blocks):
        k0 = block * 2
        k1 = block * 2 + 1
        if k1 >= p:
            break
        
        v_e = k0
        v_f = k1
        v_g = p + k0
        v_h = p + k1
        
        w_c00 = k0 * n
        w_c10 = k0 * n + 1
        w_c01 = k1 * n
        w_c11 = k1 * n + 1
        
        # m1 = (a+d)(e+h)
        U[0][r] += 1; U[3][r] += 1
        V[v_e][r] += 1; V[v_h][r] += 1
        W[w_c00][r] += 1; W[w_c11][r] += 1
        r += 1
        
        # m2 = (c+d)e
        U[2][r] += 1; U[3][r] += 1
        V[v_e][r] += 1
        W[w_c10][r] += 1; W[w_c11][r] += -1
        r += 1
        
        # m3 = a(f-h)
        U[0][r] += 1
        V[v_f][r] += 1; V[v_h][r] += -1
        W[w_c01][r] += 1; W[w_c11][r] += 1
        r += 1
        
        # m4 = d(g-e)
        U[3][r] += 1
        V[v_g][r] += 1; V[v_e][r] += -1
        W[w_c00][r] += 1; W[w_c10][r] += 1
        r += 1
        
        # m5 = (a+b)h
        U[0][r] += 1; U[1][r] += 1
        V[v_h][r] += 1
        W[w_c00][r] += -1; W[w_c01][r] += 1
        r += 1
        
        # m6 = (c-a)(e+f)
        U[2][r] += 1; U[0][r] += -1
        V[v_e][r] += 1; V[v_f][r] += 1
        W[w_c11][r] += 1
        r += 1
        
        # m7 = (b-d)(g+h)
        U[1][r] += 1; U[3][r] += -1
        V[v_g][r] += 1; V[v_h][r] += 1
        W[w_c00][r] += 1
        r += 1
    
    # Schoolbook for remaining columns
    if last_col_schoolbook:
        start_col = num_blocks * 2
        for k in range(start_col, p):
            # a * B[0,k]
            U[0][r] += 1
            V[k][r] += 1
            W[k * n][r] += 1
            r += 1
            # b * B[1,k]
            U[1][r] += 1
            V[p + k][r] += 1
            W[k * n][r] += 1
            r += 1
            # c * B[0,k]
            U[2][r] += 1
            V[k][r] += 1
            W[k * n + 1][r] += 1
            r += 1
            # d * B[1,k]
            U[3][r] += 1
            V[p + k][r] += 1
            W[k * n + 1][r] += 1
            r += 1
    
    return {"U": U, "V": V, "W": W}, r

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition is correct."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    rank = U_arr.shape[1]
    
    # Build target tensor
    T_target = np.zeros((n*m, m*p, p*n))
    for a in range(n):
        for b in range(m):
            for c in range(m):
                for k in range(p):
                    i = a*m + b
                    j = b*p + k
                    l = k*n + a
                    T_target[i, j, l] = 1.0
    
    # Reconstruct from decomposition
    T_recon = np.zeros_like(T_target)
    for r_idx in range(rank):
        u_col = U_arr[:, r_idx]
        v_col = V_arr[:, r_idx]
        w_col = W_arr[:, r_idx]
        T_recon += np.einsum('i,j,k->ijk', u_col, v_col, w_col)
    
    return np.allclose(T_target, T_recon, atol=1e-10)

def _try_reduce_rank(U, V, W, n, m, p, max_iterations=50):
    """Try to reduce rank by removing linearly dependent columns."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    rank = U_arr.shape[1]
    keep = list(range(rank))
    
    for iteration in range(max_iterations):
        # Try removing each column
        removed = False
        for i in range(len(keep)):
            test_keep = [k for j, k in enumerate(keep) if j != i]
            if len(test_keep) == 0:
                continue
            
            U_test = U_arr[:, test_keep]
            V_test = V_arr[:, test_keep]
            W_test = W_arr[:, test_keep]
            
            if _verify_decomposition(U_test.tolist(), V_test.tolist(), W_test.tolist(), n, m, p):
                keep = test_keep
                removed = True
                break
        
        if not removed:
            break
    
    if len(keep) < rank:
        return U_arr[:, keep].tolist(), V_arr[:, keep].tolist(), W_arr[:, keep].tolist()
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 100  # Safety margin before 150s limit
    
    n, m, p = 2, 2, 11
    
    # Build the standard Strassen-based decomposition
    result, actual_rank = _build_strassen_decomposition(n, m, p, 5, last_col_schoolbook=True)
    
    U, V, W = result["U"], result["V"], result["W"]
    
    # Verify
    if not _verify_decomposition(U, V, W, n, m, p):
        # Should not happen, but fallback
        pass
    
    # Try to reduce rank
    if time.time() < deadline - 10:
        U, V, W = _try_reduce_rank(U, V, W, n, m, p)
    
    # Final verification
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fallback to original
        result, _ = _build_strassen_decomposition(n, m, p, 5, last_col_schoolbook=True)
        U, V, W = result["U"], result["V"], result["W"]
    
    return {"U": U, "V": V, "W": W}
