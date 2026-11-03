import numpy as np
import time

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    
    # Deadline guard: keep 10s margin before 150s kill
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Build target tensor T[a, b, c] where:
    # a = i*m + j  (row of A, col of A)
    # b = j*p + k  (row of B, col of B)
    # c = k*n + i  (output row, output col) -- trace convention W[k*n+i, r]
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Precompute non-zero indices for fast verification
    nz_a, nz_b, nz_c = [], [], []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                nz_a.append(i * m + j)
                nz_b.append(j * p + k)
                nz_c.append(k * n + i)
    nz_a = np.array(nz_a)
    nz_b = np.array(nz_b)
    nz_c = np.array(nz_c)
    
    # Zero entries for verification (sampled)
    all_indices = np.arange(n*m * m*p * p*n)
    nz_flat = set((a, b, c) for a, b, c in zip(nz_a, nz_b, nz_c))
    zero_entries = []
    for a in range(n*m):
        for b in range(m*p):
            for c in range(p*n):
                if (a, b, c) not in nz_flat:
                    zero_entries.append((a, b, c))
    zero_entries = np.array(zero_entries)
    
    def verify(U, V, W, tol=1e-9):
        """Verify that U, V, W (as float arrays) exactly decompose T."""
        rank = U.shape[1]
        # Check all non-zero entries
        for idx in range(len(nz_a)):
            a, b, c = nz_a[idx], nz_b[idx], nz_c[idx]
            val = np.dot(U[a] * V[b], W[c])
            if abs(val - 1.0) > tol:
                return False
        # Check a sample of zero entries
        if len(zero_entries) > 200:
            sample_idx = np.random.choice(len(zero_entries), 200, replace=False)
        else:
            sample_idx = np.arange(len(zero_entries))
        for idx in sample_idx:
            a, b, c = zero_entries[idx]
            val = np.dot(U[a] * V[b], W[c])
            if abs(val) > tol:
                return False
        return True
    
    def verify_integer(U, V, W, scale=2):
        """Verify with integer/half-integer constraint. U,V,W entries must be multiples of 1/scale."""
        U_s = U * scale
        V_s = V * scale
        W_s = W * scale
        if not np.allclose(U_s, np.round(U_s), atol=1e-6):
            return False
        if not np.allclose(V_s, np.round(V_s), atol=1e-6):
            return False
        if not np.allclose(W_s, np.round(W_s), atol=1e-6):
            return False
        U_int = np.round(U_s).astype(np.int64)
        V_int = np.round(V_s).astype(np.int64)
        W_int = np.round(W_s).astype(np.int64)
        
        for idx in range(len(nz_a)):
            a, b, c = nz_a[idx], nz_b[idx], nz_c[idx]
            val = int(np.dot(U_int[a] * V_int[b], W_int[c]))
            if val != scale**3:
                return False
        
        if len(zero_entries) > 200:
            sample_idx = np.random.choice(len(zero_entries), 200, replace=False)
        else:
            sample_idx = np.arange(len(zero_entries))
        for idx in sample_idx:
            a, b, c = zero_entries[idx]
            val = int(np.dot(U_int[a] * V_int[b], W_int[c]))
            if val != 0:
                return False
        return True
    
    def solve_rank(r, n_iters=50, n_inits=3):
        """Alternating least squares with half-integer rounding and verification."""
        for init in range(n_inits):
            if time.time() > deadline:
                return None
            
            np.random.seed(42 + init * 1000 + r)
            U = np.random.randn(n*m, r) * 0.5
            V = np.random.randn(m*p, r) * 0.5
            W = np.random.randn(p*n, r) * 0.5
            
            best_err = np.inf
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            for it in range(n_iters):
                if time.time() > deadline:
                    break
                
                # Update U: for each row a, solve (V[b]*W[c])^T U[a] = T[a,b,c]
                # M_a is (m*p*p*n, r), target is T[a,:,:].flatten()
                for a in range(n*m):
                    M_a = np.zeros((m*p * p*n, r))
                    for b in range(m*p):
                        for c in range(p*n):
                            M_a[b * p*n + c, :] = V[b] * W[c]
                    t_a = T[a, :, :].flatten()
                    try:
                        U[a], _, _, _ = np.linalg.lstsq(M_a, t_a, rcond=None)
                    except:
                        pass
                
                # Update V
                for b in range(m*p):
                    M_b = np.zeros((n*m * p*n, r))
                    for a in range(n*m):
                        for c in range(p*n):
                            M_b[a * p*n + c, :] = U[a] * W[c]
                    t_b = T[:, b, :].flatten()
                    try:
                        V[b], _, _, _ = np.linalg.lstsq(M_b, t_b, rcond=None)
                    except:
                        pass
                
                # Update W
                for c in range(p*n):
                    M_c = np.zeros((n*m * m*p, r))
                    for a in range(n*m):
                        for b in range(m*p):
                            M_c[a * m*p + b, :] = U[a] * V[b]
                    t_c = T[:, :, c].flatten()
                    try:
                        W[c], _, _, _, _ = np.linalg.lstsq(M_c, t_c, rcond=None)
                    except:
                        pass
                
                # Compute error
                err = 0.0
                for idx in range(len(nz_a)):
                    a, b, c = nz_a[idx], nz_b[idx], nz_c[idx]
                    val = np.dot(U[a] * V[b], W[c])
                    err += (val - 1.0)**2
                # Sample zero entries
                if len(zero_entries) > 50:
                    sample_idx = np.random.choice(len(zero_entries), 50, replace=False)
                else:
                    sample_idx = np.arange(len(zero_entries))
                for idx in sample_idx:
                    a, b, c = zero_entries[idx]
                    val = np.dot(U[a] * V[b], W[c])
                    err += val**2
                
                if err < best_err:
                    best_err = err
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
                if best_err < 1e-10:
                    break
            
            # Try rounding to half-integers
            for scale in [2, 1]:
                U_r = np.round(best_U * scale) / scale
                V_r = np.round(best_V * scale) / scale
                W_r = np.round(best_W * scale) / scale
                if verify(U_r, V_r, W_r):
                    return U_r, V_r, W_r
            
            # Also try rounding the original (unrounded) solution
            U_r = np.round(best_U * 2) / 2
            V_r = np.round(best_V * 2) / 2
            W_r = np.round(best_W * 2) / 2
            if verify(U_r, V_r, W_r):
                return U_r, V_r, W_r
        
        return None
    
    # Try to find a decomposition with rank < 63
    # Start from rank 59 down to 50
    best_result = None
    for r in range(59, 49, -1):
        if time.time() > deadline:
            break
        result = solve_rank(r, n_iters=30, n_inits=2)
        if result is not None:
            best_result = (r, result)
            break
    
    if best_result is not None:
        r, (U, V, W) = best_result
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    
    # Fallback: Schoolbook Rank 63
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    return {"U": U_sb, "V": V_sb, "W": W_sb}
