# EVOLVE-BLOCK-START
def construct():
    import numpy as np
    import time
    
    start_time = time.time()
    deadline = start_time + 130  # 20s margin before 150s kill
    
    n, m, p = 4, 4, 8
    
    # 1. Construct Schoolbook Decomposition (Rank 128)
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
                
    def to_list(mat):
        return mat.tolist()
        
    best_U = to_list(U_school)
    best_V = to_list(V_school)
    best_W = to_list(W_school)
    best_rank = rank_school
    
    # 2. Attempt to find lower rank via vectorized ALS with integer/half-integer rounding
    def try_rank(rank, seed_base):
        if rank >= best_rank:
            return None
        if time.time() > deadline - 10:
            return None
            
        # Try a few random restarts
        for seed in range(10):
            if time.time() > deadline - 10:
                return None
            rng = np.random.RandomState(seed_base + seed * 1000 + rank)
            U = rng.randn(n * m, rank)
            V = rng.randn(m * p, rank)
            W = rng.randn(p * n, rank)
            
            # Target tensor T[i,j,k] = 1
            # We want sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
            
            # Vectorized ALS
            for it in range(100):
                if time.time() > deadline - 10:
                    return None
                
                # Update U
                for i in range(n):
                    for j in range(m):
                        V_slice = V[j*p : j*p+p, :] # (p, rank)
                        W_slice = W[i::n, :] # (p, rank)
                        
                        A = V_slice * W_slice # (p, rank)
                        b = np.ones(p)
                        
                        try:
                            u, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                            U[i*m + j, :] = u
                        except:
                            pass
                
                # Update V
                for j in range(m):
                    for k in range(p):
                        U_slice = U[j::m, :] # (n, rank)
                        W_slice = W[k*n : k*n+n, :] # (n, rank)
                        
                        B = U_slice * W_slice # (n, rank)
                        c = np.ones(n)
                        
                        try:
                            v, _, _, _ = np.linalg.lstsq(B, c, rcond=None)
                            V[j*p + k, :] = v
                        except:
                            pass
                
                # Update W
                for k in range(p):
                    for i in range(n):
                        U_slice = U[i*m : i*m+m, :] # (m, rank)
                        V_slice = V[k::p, :] # (m, rank)
                        
                        C = U_slice * V_slice # (m, rank)
                        d = np.ones(m)
                        
                        try:
                            w, _, _, _ = np.linalg.lstsq(C, d, rcond=None)
                            W[k*n + i, :] = w
                        except:
                            pass
                
                # Check error
                U_reshaped = U.reshape(n, m, rank)
                V_reshaped = V.reshape(m, p, rank)
                W_reshaped = W.reshape(p, n, rank)
                
                try:
                    T_rec = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
                    error = np.sum((T_rec - 1.0)**2)
                except:
                    error = float('inf')
                
                if error < 1e-6:
                    # Found a good candidate. Try to round to half-integers.
                    U_rounded = np.round(U * 2) / 2
                    V_rounded = np.round(V * 2) / 2
                    W_rounded = np.round(W * 2) / 2
                    
                    # Verify rounded version
                    U_r_reshaped = U_rounded.reshape(n, m, rank)
                    V_r_reshaped = V_rounded.reshape(m, p, rank)
                    W_r_reshaped = W_rounded.reshape(p, n, rank)
                    
                    try:
                        T_rec_r = np.einsum('ijr,jkr,kir->ijk', U_r_reshaped, V_r_reshaped, W_r_reshaped)
                        error_r = np.sum((T_rec_r - 1.0)**2)
                    except:
                        error_r = float('inf')
                    
                    if error_r < 1e-4:
                        return {
                            "U": U_rounded.tolist(),
                            "V": V_rounded.tolist(),
                            "W": W_rounded.tolist()
                        }
            return None
            
    # Try a few lower ranks
    # Schoolbook is 128.
    # We try 120, 118, 116...
    # Given the difficulty, we might not find any.
    for r in range(126, 100, -2):
        if time.time() > deadline - 15:
            break
        res = try_rank(r, 42)
        if res:
            best_U = res["U"]
            best_V = res["V"]
            best_W = res["W"]
            best_rank = r
            break
            
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
