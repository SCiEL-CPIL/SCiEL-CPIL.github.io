import numpy as np
import time

DIMENSIONS = [2, 3, 10]

def construct():
    start_time = time.time()
    deadline = 145.0  # Leave 5s safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Build target tensor: T[i,j,k] = 1 for all valid (i,j,k)
    T = np.ones((n, m, p), dtype=float)
    
    def verify_fast(U, V, W, r):
        """Vectorized verification: build tensor of triple products and check max |T-1|."""
        # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # Reshape U to (n, m, r), V to (m, p, r), W to (p, n, r)
        Ur = U.reshape(n, m, r)
        Vr = V.reshape(m, p, r)
        Wr = W.reshape(p, n, r)
        # Compute T via einsum: T[i,j,k] = sum_c Ur[i,j,c] * Vr[j,k,c] * Wr[k,i,c]
        T = np.einsum('ijc,jkc,kic->ijk', Ur, Vr, Wr, optimize=True)
        return float(np.max(np.abs(T - 1.0)))
    
    # Alias for backward compatibility
    verify = verify_fast
    
    # Rank-55 decomposition: j=0 separately + Strassen for j=1,2
    def rank55_decomposition():
        rank = 55
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        
        col = 0
        
        # Part A: j=0 (first column of A, first row of B)
        for i in range(n):
            for k in range(p):
                U[i*m + 0, col] = 1
                V[0*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
        
        # Part B: j=1,2 with Strassen on 5 blocks of 2x2
        for b in range(5):
            k0 = 2 * b
            k1 = 2 * b + 1
            
            w_00 = k0 * n + 0
            w_01 = k1 * n + 0
            w_10 = k0 * n + 1
            w_11 = k1 * n + 1
            
            u_11 = 0 * m + 1
            u_12 = 0 * m + 2
            u_21 = 1 * m + 1
            u_22 = 1 * m + 2
            
            v_11 = 1 * p + k0
            v_12 = 1 * p + k1
            v_21 = 2 * p + k0
            v_22 = 2 * p + k1
            
            # M1 = (a11 + a22)(b11 + b22)
            U[u_11, col] += 1
            U[u_22, col] += 1
            V[v_11, col] += 1
            V[v_22, col] += 1
            W[w_00, col] += 1
            W[w_11, col] += 1
            col += 1
            
            # M2 = (a21 + a22)b11
            U[u_21, col] += 1
            U[u_22, col] += 1
            V[v_11, col] += 1
            W[w_10, col] += 1
            W[w_11, col] += -1
            col += 1
            
            # M3 = a11(b12 - b22)
            U[u_11, col] += 1
            V[v_12, col] += 1
            V[v_22, col] += -1
            W[w_01, col] += 1
            W[w_11, col] += 1
            col += 1
            
            # M4 = a22(b21 - b11)
            U[u_22, col] += 1
            V[v_21, col] += 1
            V[v_11, col] += -1
            W[w_00, col] += 1
            W[w_10, col] += 1
            col += 1
            
            # M5 = (a11 + a12)b22
            U[u_11, col] += 1
            U[u_12, col] += 1
            V[v_22, col] += 1
            W[w_00, col] += -1
            W[w_01, col] += 1
            col += 1
            
            # M6 = (a21 - a11)(b11 + b12)
            U[u_21, col] += 1
            U[u_11, col] += -1
            V[v_11, col] += 1
            V[v_12, col] += 1
            W[w_11, col] += 1
            col += 1
            
            # M7 = (a12 - a22)(b21 + b22)
            U[u_12, col] += 1
            U[u_22, col] += -1
            V[v_21, col] += 1
            V[v_22, col] += 1
            W[w_00, col] += 1
            col += 1
        
        return U, V, W, rank
    
    U_best, V_best, W_best, rank_best = rank55_decomposition()
    err = verify(U_best, V_best, W_best, rank_best)
    
    def block_als(U, V, W, r, max_iters=30, tol=1e-12):
        """ALS with block-wise solves exploiting tensor structure (vectorized)."""
        ones_m = np.ones(m)
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Solve for U: for each (i,k), M[j,c] = V[j*p+k,c]*W[k*n+i,c]; solve M x = ones(m)
            U_new = np.zeros_like(U)
            for i in range(n):
                for k in range(p):
                    M_sub = V[np.arange(m)*p + k, :] * W[k*n + i, :]
                    try:
                        sol = np.linalg.lstsq(M_sub, ones_m, rcond=None)[0]
                    except:
                        sol = np.zeros(r)
                    for j in range(m):
                        U_new[i*m+j, :] = sol
            U = U_new
            
            # Solve for V: for each (i,k), N[j,c] = U[i*m+j,c]*W[k*n+i,c]; solve N x = ones(m)
            V_new = np.zeros_like(V)
            for i in range(n):
                for k in range(p):
                    N_sub = U[np.arange(m)*m + i, :] * W[k*n + i, :]
                    try:
                        sol = np.linalg.lstsq(N_sub, ones_m, rcond=None)[0]
                    except:
                        sol = np.zeros(r)
                    for j in range(m):
                        V_new[j*p+k, :] = sol
            V = V_new
            
            # Solve for W: for each (i,k), P[j,c] = U[i*m+j,c]*V[j*p+k,c]; solve P x = ones(m)
            W_new = np.zeros_like(W)
            for i in range(n):
                for k in range(p):
                    P_sub = U[np.arange(m)*m + i, :] * V[np.arange(m)*p + k, :]
                    try:
                        sol = np.linalg.lstsq(P_sub, ones_m, rcond=None)[0]
                    except:
                        sol = np.zeros(r)
                    for j in range(m):
                        W_new[k*n+i, :] = sol
            W = W_new
            
            # Check convergence (vectorized)
            max_err = verify_fast(U, V, W, r)
            if max_err < tol:
                break
        
        return U, V, W
    
    # Try to reduce rank using improved ALS
    if time.time() < deadline - 10:
        # Try ranks from 54 down to some minimum
        for r in range(rank_best - 1, max(nm, mp, pn) - 1, -1):
            if time.time() > deadline - 5:
                break
            
            found = False
            # More restarts for higher ranks, fewer for lower
            max_restarts = 25 if r >= 50 else 15
            
            for trial in range(max_restarts):
                if time.time() > deadline - 5:
                    break
                
                np.random.seed(trial * 1000 + r)
                
                # Initialize near the rank-55 solution with varying perturbation
                if r <= rank_best:
                    noise_scale = 0.05 + 0.03 * (trial % 5)
                    U_c = U_best[:, :r].copy() + noise_scale * np.random.randn(nm, r)
                    V_c = V_best[:, :r].copy() + noise_scale * np.random.randn(mp, r)
                    W_c = W_best[:, :r].copy() + noise_scale * np.random.randn(pn, r)
                else:
                    U_c = np.random.randn(nm, r) * 0.5
                    V_c = np.random.randn(mp, r) * 0.5
                    W_c = np.random.randn(pn, r) * 0.5
                
                # Run improved ALS
                U_c, V_c, W_c = block_als(U_c, V_c, W_c, r, max_iters=80, tol=1e-13)
                
                # Check convergence
                max_err = verify_fast(U_c, V_c, W_c, r)
                
                if max_err < 1e-8:
                    # Try to round to half-integers
                    U_r = np.round(U_c * 2) / 2
                    V_r = np.round(V_c * 2) / 2
                    W_r = np.round(W_c * 2) / 2
                    
                    max_err2 = verify_fast(U_r, V_r, W_r, r)
                    
                    if max_err2 < 1e-10:
                        U_best, V_best, W_best, rank_best = U_r, V_r, W_r, r
                        found = True
                        break
                    else:
                        # Use unrounded if close enough
                        if max_err < 1e-10:
                            U_best, V_best, W_best, rank_best = U_c, V_c, W_c, r
                            found = True
                            break
            
            if found:
                break
    
    # Final verification
    final_err = verify_fast(U_best, V_best, W_best, rank_best)
    if final_err > 1e-8:
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
        U_best, V_best, W_best = U, V, W
        rank_best = rank
    else:
        # Polish pass with tighter tolerance if we have time
        if time.time() < deadline - 10:
            U_pol, V_pol, W_pol = block_als(U_best, V_best, W_best, rank_best, max_iters=50, tol=1e-15)
            pol_err = verify_fast(U_pol, V_pol, W_pol, rank_best)
            if pol_err < 1e-12:
                U_best, V_best, W_best = U_pol, V_pol, W_pol
    
    return {"U": U_best.tolist(), "V": V_best.tolist(), "W": W_best.tolist()}
