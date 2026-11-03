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
    
    # Verify a decomposition (vectorized)
    def verify(U, V, W, r, tol=1e-10):
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m + j, :], V[j*p + k, :] * W[k*n + i, :])
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
        return max_err

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
    
    # Improved ALS with correct linear solves
    def block_als(U, V, W, r, max_iters=100, tol=1e-12):
        """ALS with correct block-wise linear solves."""
        prev_err = float('inf')
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # 1. Solve for U
            # Fix V, W. For each i, j:
            # sum_k sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1
            # A[i,j] is p x r matrix: A[k, r'] = V[j,k,r'] * W[k,i,r']
            # Solve A[i,j] @ U[i,j,:]^T = ones(p)
            U_new = np.zeros_like(U)
            for i in range(n):
                for j in range(m):
                    A = np.zeros((p, r))
                    for k in range(p):
                        for r_idx in range(r):
                            A[k, r_idx] = V[j*p + k, r_idx] * W[k*n + i, r_idx]
                    try:
                        # Solve least squares
                        sol, residuals, rank_mat, sv = np.linalg.lstsq(A, np.ones(p), rcond=None)
                        U_new[i*m + j, :] = sol
                    except:
                        U_new[i*m + j, :] = U[i*m + j, :]
            U = U_new
            
            # 2. Solve for V
            # Fix U, W. For each j, k:
            # sum_i sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1
            # B[j,k] is n x r matrix: B[i, r'] = U[i,j,r'] * W[k,i,r']
            # Solve B[j,k] @ V[j,k,:]^T = ones(n)
            V_new = np.zeros_like(V)
            for j in range(m):
                for k in range(p):
                    B = np.zeros((n, r))
                    for i in range(n):
                        for r_idx in range(r):
                            B[i, r_idx] = U[i*m + j, r_idx] * W[k*n + i, r_idx]
                    try:
                        sol, residuals, rank_mat, sv = np.linalg.lstsq(B, np.ones(n), rcond=None)
                        V_new[j*p + k, :] = sol
                    except:
                        V_new[j*p + k, :] = V[j*p + k, :]
            V = V_new
            
            # 3. Solve for W
            # Fix U, V. For each k, i:
            # sum_j sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1
            # C[k,i] is m x r matrix: C[j, r'] = U[i,j,r'] * V[j,k,r']
            # Solve C[k,i] @ W[k,i,:]^T = ones(m)
            W_new = np.zeros_like(W)
            for k in range(p):
                for i in range(n):
                    C = np.zeros((m, r))
                    for j in range(m):
                        for r_idx in range(r):
                            C[j, r_idx] = U[i*m + j, r_idx] * V[j*p + k, r_idx]
                    try:
                        sol, residuals, rank_mat, sv = np.linalg.lstsq(C, np.ones(m), rcond=None)
                        W_new[k*n + i, :] = sol
                    except:
                        W_new[k*n + i, :] = W[k*n + i, :]
            W = W_new
            
            # Check convergence
            max_err = verify(U, V, W, r)
            if max_err < tol:
                break
            if max_err > prev_err * 1.1:
                # Diverging or stuck, break early
                break
            prev_err = max_err
            
        return U, V, W
    
    # Try to reduce rank using improved ALS
    if time.time() < deadline - 10:
        # Try ranks from 54 down to some minimum
        for r in range(rank_best - 1, max(nm, mp, pn) - 1, -1):
            if time.time() > deadline - 5:
                break
            
            found = False
            # More restarts for higher ranks, fewer for lower
            max_restarts = 20 if r >= 50 else 12
            
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
                U_c, V_c, W_c = block_als(U_c, V_c, W_c, r, max_iters=50, tol=1e-12)
                
                # Check convergence
                max_err = verify(U_c, V_c, W_c, r)
                
                if max_err < 1e-8:
                    # Try to round to half-integers
                    U_r = np.round(U_c * 2) / 2
                    V_r = np.round(V_c * 2) / 2
                    W_r = np.round(W_c * 2) / 2
                    
                    max_err2 = verify(U_r, V_r, W_r, r)
                    
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
    final_err = verify(U_best, V_best, W_best, rank_best)
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
            U_pol, V_pol, W_pol = block_als(U_best, V_best, W_best, rank_best, max_iters=30, tol=1e-14)
            pol_err = verify(U_pol, V_pol, W_pol, rank_best)
            if pol_err < 1e-10:
                U_best, V_best, W_best = U_pol, V_pol, W_pol
    
    return {"U": U_best.tolist(), "V": V_best.tolist(), "W": W_best.tolist()}
