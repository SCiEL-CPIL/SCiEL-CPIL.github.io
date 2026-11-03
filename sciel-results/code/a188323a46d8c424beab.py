import numpy as np
import time

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    pn = p * n  # 12 (W dimension)
    nm = n * m  # 9  (U dimension)
    mp = m * p  # 12 (V dimension)
    
    # Build target: for each (i,j,k), we need sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    # And for all other (b,d,a) combinations, the sum should be 0.
    
    # Create sparse representation of constraints
    # Non-zero constraints: (a,b,d) = (k*n+i, i*m+j, j*p+k) for i in 0..2, j in 0..2, k in 0..3
    nz_a, nz_b, nz_d = [], [], []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                nz_a.append(k * n + i)
                nz_b.append(i * m + j)
                nz_d.append(j * p + k)
    nz_a = np.array(nz_a)
    nz_b = np.array(nz_b)
    nz_d = np.array(nz_d)
    
    # Precompute the full tensor T as a dense array for faster error checking
    T = np.zeros((pn, nm, mp))
    for idx in range(len(nz_a)):
        T[nz_a[idx], nz_b[idx], nz_d[idx]] = 1.0
    
    # Precompute the set of non-zero indices for quick lookup
    nz_set = set()
    for idx in range(len(nz_a)):
        nz_set.add((nz_a[idx], nz_b[idx], nz_d[idx]))
    
    best_U, best_V, best_W = None, None, None
    best_rank = n * m * p  # 36
    
    start_time = time.time()
    deadline = start_time + 140
    
    for r in range(18, best_rank + 1):
        if time.time() > deadline:
            break
        
        found = False
        for attempt in range(30):
            if time.time() > deadline:
                break
            
            U = np.random.randn(nm, r) * 0.1
            V = np.random.randn(mp, r) * 0.1
            W = np.random.randn(pn, r) * 0.1
            
            # ALS iterations
            for it in range(500):
                if time.time() > deadline:
                    break
                
                # Update U: for each b, U[b,:] = argmin sum_a,d (T[a,b,d] - sum_c U[b,c]V[d,c]W[a,c])^2
                # This is linear in U[b,:]: 
                # min U[b,:] sum_a,d (T[a,b,d] - U[b,:] @ (V[d,:] * W[a,:]))^2
                # = min U[b,:] || T_b - U[b,:] @ M_b ||^2 where M_b[c] = sum_a,d T[a,b,d]... 
                # Actually: for fixed b, the equation is:
                # T[a,b,d] = sum_c U[b,c] * V[d,c] * W[a,c]
                # Let G_b[c] = sum_a,d T[a,b,d] * V[d,c] * W[a,c]  -- no, this is wrong
                # 
                # The residual is: R[a,b,d] = T[a,b,d] - sum_c U[b,c]*V[d,c]*W[a,c]
                # d/dU[b,c] sum_{a,d} R[a,b,d]^2 = -2 * sum_{a,d} T[a,b,d]*V[d,c]*W[a,c] + 2*sum_{a,d} (sum_c' U[b,c']*V[d,c']*W[a,c]) * V[d,c]*W[a,c]
                #
                # This is: -2 * (T[:,b,:] * (V*W)) : e_c + 2 * (sum_c' U[b,c'] * (V[d,c']*W[a,c'])) * V[d,c]*W[a,c] summed over a,d
                #
                # Let me define: for each c, let H_c[a,d] = V[d,c]*W[a,c]
                # Then the equation is: T[:,b,:] = sum_c U[b,c] * H_c
                # This is a linear system: T_b (pn*mp,) = U[b,:] (r,) @ H_flat^T (r, pn*mp)
                # So U[b,:] = T_b @ pinv(H_flat^T)
                
                H_flat = np.zeros((r, pn * mp))
                for c in range(r):
                    H_flat[c, :] = (W[:, c][:, None] * V[:, c][None, :]).flatten()
                
                T_flat = np.zeros((nm, pn * mp))
                for idx in range(len(nz_a)):
                    T_flat[nz_b[idx], nz_a[idx] * mp + nz_d[idx]] = 1.0
                
                try:
                    H_pinv = np.linalg.pinv(H_flat.T)
                    U_new = T_flat @ H_pinv
                except:
                    break
                
                # Update V: T[a,b,d] = sum_c V[d,c] * U[b,c] * W[a,c]
                # For fixed d: T[a,b,d] = sum_c V[d,c] * (U[b,c]*W[a,c])
                # Let K_c[a,b] = U[b,c]*W[a,c]
                # T[:, :, d] = sum_c V[d,c] * K_c
                # V[d,:] = T_d @ pinv(K_flat^T)
                
                K_flat = np.zeros((r, pn * nm))
                for c in range(r):
                    K_flat[c, :] = (W[:, c][:, None] * U_new[:, c][None, :]).flatten()
                
                T_flat2 = np.zeros((mp, pn * nm))
                for idx in range(len(nz_a)):
                    T_flat2[nz_d[idx], nz_a[idx] * nm + nz_b[idx]] = 1.0
                
                try:
                    K_pinv = np.linalg.pinv(K_flat.T)
                    V_new = T_flat2 @ K_pinv
                except:
                    break
                
                # Update W: T[a,b,d] = sum_c W[a,c] * U[b,c] * V[d,c]
                # For fixed a: T[a,b,d] = sum_c W[a,c] * (U[b,c]*V[d,c])
                # Let L_c[b,d] = U[b,c]*V[d,c]
                # T[a,:,:] = sum_c W[a,c] * L_c
                # W[a,:] = T_a @ pinv(L_flat^T)
                
                L_flat = np.zeros((r, nm * mp))
                for c in range(r):
                    L_flat[c, :] = (U_new[:, c][:, None] * V_new[:, c][None, :]).flatten()
                
                T_flat3 = np.zeros((pn, nm * mp))
                for idx in range(len(nz_a)):
                    T_flat3[nz_a[idx], nz_b[idx] * mp + nz_d[idx]] = 1.0
                
                try:
                    L_pinv = np.linalg.pinv(L_flat.T)
                    W_new = T_flat3 @ L_pinv
                except:
                    break
                
                U, V, W = U_new, V_new, W_new
                
                # Check error using dense tensor
                # Compute the tensor product: T_approx[a,b,d] = sum_c U[b,c]*V[d,c]*W[a,c]
                # T_approx = np.einsum('bc,dc,ac->abd', U, V, W)
                T_approx = np.zeros((pn, nm, mp))
                for c in range(r):
                    T_approx += np.outer(np.outer(W[:, c], U[:, c]), V[:, c])
                
                err = np.max(np.abs(T_approx - T))
                
                if err < 1e-8:
                    found = True
                    break
            
            if found:
                best_U, best_V, best_W = U, V, W
                best_rank = r
                break
        
        if found:
            break
    
    if best_U is not None:
        return {
            "U": best_U.tolist(),
            "V": best_V.tolist(),
            "W": best_W.tolist()
        }
    else:
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
