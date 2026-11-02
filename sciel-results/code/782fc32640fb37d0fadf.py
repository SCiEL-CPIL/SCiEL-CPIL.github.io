import numpy as np
import time

def construct():
    n, m, p = 3, 3, 4
    start_time = time.time()
    deadline = 140.0  # 10s safety margin
    
    # Build the multiplication tensor
    # T[a, b, c] where a = i*m+j, b = j*p+k, c = k*n+i
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Try ranks from low to high
    best_result = None
    
    for rank in range(30, 37):
        if time.time() > deadline - 20:
            break
        
        # Try multiple random initializations
        for trial in range(200):
            if time.time() > deadline - 5:
                break
            
            # Initialize
            U = np.random.randn(n*m, rank)
            V = np.random.randn(m*p, rank)
            W = np.random.randn(p*n, rank)
            
            # ALS iterations
            for it in range(200):
                if time.time() > deadline - 5:
                    break
                
                # Update U: for each a, U[a,:] = argmin ||T[:,a,:] - V*W^T * U[a,:]||
                # Reshape: T is (nm, mp, pn). For fixed a, we need T[a,:,:] = sum_r U[a,r] V[:,r] W[r,:]
                # T[a] (mp, pn) = U[a] (rank,) @ (V (mp,rank) @ W.T (rank,pn))
                # So: T[a] = U[a] @ (V @ W.T)
                # Minimize over U[a]: ||T[a] - U[a] @ (V@W.T)||_F
                # Solution: U[a] = T[a] @ (V@W.T)^+ 
                
                VW = V @ W.T  # (mp, pn)
                # U[a] @ VW = T[a]
                # U[a] = T[a] @ VW^+ (pseudo-inverse)
                for a in range(n*m):
                    U[a] = np.linalg.lstsq(VW, T[a].flatten(), rcond=None)[0]
                
                # Update V: for each b, V[b,:] = argmin ||T[:,:,b] - U*W[:,b]||
                # T[:,:,b] (nm, pn) = sum_r U[:,r] W[r,b] = U @ diag(W[:,b]) ... 
                # Actually: T[:,:,b] = U (nm, rank) @ W[:,b] (rank,) reshaped...
                # T[:,:,b] = sum_r U[:,r] * W[r,b]
                # This is: T[:,:,b] = U @ W[:,b] where W[:,b] is (rank,)
                # So: T[:,:,b] (nm, pn) = U (nm,rank) @ (W[:,b] (rank,) broadcast)
                # Hmm, that gives (nm, rank) @ (rank,) = (nm,), not (nm, pn)
                
                # Let me reconsider. T[a, b, c] = sum_r U[a,r] V[b,r] W[c,r]
                # For fixed b: T[:, b, :] (nm, pn) = sum_r U[:,r] * V[b,r] * W[:,r]
                # = U (nm, rank) @ diag(V[b,:]) @ W.T (rank, pn)
                # So: T[:,b,:] = U @ diag(V[b]) @ W.T
                # Minimize over V[b]: ||T[:,b,:] - U @ diag(V[b]) @ W.T||_F
                # Let M = U (nm, rank), N = W.T (rank, pn)
                # T[:,b,:] = M @ diag(V[b]) @ N
                # This is: element (a,c) of T[:,b,:] = sum_r M[a,r] V[b,r] N[r,c]
                # = sum_r V[b,r] * M[a,r] * N[r,c]
                # So it's a linear system in V[b]:
                # T[:,b,:].flatten() = (N.T * M.T).T.flatten() @ V[b] ... 
                # Let's vectorize: T[:,b,:].flatten() (nm*pn,) 
                # = sum_r V[b,r] * (M[:,r] * N[r,:]).flatten()
                # So: T[:,b,:].flatten() = X @ V[b] where X[c*nm+a, r] = M[a,r]*N[r,c]
                
                UW = U * W.T  # (nm, rank, pn) -> need to reshape
                # X (nm*pn, rank): X[c*nm+a, r] = U[a,r] * W[c,r]
                X = (U[:, :, None] * W[None, :, :]).reshape(n*m*p*n, rank)
                for b in range(m*p):
                    rhs = T[:, b, :].flatten()
                    V[b] = np.linalg.lstsq(X, rhs, rcond=None)[0]
                
                # Update W: for each c, W[c,:] = argmin ||T[:,:,c] - U*V[:,c]||
                # T[:,:,c] (nm, mp) = sum_r U[:,r] V[:,r] W[c,r]
                # = U (nm,rank) @ diag(W[c]) @ V.T (rank, mp)
                # Similar to V update
                UV = (U[:, :, None] * V[None, :, :]).reshape(n*m*m*p, rank)
                # Wait, T[:,:,c] has shape (nm, mp). 
                # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
                # For fixed c: T[:,:,c] (nm, mp) = sum_r U[:,r] * V[:,r] * W[c,r]
                # Element (a,b): sum_r U[a,r] V[b,r] W[c,r] = sum_r W[c,r] * U[a,r] * V[b,r]
                # So: T[:,:,c].flatten() (nm*mp,) = Y @ W[c] where Y[b*nm+a, r] = U[a,r]*V[b,r]
                Y = (U[:, :, None] * V[None, :, :]).reshape(n*m*m*p, rank)
                for c in range(p*n):
                    rhs = T[:, :, c].flatten()
                    W[c] = np.linalg.lstsq(Y, rhs, rcond=None)[0]
                
                # Check convergence
                if it > 0:
                    recon = np.einsum('ar,br,cr->abc', U, V, W)
                    err = np.linalg.norm(recon - T) / np.linalg.norm(T)
                    if err < 1e-10:
                        break
            
            # Check if we have a good solution
            recon = np.einsum('ar,br,cr->abc', U, V, W)
            err = np.linalg.norm(recon - T) / np.linalg.norm(T)
            
            if err < 1e-8:
                # Try to round to integers/half-integers
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                
                recon_r = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
                err_r = np.linalg.norm(recon_r - T) / np.linalg.norm(T)
                
                if err_r < 1e-10:
                    # Found a valid decomposition!
                    best_result = (rank, U_r, V_r, W_r)
                    break
        
        if best_result:
            break
    
    if best_result:
        rank, U, V, W = best_result
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
                    
        return {"U": U, "V": V, "W": W}
