import numpy as np
import time

def construct():
    n, m, p = 2, 3, 9
    
    # Build the target tensor
    # T[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i, else 0
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Try to find rank-r decomposition
    # Use random initialization and gradient descent or similar
    
    best_result = None
    
    # Start with schoolbook
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n*m, rank_schoolbook))
    V_sb = np.zeros((m*p, rank_schoolbook))
    W_sb = np.zeros((p*n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    def to_list(mat):
        return [[float(x) for x in row] for row in mat]
    
    best_rank = rank_schoolbook
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    deadline = time.time() + 100  # 100 second budget
    
    for r in range(33, rank_schoolbook):  # Try ranks from 33 to 53
        if time.time() > deadline - 5:
            break
        if r >= best_rank:
            continue
            
        # Try multiple random initializations
        for trial in range(50):
            if time.time() > deadline - 5:
                break
                
            np.random.seed(42 + r * 1000 + trial)
            U = np.random.randn(n*m, r)
            V = np.random.randn(m*p, r)
            W = np.random.randn(p*n, r)
            
            # Normalize
            for col_idx in range(r):
                norm = np.sqrt(np.sum(U[:, col_idx]**2) + np.sum(V[:, col_idx]**2) + np.sum(W[:, col_idx]**2))
                if norm > 1e-10:
                    U[:, col_idx] /= norm
                    V[:, col_idx] /= norm
                    W[:, col_idx] /= norm
            
            # Use gradient descent to minimize ||T - sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r]||^2
            # This is expensive, so use a smarter approach
            
            # Alternative: use the fact that we can fix one matrix and solve for others
            # Or use ALS (Alternating Least Squares)
            
            success = False
            for iteration in range(200):
                if time.time() > deadline - 5:
                    break
                    
                # Update U: for each a, U[a,:] should satisfy
                # sum_r U[a,r] * (V[:,r] ⊙ W[:,r])[b,c] = T[a,b,c] for all b,c
                # This is a linear system in U[a,:]
                
                # Build the system: for each a
                for a in range(n*m):
                    # T[a,:,:] is a (m*p) x (p*n) matrix
                    # We need U[a,:] such that sum_r U[a,r] * V[:,r] ⊙ W[:,r] approximates T[a,:,:]
                    # Let M_a[b,c] = sum_r V[b,r] * W[c,r]
                    # Then we need U[a,:] * M_a ≈ T[a,:,:]
                    # This is: for each (b,c), sum_r U[a,r] * V[b,r] * W[c,r] = T[a,b,c]
                    
                    # Reshape: let M_a be (m*p*p*n) x r matrix where M_a[b*(p*n)+c, r] = V[b,r]*W[c,r]
                    # Then U[a,:] * M_a.flatten() = T[a,:,:].flatten()
                    
                    M_a = np.outer(V.flatten(), W.flatten())  # This is wrong
                    
                    # Actually, M_a[b*(p*n)+c, r] = V[b,r] * W[c,r]
                    M_a = np.zeros((m*p*p*n, r))
                    for b in range(m*p):
                        for c in range(p*n):
                            M_a[b*(p*n)+c, :] = V[b, :] * W[c, :]
                    
                    target = T[a, :, :].flatten()
                    
                    # Solve least squares: U[a,:] @ M_a = target
                    # This is M_a.T @ U[a,:] = target
                    try:
                        U[a, :], residuals, rank_sol, sv = np.linalg.lstsq(M_a.T, target, rcond=None)
                    except:
                        pass
                
                # Similarly update V and W
                for b in range(m*p):
                    M_b = np.zeros((n*m*p*n, r))
                    for a in range(n*m):
                        for c in range(p*n):
                            M_b[a*(p*n)+c, :] = U[a, :] * W[c, :]
                    target = T[:, b, :].reshape(n*m, p*n).flatten()
                    try:
                        V[b, :], residuals, rank_sol, sv = np.linalg.lstsq(M_b.T, target, rcond=None)
                    except:
                        pass
                
                for c in range(p*n):
                    M_c = np.zeros((n*m*m*p, r))
                    for a in range(n*m):
                        for b in range(m*p):
                            M_c[a*(m*p)+b, :] = U[a, :] * V[b, :]
                    target = T[:, :, c].reshape(n*m, m*p).flatten()
                    try:
                        W[c, :], residuals, rank_sol, sv = np.linalg.lstsq(M_c.T, target, rcond=None)
                    except:
                        pass
                
                # Check convergence
                # Compute the error
                error = 0.0
                for a in range(n*m):
                    for b in range(m*p):
                        for c in range(p*n):
                            val = np.dot(U[a, :], V[b, :] * W[c, :])
                            error += (val - T[a, b, c])**2
                
                if error < 1e-12:
                    success = True
                    break
                elif error > 1e6:
                    break  # Diverged
            
            if success:
                best_rank = r
                best_U, best_V, best_W = U, V, W
                break  # Found at this rank, no need to try higher r
        
        if best_rank == r:
            break  # Found at this rank
    
    # Convert to lists
    U_list = to_list(best_U)
    V_list = to_list(best_V)
    W_list = to_list(best_W)
    
    return {"U": U_list, "V": V_list, "W": W_list}
