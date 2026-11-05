import numpy as np
import time

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 3, 6
    deadline = time.time() + 140  # safety margin before 150s kill
    
    # Schoolbook rank 54 as fallback
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W
    
    best_U, best_V, best_W = schoolbook()
    best_rank = n * m * p
    
    # Target tensor: T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    # Try to find lower rank decomposition via alternating minimization
    # with structured initialization
    
    def verify(U, V, W, tol=1e-6):
        """Check if U, V, W exactly decompose T"""
        # Reconstruct tensor: sum_r U[:,r] outer V[:,r] outer W[:,r]
        r = U.shape[1]
        T_recon = np.zeros((n*m, m*p, p*n))
        for rr in range(r):
            T_recon += np.einsum('i,j,k->ijk', U[:, rr], V[:, rr], W[:, rr])
        return np.allclose(T_recon, T, atol=tol)
    
    def try_rank(r, init_type='random', seed=0):
        """Try to find a rank-r decomposition via alternating minimization"""
        np.random.seed(seed)
        
        if init_type == 'structured':
            # Initialize using field extension structure
            # 3x3x6 = 2 copies of 3x3x3 with sharing
            # Start with schoolbook 54 and zero out some columns
            U, V, W = schoolbook()
            if r < 54:
                # Keep first r columns
                U = U[:, :r]
                V = V[:, :r]
                W = W[:, :r]
        else:
            U = np.random.randn(n * m, r)
            V = np.random.randn(m * p, r)
            W = np.random.randn(p * n, r)
        
        # Alternating minimization
        for iteration in range(500):
            if time.time() > deadline:
                return None
            
            # Update U: for each column r, U[:,r] = (T * V[:,r]) / W[:,r]
            # T * V[:,r] gives a (n*m, p*n) matrix
            # We want U[:,r] such that sum_k W[k,r] * U[i,r] * V[j,r] = T[i,j,k]
            # Actually: T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
            # For fixed r: U[i,r] * V[j,r] * W[k,r] = T[i,j,k]
            # This is a rank-1 approximation problem
            
            # Better approach: for each r, solve for U[:,r] given V[:,r] and W[:,r]
            # T[i,j,k] = sum_s U[i,s]*V[j,s]*W[k,s]
            # Let M_r[i,k] = sum_{s != r} U[i,s]*V[j,s]*W[k,s]  (residual without r)
            # Then U[i,r]*V[j,r]*W[k,r] = T[i,j,k] - M_r[i,k]
            
            # Simpler: use gradient descent on the loss
            # Loss = ||T - sum_r U_r outer V_r outer W_r||^2
            
            # For efficiency, use the fact that we can update one factor at a time
            # Fix V, W and update U:
            # T[i,j,k] = sum_r U[i,r] * (V[j,r]*W[k,r])
            # For each r: U[:,r] = argmin ||T - sum_s U_s * (V_s outer W_s)||
            # This is a linear system: for each r, 
            # sum_i U[i,r] * V[j,r] * W[k,r] contributes to T[i,j,k]
            
            # Let's use a simpler iterative approach:
            # Update U: U[:,r] = (sum_{j,k} T[:,:,k] * V[:,r] * W[:,r]) / (sum_{j,k} V[:,r]^2 * W[:,r]^2)
            # This isn't quite right. Let me use proper alternating least squares.
            
            # For each r, the contribution is U[:,r] outer (V[:,r] * W[:,r])
            # Wait, that's not right either. The tensor is:
            # T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
            
            # So for fixed r, the "slice" is U[:,r] * V[:,r] * W[:,r] (element-wise in r)
            # To update U[:,r], fix all other terms:
            # residual[i,j,k] = T[i,j,k] - sum_{s != r} U[i,s]*V[j,s]*W[k,s]
            # We want: U[i,r]*V[j,r]*W[k,r] = residual[i,j,k]
            # This is a rank-1 tensor fitting problem for the 3D array residual.
            
            # Use SVD-based approach for the 3D tensor:
            # Flatten residual into a matrix and use SVD
            # Actually, let's just use coordinate descent with small steps.
            
            # Gradient-based update
            loss = 0.0
            for rr in range(r):
                # Compute residual excluding this term
                T_rec = np.zeros((n*m, m*p, p*n))
                for ss in range(r):
                    if ss == rr:
                        continue
                    T_rec += np.einsum('i,j,k->ijk', U[:, ss], V[:, ss], W[:, ss])
                residual = T - T_rec
                
                # Update U[:,rr]: U[i,rr] = (sum_{j,k} residual[i,j,k] * V[j,rr] * W[k,rr]) / (sum_{j,k} V[j,rr]^2 * W[k,rr]^2)
                num_U = np.einsum('ijk,j,k->i', residual, V[:, rr], W[:, rr])
                den_U = np.sum(V[:, rr]**2) * np.sum(W[:, rr]**2)
                if den_U > 1e-10:
                    U[:, rr] = num_U / den_U
                
                # Recompute residual with updated U
                T_rec = np.zeros((n*m, m*p, p*n))
                for ss in range(r):
                    if ss == rr:
                        continue
                    T_rec += np.einsum('i,j,k->ijk', U[:, ss], V[:, ss], W[:, ss])
                residual = T - T_rec
                
                # Update V[:,rr]
                num_V = np.einsum('ijk,i,k->j', residual, U[:, rr], W[:, rr])
                den_V = np.sum(U[:, rr]**2) * np.sum(W[:, rr]**2)
                if den_V > 1e-10:
                    V[:, rr] = num_V / den_V
                
                # Recompute residual with updated V
                T_rec = np.zeros((n*m, m*p, p*n))
                for ss in range(r):
                    if ss == rr:
                        continue
                    T_rec += np.einsum('i,j,k->ijk', U[:, ss], V[:, ss], W[:, ss])
                residual = T - T_rec
                
                # Update W[:,rr]
                num_W = np.einsum('ijk,i,j->k', residual, U[:, rr], V[:, rr])
                den_W = np.sum(U[:, rr]**2) * np.sum(V[:, rr]**2)
                if den_W > 1e-10:
                    W[:, rr] = num_W / den_W
                
                if time.time() > deadline:
                    return None
            
            # Check convergence
            T_recon = np.zeros((n*m, m*p, p*n))
            for rr in range(r):
                T_recon += np.einsum('i,j,k->ijk', U[:, rr], V[:, rr], W[:, rr])
            loss = np.linalg.norm(T - T_recon)
            
            if loss < 1e-4:
                # Try to round to exact values
                if verify(U, V, W, tol=1e-4):
                    return U, V, W
            
            if loss < 1e-8:
                if verify(U, V, W, tol=1e-8):
                    return U, V, W
        
        return None
    
    # Try various ranks below 54
    for r in range(30, 54):
        if time.time() > deadline:
            break
        
        for seed in range(3):
            if time.time() > deadline:
                break
            
            result = try_rank(r, init_type='random', seed=seed)
            if result is not None:
                U, V, W = result
                if verify(U, V, W, tol=1e-6):
                    if r < best_rank:
                        best_U, best_V, best_W = U, V, W
                        best_rank = r
                    break
            if time.time() > deadline:
                break
    
    # Also try structured initialization
    for r in range(30, 54):
        if time.time() > deadline:
            break
        
        result = try_rank(r, init_type='structured', seed=0)
        if result is not None:
            U, V, W = result
            if verify(U, V, W, tol=1e-6):
                if r < best_rank:
                    best_U, best_V, best_W = U, V, W
                    best_rank = r
                break
    
    # Return best result
    if best_rank < n * m * p:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    else:
        U, V, W = schoolbook()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
