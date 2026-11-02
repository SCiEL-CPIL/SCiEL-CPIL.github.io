import numpy as np
import time

DIMENSIONS = [3, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Build the tensor T
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Schoolbook as fallback
    def schoolbook():
        R = n*m*p
        U = np.zeros((n*m, R))
        V = np.zeros((m*p, R))
        W = np.zeros((p*n, R))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W
    
    best_U, best_V, best_W = schoolbook()
    best_rank = n*m*p
    
    # Try to find a lower rank decomposition
    # Use alternating least squares
    for R in range(20, n*m*p):
        if time.time() - start_time > 100:
            break
        
        for trial in range(10):
            # Initialize
            U = np.random.randn(n*m, R)
            V = np.random.randn(m*p, R)
            W = np.random.randn(p*n, R)
            
            # Alternating minimization
            for it in range(100):
                # Fix V and W, solve for U
                # T[a,b,c] = Σ_r U[a,r] V[b,r] W[c,r]
                # For fixed a: T[a,:,:] = Σ_r U[a,r] (V[:,r] ⊗ W[:,r])
                # This is a linear system in U[a,:]
                
                # Matricize: for each a, T[a] is a matrix of size (m*p) x (p*n)
                # T[a] = Σ_r U[a,r] * (V[:,r] * W[:,r]^T)
                # So T[a] = U[a,:] @ (V * W^T) where (V*W^T) has shape (R, m*p, p*n)
                
                # Let M = V * W^T, shape (R, m*p, p*n)
                # Then T[a] = Σ_r U[a,r] M[r]
                # Flatten: T[a].flatten() = U[a,:] @ M.flatten(axis=1).T
                
                M = V[:, :, None] * W[:, None, :]  # (R, m*p, p*n)
                M_flat = M.reshape(R, -1)  # (R, m*p*p*n)
                
                # For each a: T[a].flatten() = U[a,:] @ M_flat.T
                # So U[a,:] = T[a].flatten() @ pinv(M_flat.T)
                
                T_flat = T.reshape(n*m, -1)  # (n*m, m*p*p*n)
                
                # Solve least squares
                try:
                    U_new = T_flat @ np.linalg.pinv(M_flat.T)
                except:
                    break
                
                U = U_new
                
                # Similarly for V
                # T[a,b,c] = Σ_r U[a,r] V[b,r] W[c,r]
                # For fixed b: T[:,b,:] = Σ_r V[b,r] (U[:,r] * W[:,r]^T)
                M = U[:, :, None] * W[:, None, :]  # (R, n*m, p*n)
                M_flat = M.reshape(R, -1)
                T_b = T.reshape(-1, m*p, p*n).transpose(1,0,2).reshape(m*p, -1)
                try:
                    V_new = T_b @ np.linalg.pinv(M_flat.T)
                except:
                    break
                V = V_new
                
                # For W
                M = U[:, :, None] * V[:, None, :]  # (R, n*m, m*p)
                M_flat = M.reshape(R, -1)
                T_c = T.reshape(n*m, m*p, -1).transpose(2,0,1).reshape(p*n, -1)
                try:
                    W_new = T_c @ np.linalg.pinv(M_flat.T)
                except:
                    break
                W = W_new
            
            # Round to nearest half-integers
            U_r = np.round(U * 2) / 2
            V_r = np.round(V * 2) / 2
            W_r = np.round(W * 2) / 2
            
            # Check if exact
            T_check = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
            if np.allclose(T_check, T, atol=1e-10):
                best_rank = R
                best_U, best_V, best_W = U_r, V_r, W_r
                break
        
        if best_rank == R:
            break
    
    # Convert to lists
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
