import numpy as np
import time

DIMENSIONS = [3, 3, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    # n=3 (rows of A, cols of C), m=3 (cols of A, rows of B), p=7 (cols of B, cols of C)
    # U: (n*m, rank) - represents φ_c
    # V: (m*p, rank) - represents ψ_c
    # W: (p*n, rank) - represents z_c (flattened)
    
    # Target: C[i,k] = Σ_j A[i,j] * B[j,k]
    # In tensor form: Σ_c φ_c(A) * ψ_c(B) * z_c[i,k] = Σ_j A[i,j] * B[j,k]
    
    # φ_c is a 3x3 matrix (flattened to 9), ψ_c is a 3x7 matrix (flattened to 21), z_c is a 3x7 matrix (flattened to 21)
    
    # The equation is:
    # Σ_c (Σ_{i,j} U[c, i*m+j] * A[i,j]) * (Σ_{j,k} V[c, j*p+k] * B[j,k]) * W[c, k*n+i] = Σ_j A[i,j] * B[j,k]
    
    # This is a tensor equation. To solve it, we can use the fact that it must hold for all A, B.
    # This means the tensor T[i,j,k] = 1 must be expressible as:
    # Σ_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1
    # where U[i,j,c] = U[c, i*m+j], V[j,k,c] = V[c, j*p+k], W[k,i,c] = W[c, k*n+i]
    
    # So we need: Σ_c U_flat[i*m+j, c] * V_flat[j*p+k, c] * W_flat[k*n+i, c] = 1 for all i,j,k
    
    # This is the same as the original formulation!
    
    T = np.ones((n, m, p))
    T_flat = T.flatten()
    
    # Schoolbook fallback
    rank_school = n * m * p
    U = np.zeros((n*m, rank_school))
    V = np.zeros((m*p, rank_school))
    W = np.zeros((p*n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    
    best = {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
    best_rank = rank_school
    
    # Numerical search
    for rank in range(20, rank_school):
        if time.time() > deadline:
            break
        
        found = False
        for trial in range(10):
            if time.time() > deadline:
                break
            
            np.random.seed(rank * 1000 + trial)
            U_c = np.random.randn(n*m, rank)
            V_c = np.random.randn(m*p, rank)
            W_c = np.random.randn(p*n, rank)
            
            # Alternating least squares
            for it in range(200):
                if time.time() > deadline:
                    break
                
                # Update U: for each c, U[:,c] is a vector of length n*m
                # The equation is: Σ_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
                # For fixed V, W, this is linear in U.
                # Let's rewrite: for each (i,j,k): Σ_c U[i*m+j, c] * (V[j*p+k, c] * W[k*n+i, c]) = 1
                # For fixed (i,j): Σ_c U[i*m+j, c] * B_c[i,j,k] = 1 where B_c[i,j,k] = V[j*p+k, c] * W[k*n+i, c]
                # But B_c depends on k, so for fixed (i,j), we have p equations:
                # Σ_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for k=0,...,p-1
                # This is: [V[j*p+0,c]*W[0*n+i,c], ..., V[j*p+p-1,c]*W[(p-1)*n+i,c]] * U[i*m+j,:]^T = [1,1,...,1]
                # So: M_U[i,j] * U[i*m+j,:]^T = ones(p)
                # where M_U[i,j] is a p x rank matrix with M_U[i,j][k,c] = V[j*p+k, c] * W[k*n+i, c]
                
                for i in range(n):
                    for j in range(m):
                        M_mat = np.zeros((p, rank))
                        for k in range(p):
                            for c in range(rank):
                                M_mat[k, c] = V_c[j*p+k, c] * W_c[k*n+i, c]
                        rhs = np.ones(p)
                        try:
                            U_c[i*m+j, :] = np.linalg.lstsq(M_mat, rhs, rcond=None)[0]
                        except:
                            pass
                
                # Update V: for each (j,k), V[j*p+k, c]
                # Σ_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
                # For fixed (j,k): Σ_c V[j*p+k, c] * (U[i*m+j, c] * W[k*n+i, c]) = 1 for all i
                # This is: [U[0*m+j,c]*W[k*n+0,c], ..., U[(n-1)*m+j,c]*W[k*n+(n-1),c]] * V[j*p+k,:]^T = [1,1,...,1]
                # So: M_V[j,k] * V[j*p+k,:]^T = ones(n)
                
                for j in range(m):
                    for k in range(p):
                        M_mat = np.zeros((n, rank))
                        for i in range(n):
                            for c in range(rank):
                                M_mat[i, c] = U_c[i*m+j, c] * W_c[k*n+i, c]
                        rhs = np.ones(n)
                        try:
                            V_c[j*p+k, :] = np.linalg.lstsq(M_mat, rhs, rcond=None)[0]
                        except:
                            pass
                
                # Update W: for each (k,i), W[k*n+i, c]
                # Σ_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
                # For fixed (k,i): Σ_c W[k*n+i, c] * (U[i*m+j, c] * V[j*p+k, c]) = 1 for all j
                # This is: [U[i*m+0,c]*V[0*p+k,c], ..., U[i*m+(m-1),c]*V[(m-1)*p+k,c]] * W[k*n+i,:]^T = [1,1,...,1]
                # So: M_W[k,i] * W[k*n+i,:]^T = ones(m)
                
                for k in range(p):
                    for i in range(n):
                        M_mat = np.zeros((m, rank))
                        for j in range(m):
                            for c in range(rank):
                                M_mat[j, c] = U_c[i*m+j, c] * V_c[j*p+k, c]
                        rhs = np.ones(m)
                        try:
                            W_c[k*n+i, :] = np.linalg.lstsq(M_mat, rhs, rcond=None)[0]
                        except:
                            pass
            
            # Check error
            T_recon = np.zeros((n, m, p))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T_recon[i,j,k] = np.sum(U_c[i*m+j, :] * V_c[j*p+k, :] * W_c[k*n+i, :])
            error = np.linalg.norm(T_flat - T_recon.flatten())
            
            if error < 1e-6:
                # Try rounding
                U_r = np.round(U_c * 2) / 2
                V_r = np.round(V_c * 2) / 2
                W_r = np.round(W_c * 2) / 2
                
                T_recon_r = np.zeros((n, m, p))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            T_recon_r[i,j,k] = np.sum(U_r[i*m+j, :] * V_r[j*p+k, :] * W_r[k*n+i, :])
                error_r = np.linalg.norm(T_flat - T_recon_r.flatten())
                
                if error_r < 1e-6:
                    best = {'U': U_r.tolist(), 'V': V_r.tolist(), 'W': W_r.tolist()}
                    best_rank = rank
                    found = True
                    break
                else:
                    best = {'U': U_c.tolist(), 'V': V_c.tolist(), 'W': W_c.tolist()}
                    best_rank = rank
                    found = True
                    break
        
        if found:
            break
    
    return best
