DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110
    
    # Schoolbook fallback
    rank_school = n * m * p
    U_fb = np.zeros((n*m, rank_school))
    V_fb = np.zeros((m*p, rank_school))
    W_fb = np.zeros((p*n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i*m+j, col] = 1
                V_fb[j*p+k, col] = 1
                W_fb[k*n+i, col] = 1
                col += 1
    
    # Build target tensor
    T = np.zeros((n*m, m*p, p*n))
    for i_a in range(n):
        for j_a in range(m):
            for j_b in range(m):
                for k_b in range(p):
                    for i_c in range(n):
                        if j_a == j_b and i_a == i_c:
                            T[i_a*m+j_a, j_b*p+k_b, k_b*n+i_c] = 1.0
    
    best_rank = rank_school
    best_U, best_V, best_W = U_fb, V_fb, W_fb
    
    # Try to find lower rank using SVD-based approach
    # Reshape T to (n*m, m*p*p*n) and try low-rank factorization
    T_2d = T.reshape(n*m, m*p*p*n)
    
    for rank in range(40, 24, -1):
        if time.time() > deadline - 5:
            break
        
        # Try ALS with random init
        for trial in range(3):
            if time.time() > deadline - 5:
                break
            
            np.random.seed(rank*100+trial)
            U = np.random.randn(n*m, rank) * 0.5
            V = np.random.randn(m*p, rank) * 0.5
            W = np.random.randn(p*n, rank) * 0.5
            
            for it in range(20):
                if time.time() > deadline - 5:
                    break
                
                # Update U[i,:] by solving linear system
                for i in range(n*m):
                    # Build matrix: M[t, j*p*n+k] = V[j,t] * W[k,t]
                    M = np.outer(V, W).reshape(rank, m*p*p*n)
                    rhs = T[i,:,:].flatten()
                    try:
                        U[i,:] = np.linalg.lstsq(M, rhs, rcond=None)[0]
                    except:
                        break
                
                if time.time() > deadline - 5:
                    break
                
                # Update V[j,:]
                for j in range(m*p):
                    M = np.outer(U, W).reshape(rank, n*m*p*n)
                    rhs = T[:,j,:].transpose(1,0).flatten()
                    try:
                        V[j,:] = np.linalg.lstsq(M, rhs, rcond=None)[0]
                    except:
                        break
                
                if time.time() > deadline - 5:
                    break
                
                # Update W[k,:]
                for k in range(p*n):
                    M = np.outer(U, V).reshape(rank, n*m*m*p)
                    rhs = T[:,:,k].transpose(1,0).flatten()
                    try:
                        W[k,:] = np.linalg.lstsq(M, rhs, rcond=None)[0]
                    except:
                        break
                
                # Check error
                if time.time() > deadline - 5:
                    break
                recon = np.einsum('it,jt,kt->ijk', U, V, W)
                err = np.linalg.norm(recon - T)
                if err < 1e-4:
                    # Round to nearest half-integer and verify
                    U_r = np.round(U*2)/2
                    V_r = np.round(V*2)/2
                    W_r = np.round(W*2)/2
                    recon_r = np.einsum('it,jt,kt->ijk', U_r, V_r, W_r)
                    if np.linalg.norm(recon_r - T) < 1e-6:
                        best_rank = rank
                        best_U, best_V, best_W = U_r, V_r, W_r
                        break
            if best_rank == rank:
                break
    
    # Convert to lists
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
