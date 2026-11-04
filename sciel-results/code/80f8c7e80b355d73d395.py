import time
import numpy as np
import random

DIMENSIONS = [4, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115.0  # Safety margin
    
    # Initialize best result with schoolbook algorithm (rank = n*m*p)
    best_rank = n * m * p
    best_U = None
    best_V = None
    best_W = None
    
    # Construct schoolbook solution
    U = np.zeros((n * m, best_rank))
    V = np.zeros((m * p, best_rank))
    W = np.zeros((p * n, best_rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
                
    best_U = U.tolist()
    best_V = V.tolist()
    best_W = W.tolist()
    
    # Search for lower rank decompositions
    # Theoretical lower bound is max(n*m, m*p, p*n) = max(20, 30, 24) = 30
    # However, known bounds for 4x5x6 are not trivial. 
    # We will search from a plausible low rank upwards.
    
    # To make the search effective, we use a randomized approach.
    # We define the tensor T[i, j, k] which should be 1 if the indices correspond to a valid multiplication term.
    # Actually, the condition is: sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] = 1 for all i,j,k.
    # This is equivalent to saying the matrix multiplication tensor is decomposed.
    
    # We can formulate this as minimizing || sum_c u_c \otimes v_c \otimes w_c - T ||_F
    # where T is the tensor with T[i,j,k]=1 for all i,j,k.
    # Wait, the tensor for matrix multiplication A*B=C has entries T[i,j,k] = 1 if we are computing C[i,k] += A[i,j]B[j,k].
    # Yes, the tensor is all ones for the indices involved in the product.
    # Specifically, for each (i, j, k), the term A[i,j]B[j,k] contributes to C[i,k].
    # So the target tensor T has T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    
    T = np.ones((n, m, p))
    
    # Flattened indices for U, V, W
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    
    # We will try ranks from 30 to best_rank - 1
    # To save time, we might skip some ranks or use a coarse-to-fine search.
    # Given the time limit, we try a few random restarts for each rank.
    
    start_rank = 30
    # If time is tight, we might not check all ranks. 
    # Let's prioritize lower ranks first as they are more valuable.
    
    for r in range(start_rank, best_rank):
        if time.time() > deadline - 10.0:
            break
            
        # Try to find a solution of rank r
        # Use a simple iterative method:
        # Initialize U, V, W with small random values.
        # Iterate:
        #   Compute residual R = T - sum_c u_c \otimes v_c \otimes w_c
        #   Update one factor at a time to minimize residual (alternating least squares style)
        #   This is effectively a tensor train / CP decomposition approach.
        
        found = False
        max_iters = 50 # Limit iterations per restart to ensure speed
        restarts = 5   # Limit restarts per rank
        
        for restart in range(restarts):
            if time.time() > deadline - 10.0:
                break
                
            # Initialize
            U_curr = np.random.randn(n * m, r) * 0.1
            V_curr = np.random.randn(m * p, r) * 0.1
            W_curr = np.random.randn(p * n, r) * 0.1
            
            # Normalize to prevent explosion
            # We can normalize columns
            for c in range(r):
                norm_u = np.linalg.norm(U_curr[:, c])
                norm_v = np.linalg.norm(V_curr[:, c])
                norm_w = np.linalg.norm(W_curr[:, c])
                if norm_u > 0 and norm_v > 0 and norm_w > 0:
                    # Distribute scale
                    # Keep it simple, just check magnitude
                    pass
            
            for it in range(max_iters):
                if time.time() > deadline - 10.0:
                    break
                    
                # Update U
                # For each c, we want to minimize || T - sum u \otimes v \otimes w ||
                # This is equivalent to for each c, u_c should be close to 
                # (T - sum_{c'!=c} u_{c'} \otimes v_{c'} \otimes w_{c'}) contracted with v_c and w_c.
                
                # Efficiently, we can compute the residual once and then update each factor.
                # Residual R[i,j,k] = T[i,j,k] - sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
                
                # To update U[:, c], we fix V[:, c] and W[:, c].
                # We want to find u such that <u, R_flat_contracted> is minimized?
                # Actually, this is a linear least squares problem for each column c if we fix the others.
                
                # Let's compute the current tensor S = sum_c u_c \otimes v_c \otimes w_c
                # S[i,j,k] = sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
                
                # This is O(n*m*p*r). 4*5*6*r = 120r. Very fast.
                
                # We can update U column by column.
                # For a fixed c, let U_c = U[:, c], V_c = V[:, c], W_c = W[:, c].
                # The term involving U_c is U_c[i*m+j] * V_c[j*p+k] * W_c[k*n+i].
                # We want to minimize || T - S ||^2.
                # Derivative w.r.t U_c[i*m+j] is -2 * sum_{j,k} (T[i,j,k] - S[i,j,k]) * V_c[j*p+k] * W_c[k*n+i]
                # This is a linear system? No, it's a direct gradient.
                # We can use gradient descent.
                
                # Compute S
                S = np.zeros((n, m, p))
                for c in range(r):
                    # u is (n*m,), v is (m*p,), w is (p*n,)
                    # We need to reshape them to match indices.
                    # u[i*m+j] * v[j*p+k] * w[k*n+i]
                    # This is a bit complex to vectorize simply without einsum.
                    # np.einsum is perfect here.
                    # U_c is vector of size n*m. Reshape to (n, m).
                    # V_c is vector of size m*p. Reshape to (m, p).
                    # W_c is vector of size p*n. Reshape to (p, n).
                    # Term = U_c.reshape(n,m) * V_c.reshape(m,p) * W_c.reshape(p,n)
                    # But indices are: U[i,j], V[j,k], W[k,i].
                    # So term[i,j,k] = U[i,j] * V[j,k] * W[k,i].
                    # np.einsum('ij,jk,ki->ijk', U_c.reshape(n,m), V_c.reshape(m,p), W_c.reshape(p,n))
                    term = np.einsum('ij,jk,ki->ijk', U_curr[:, c].reshape(n, m), 
                                      V_curr[:, c].reshape(m, p), 
                                      W_curr[:, c].reshape(p, n))
                    S += term
                
                Res = T - S
                
                # Gradient Descent Step
                # Update U
                for c in range(r):
                    # Grad_U_c[i*m+j] = -2 * sum_{j,k} Res[i,j,k] * V_c[j*p+k] * W_c[k*n+i]
                    # Let's compute the contraction.
                    # G_U[i,j] = sum_{j,k} Res[i,j,k] * V_c[j,k] * W_c[k,i]
                    # Note: V_c is indexed by j*p+k, which corresponds to matrix V_c.reshape(m,p)[j,k].
                    # W_c is indexed by k*n+i, which corresponds to matrix W_c.reshape(p,n)[k,i].
                    
                    V_mat = V_curr[:, c].reshape(m, p)
                    W_mat = W_curr[:, c].reshape(p, n)
                    
                    # G_U[i,j] = sum_k Res[i,j,k] * V_mat[j,k] * W_mat[k,i]
                    # This is: Res[i,j,:] dot (V_mat[j,:] * W_mat[:,i])?
                    # Let's use einsum: 'ijk,jk,ki->ij'
                    G_U = np.einsum('ijk,jk,ki->ij', Res, V_mat, W_mat)
                    
                    # Update
                    U_curr[:, c] -= 0.01 * G_U.reshape(-1)
                    
                # Update V
                for c in range(r):
                    V_mat = V_curr[:, c].reshape(m, p)
                    W_mat = W_curr[:, c].reshape(p, n)
                    U_mat = U_curr[:, c].reshape(n, m)
                    
                    # G_V[j,k] = sum_{i,j} Res[i,j,k] * U_mat[i,j] * W_mat[k,i]
                    # 'ijk,ij,ki->jk'
                    G_V = np.einsum('ijk,ij,ki->jk', Res, U_mat, W_mat)
                    
                    V_curr[:, c] -= 0.01 * G_V.reshape(-1)
                    
                # Update W
                for c in range(r):
                    V_mat = V_curr[:, c].reshape(m, p)
                    W_mat = W_curr[:, c].reshape(p, n)
                    U_mat = U_curr[:, c].reshape(n, m)
                    
                    # G_W[k,i] = sum_{i,j} Res[i,j,k] * U_mat[i,j] * V_mat[j,k]
                    # 'ijk,ij,jk->ki'
                    G_W = np.einsum('ijk,ij,jk->ki', Res, U_mat, V_mat)
                    
                    W_curr[:, c] -= 0.01 * G_W.reshape(-1)
                    
                # Check convergence
                if np.linalg.norm(Res) < 1e-8:
                    # Verify exactly
                    # The problem asks for integer/half-integer entries. 
                    # Gradient descent might find a real solution. 
                    # We can try to round to nearest 0.5 and verify.
                    U_rounded = np.round(U_curr * 2) / 2
                    V_rounded = np.round(V_curr * 2) / 2
                    W_rounded = np.round(W_curr * 2) / 2
                    
                    # Check if rounded solution is valid
                    S_rounded = np.zeros((n, m, p))
                    for c in range(r):
                        term = np.einsum('ij,jk,ki->ijk', U_rounded[:, c].reshape(n, m), 
                                          V_rounded[:, c].reshape(m, p), 
                                          W_rounded[:, c].reshape(p, n))
                        S_rounded += term
                    if np.allclose(S_rounded, T, atol=1e-9):
                        best_rank = r
                        best_U = U_rounded.tolist()
                        best_V = V_rounded.tolist()
                        best_W = W_rounded.tolist()
                        found = True
                        break
                    # If not, continue iterating or break restart
                    # For now, if we get close, we might have a good starting point, 
                    # but since we need exact integer/half-integer, we accept only if it rounds perfectly.
                    
            if found:
                break
                
    return {"U": best_U, "V": best_V, "W": best_W}
