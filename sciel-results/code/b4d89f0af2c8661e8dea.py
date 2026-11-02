"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 140.0  # safety margin
    
    n, m, p = DIMENSIONS
    # U: (n*m) x r, V: (m*p) x r, W: (p*n) x r
    # Tensor T: T[i*m+j, j*p+k, k*n+i] = 1 for valid (i,j,k)
    
    # Build the tensor as a 3D array
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    best_rank = n*m*p  # schoolbook
    best_result = None
    
    # Try ranks from low to high
    for r in range(40, n*m*p+1):
        if time.time() > deadline - 5:
            break
        
        success = False
        for trial in range(20):
            if time.time() > deadline - 3:
                break
            
            # Random initialization
            np.random.seed(trial * 1000 + r)
            U = np.random.randn(n*m, r) * 0.1
            V = np.random.randn(m*p, r)
            W = np.random.randn(p*n, r)
            
            # Alternating minimization
            for it in range(100):
                if time.time() > deadline - 3:
                    break
                
                # Update U: for each row i, solve linear system
                for i in range(n*m):
                    # T[i, :, :] = U[i,:] · (V ⊙ W) element-wise
                    # We need: sum_t U[i,t] * V[j,t] * W[k,t] = T[i,j,k]
                    # This is: U[i,:] · M = T[i,:,:].reshape(-1)
                    # where M[j*k + l, t] = V[j,t] * W[k,t] for j in range(m*p), k in range(p*n)
                    # But j and k are not independent! j is from 0..m*p-1, k from 0..p*n-1
                    # Actually, T[i, j, k] is only non-zero for specific (j,k) pairs
                    # Let me think about this differently
                    
                    # For fixed i, we have equations:
                    # For each (j,k) where T[i,j,k] != 0:
                    #   sum_t U[i,t] * V[j,t] * W[k,t] = T[i,j,k]
                    # This is a linear system in U[i,:]
                    
                    # Collect all (j,k) pairs for this i
                    A_mat = np.zeros((m*p*p*n, r))  # this is too large
                    # Actually, let me just use all (j,k) pairs, but T is sparse
                    # For efficiency, only use non-zero entries
                    nonzero_jk = np.argwhere(T[i] != 0)
                    if len(nonzero_jk) == 0:
                        continue
                    A_sub = np.zeros((len(nonzero_jk), r))
                    b_sub = T[i][nonzero_jk[:,0], nonzero_jk[:,1]]
                    for idx, (j, k) in enumerate(nonzero_jk):
                        A_sub[idx] = V[j] * W[k]
                    # Solve least squares
                    U[i], residuals, rank_A, sv = np.linalg.lstsq(A_sub, b_sub, rcond=None)
                
                # Update V similarly
                for j in range(m*p):
                    nonzero_ik = np.argwhere(T[:, j, :] != 0)
                    if len(nonzero_ik) == 0:
                        continue
                    A_sub = np.zeros((len(nonzero_ik), r))
                    b_sub = T[nonzero_ik[:,0], j, nonzero_ik[:,1]]
                    for idx, (i, k) in enumerate(nonzero_ik):
                        A_sub[idx] = U[i] * W[k]
                    V[j], residuals, rank_A, sv = np.linalg.lstsq(A_sub, b_sub, rcond=None)
                
                # Update W similarly
                for k in range(p*n):
                    nonzero_ij = np.argwhere(T[:, :, k] != 0)
                    if len(nonzero_ij) == 0:
                        continue
                    A_sub = np.zeros((len(nonzero_ij), r))
                    b_sub = T[nonzero_ij[:,0], nonzero_ij[:,1], k]
                    for idx, (i, j) in enumerate(nonzero_ij):
                        A_sub[idx] = U[i] * V[j]
                    W[k], residuals, rank_A, sv = np.linalg.lstsq(A_sub, b_sub, rcond=None)
            
            # Check if we have a good solution
            error = 0.0
            for i in range(n*m):
                for j in range(m*p):
                    for k in range(p*n):
                        val = np.sum(U[i] * V[j] * W[k])
                        error += (val - T[i,j,k])**2
            
            if error < 1e-10:
                # Round to half-integers
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                # Verify rounded solution
                error_rounded = 0.0
                for i in range(n*m):
                    for j in range(m*p):
                        for k in range(p*n):
                            val = np.sum(U_rounded[i] * V_rounded[j] * W_rounded[k])
                            error_rounded += (val - T[i,j,k])**2
                
                if error_rounded < 1e-10:
                    success = True
                    best_rank = r
                    best_result = (U_rounded, V_rounded, W_rounded)
                    break
        
        if success:
            break
    
    if best_result is None:
        # Fall back to schoolbook
        r = n*m*p
        U = np.zeros((n*m, r))
        V = np.zeros((m*p, r))
        W = np.zeros((p*n, r))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        best_result = (U, V, W)
    
    U, V, W = best_result
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
