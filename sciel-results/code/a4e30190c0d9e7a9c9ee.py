"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    deadline = time.time() + 140  # 140s to leave safety margin
    
    best_result = None
    best_rank = schoolbook_rank
    
    # Schoolbook as fallback
    U_sb = np.zeros((n*m, schoolbook_rank))
    V_sb = np.zeros((m*p, schoolbook_rank))
    W_sb = np.zeros((p*n, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    best_result = (U_sb, V_sb, W_sb)
    
    # Target ranks to try (below schoolbook)
    target_ranks = list(range(schoolbook_rank - 1, max(schoolbook_rank - 10, 50), -1))
    
    def tensor_error(U, V, W):
        """Compute the error of the tensor decomposition."""
        n, m, p = DIMENSIONS
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
        return max_err
    
    def try_optimize(rank, seed=None, max_iters=5000, tol=1e-8):
        """Try to find a decomposition of given rank using alternating minimization."""
        if seed is not None:
            np.random.seed(seed)
        
        # Structured initialization: start from schoolbook and perturb
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        # Initialize with structured pattern
        for r in range(rank):
            # Assign each rank component to a (i,j,k) triple with some randomness
            i = random.randint(0, n-1)
            j = random.randint(0, m-1)
            k = random.randint(0, p-1)
            U[i*m+j, r] = 1.0
            V[j*p+k, r] = 1.0
            W[k*n+i, r] = 1.0
        
        # Add small random perturbations to break symmetry
        U += np.random.randn(n*m, rank) * 0.01
        V += np.random.randn(m*p, rank) * 0.01
        W += np.random.randn(p*n, rank) * 0.01
        
        # Alternating minimization
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Update U: for each (i,j), solve U[i*m+j, :] to minimize error
            for i in range(n):
                for j in range(m):
                    # U[i*m+j, r] should satisfy: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all k
                    # This is a linear system: U[i*m+j, :] * (V[j*p+k,:] * W[k*n+i,:]) = 1
                    # For each k, we have a linear equation in U[i*m+j, :]
                    # Stack: A[k, r] = V[j*p+k, r] * W[k*n+i, r], want A @ U[i*m+j,:] = 1
                    A = V[j*p:j*p+p, :] * W[:, :].T  # shape (p, rank) but need W[k*n+i, r]
                    # Actually: for k in 0..p-1: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
                    # A[k, r] = V[j*p+k, r] * W[k*n+i, r]
                    A = np.zeros((p, rank))
                    for k in range(p):
                        A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    # Solve A @ u = ones(p)
                    try:
                        u_new = np.linalg.lstsq(A, np.ones(p), rcond=None)[0]
                    except:
                        continue
                    U[i*m+j, :] = u_new
            
            # Update V: for each (j,k), solve V[j*p+k, :] 
            for j in range(m):
                for k in range(p):
                    # sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all i
                    A = np.zeros((n, rank))
                    for i in range(n):
                        A[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    try:
                        v_new = np.linalg.lstsq(A, np.ones(n), rcond=None)[0]
                    except:
                        continue
                    V[j*p+k, :] = v_new
            
            # Update W: for each (k,i), solve W[k*n+i, :]
            for k in range(p):
                for i in range(n):
                    # sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all j
                    A = np.zeros((m, rank))
                    for j in range(m):
                        A[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    try:
                        w_new = np.linalg.lstsq(A, np.ones(m), rcond=None)[0]
                    except:
                        continue
                    W[k*n+i, :] = w_new
            
            # Check convergence
            if it % 10 == 0:
                err = tensor_error(U, V, W)
                if err < tol:
                    return U, V, W, err
        
        err = tensor_error(U, V, W)
        if err < tol:
            return U, V, W, err
        return None, None, None, err
    
    # Try to find lower rank decompositions
    for target_rank in target_ranks:
        if time.time() > deadline - 20:
            break
        
        # Multiple random restarts for this rank
        num_restarts = 5
        for restart in range(num_restarts):
            if time.time() > deadline - 10:
                break
            
            seed = random.randint(0, 2**31 - 1)
            U, V, W, err = try_optimize(target_rank, seed=seed, max_iters=3000, tol=1e-7)
            
            if U is not None and err < 1e-6:
                # Found a valid decomposition!
                best_rank = target_rank
                best_result = (U, V, W)
                break
        
        if best_rank < schoolbook_rank:
            break
    
    # Convert to list of lists with rounded values
    U_final, V_final, W_final = best_result
    U_list = [[float(round(U_final[i, r], 6)) for r in range(U_final.shape[1])] for i in range(U_final.shape[0])]
    V_list = [[float(round(V_final[i, r], 6)) for r in range(V_final.shape[1])] for i in range(V_final.shape[0])]
    W_list = [[float(round(W_final[i, r], 6)) for r in range(W_final.shape[1])] for i in range(W_final.shape[0])]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
