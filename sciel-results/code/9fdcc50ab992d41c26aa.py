import numpy as np
import time
from itertools import product

def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin
    
    n, m, p = 2, 3, 6
    
    # Build the target multiplication tensor T[i,j,k] = 1 for all i,j,k
    # The decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    def verify(U, V, W):
        """Check if the decomposition is exact."""
        rank = U.shape[1]
        # Compute the reconstructed tensor
        # T_recon[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
        # Use einsum for efficiency
        # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
        # Reshape: U -> (n,m,rank), V -> (m,p,rank), W -> (p,n,rank)
        U3 = U.reshape(n, m, rank)
        V3 = V.reshape(m, p, rank)
        W3 = W.reshape(p, n, rank)
        # T_recon[i,j,k] = sum_r U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
        # Note: W3[k,i,r] uses W[k*n+i, r]
        T_recon = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        # Target: T[i,j,k] = 1 for all i,j,k
        T_target = np.ones((n, m, p))
        return np.allclose(T_recon, T_target, atol=1e-10, rtol=1e-10)
    
    def verify_exact(U, V, W):
        """Check exact decomposition using fractions or high precision."""
        rank = U.shape[1]
        U3 = U.reshape(n, m, rank)
        V3 = V.reshape(m, p, rank)
        W3 = W.reshape(p, n, rank)
        T_recon = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        T_target = np.ones((n, m, p))
        return np.allclose(T_recon, T_target, atol=1e-8, rtol=1e-8)
    
    # Schoolbook decomposition as fallback
    rank_sb = n * m * p
    U_sb = np.zeros((n*m, rank_sb))
    V_sb = np.zeros((m*p, rank_sb))
    W_sb = np.zeros((p*n, rank_sb))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    best_result = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    best_rank = rank_sb
    
    # Try to find lower rank decompositions using randomized search
    # Strategy: use a greedy approach or randomized initialization with verification
    
    for target_rank in range(18, rank_sb):
        if time.time() > deadline - 20:
            break
        
        found = False
        for attempt in range(50):
            if time.time() > deadline - 10:
                break
            
            np.random.seed(attempt * 7919 + target_rank * 104729)
            
            # Random initialization
            U = np.random.randn(n*m, target_rank)
            V = np.random.randn(m*p, target_rank)
            W = np.random.randn(p*n, target_rank)
            
            # Try to refine using a simple iterative method
            # The constraint is: for all i,j,k: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
            # This is a system of n*m*p = 36 equations in 3*target_rank variables
            
            # Use a simple coordinate descent approach
            for it in range(200):
                if time.time() > deadline - 5:
                    break
                
                # Update U: for each (i,j), we need sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1 for all k
                # This is linear in U[i*m+j,:]: let a_k = sum_r V[j*p+k,r]*W[k*n+i,r]
                # Then we need U[i*m+j,:] dot a_k = 1 for all k
                # This is an underdetermined system (p equations, target_rank unknowns)
                
                improved = False
                for i in range(n):
                    for j in range(m):
                        # For fixed i,j, we need: for all k, sum_r U[i*m+j,r] * (V[j*p+k,r] * W[k*n+i,r]) = 1
                        # Let b_k(r) = V[j*p+k,r] * W[k*n+i,r]
                        # We need U[i*m+j,:] dot b_k = 1 for k=0,...,p-1
                        B_mat = np.zeros((p, target_rank))
                        for k in range(p):
                            B_mat[k, :] = V[j*p+k, :] * W[k*n+i, :]
                        
                        # Solve: U[i*m+j,:] @ B_mat.T = ones(p)
                        # i.e., B_mat @ U[i*m+j,:] = ones(p)
                        # Under-determined if target_rank > p, so use least squares
                        sol, residuals, rank_r, sv = np.linalg.lstsq(B_mat.T, np.ones(p), rcond=None)
                        U[i*m+j, :] = sol
                        improved = True
                
                # Update V: for each (j,k), need sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1 for all i
                for j in range(m):
                    for k in range(p):
                        B_mat = np.zeros((n, target_rank))
                        for i in range(n):
                            B_mat[i, :] = U[i*m+j, :] * W[k*n+i, :]
                        sol, residuals, rank_r, sv = np.linalg.lstsq(B_mat.T, np.ones(n), rcond=None)
                        V[j*p+k, :] = sol
                
                # Update W: for each (k,i), need sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1 for all j
                for k in range(p):
                    for i in range(n):
                        B_mat = np.zeros((m, target_rank))
                        for j in range(m):
                            B_mat[j, :] = U[i*m+j, :] * V[j*p+k, :]
                        sol, residuals, rank_r, sv = np.linalg.lstsq(B_mat.T, np.ones(m), rcond=None)
                        W[k*n+i, :] = sol
                
                if not improved:
                    break
            
            # Verify
            if verify_exact(U, V, W):
                # Check for half-integer entries
                U_check = np.round(U * 2) / 2
                V_check = np.round(V * 2) / 2
                W_check = np.round(W * 2) / 2
                if verify_exact(U_check, V_check, W_check):
                    U, V, W = U_check, V_check, W_check
                
                best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                best_rank = target_rank
                found = True
                break
        
        if found:
            break
    
    return best_result
