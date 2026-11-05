"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import random

def construct():
    """
    Find a low-rank exact tensor decomposition for 3x3 by 3x8 matrix multiplication.
    
    Uses a structured search exploiting the block-sparse nature of the MM tensor.
    For MM(3,3,8), we seek rank r where r < 72 (schoolbook).
    
    Strategy: Use the structure of the tensor to guide a combinatorial search
    for a sparse exact decomposition. The tensor T[(i,j),(j',k),(k',i')] = 1 iff
    j=j' and k=k' and i=i'.
    
    We formulate this as finding rank-1 components that collectively span the tensor.
    Using iterative methods with sparsity constraints to find minimal rank.
    """
    n, m, p = DIMENSIONS
    
    # The tensor T has shape (n*m, m*p, p*n) = (9, 24, 24)
    # T[a, b, c] where a = i*m+j, b = j'*p+k, c = k'*n+i'
    # T[a, b, c] = 1 iff j==j' and k==k' and i==i'
    
    # Build the target tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Schoolbook decomposition as fallback
    def schoolbook_decomposition():
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W
    
    U_school, V_school, W_school = schoolbook_decomposition()
    
    # Try to find lower rank decompositions using iterative hard thresholding
    # and gradient descent
    
    def validate_decomposition(U, V, W, tol=1e-10):
        """Check if U*V*W decomposition is valid."""
        rank = U.shape[1]
        T_recon = np.zeros((n*m, m*p, p*n))
        for r in range(rank):
            u = U[:, r]
            v = V[:, r]
            w = W[:, r]
            T_recon += np.einsum('i,j,k->ijk', u, v, w)
        return np.allclose(T_recon, T, atol=tol)
    
    def try_rank(rank, max_iter=500, lr=0.005, seed=42):
        """Try to find a decomposition of given rank using IHT + gradient descent."""
        np.random.seed(seed)
        random.seed(seed)
        
        # Initialize with random vectors
        U = np.random.randn(n*m, rank) * 0.1
        V = np.random.randn(m*p, rank) * 0.1
        W = np.random.randn(p*n, rank) * 0.1
        
        for it in range(max_iter):
            # Compute current reconstruction
            T_recon = np.zeros((n*m, m*p, p*n))
            for r in range(rank):
                u = U[:, r]
                v = V[:, r]
                w = W[:, r]
                T_recon += np.einsum('i,j,k->ijk', u, v, w)
            
            resid = T - T_recon
            resid_norm = np.linalg.norm(resid)
            
            if resid_norm < 1e-10:
                return U, V, W, True
            
            # Gradient step
            for r in range(rank):
                u = U[:, r]
                v = V[:, r]
                w = W[:, r]
                
                # Gradient w.r.t. U[:,r]
                grad_u = np.einsum('ijk,j->i', resid, v * w)
                grad_v = np.einsum('ijk,k->j', resid, u * w)
                grad_w = np.einsum('ijk,i->k', resid, u * v)
                
                U[:, r] -= lr * grad_u
                V[:, r] -= lr * grad_v
                W[:, r] -= lr * grad_w
            
            # Hard thresholding: zero out small terms
            for r in range(rank):
                if np.linalg.norm(U[:, r]) < 1e-8:
                    U[:, r] = 0
                if np.linalg.norm(V[:, r]) < 1e-8:
                    V[:, r] = 0
                if np.linalg.norm(W[:, r]) < 1e-8:
                    W[:, r] = 0
        
        # Check if we succeeded
        if validate_decomposition(U, V, W):
            return U, V, W, True
        return None, None, None, False
    
    # Try ranks from lower bound upward
    # Theoretical lower bound: max(n*m, m*p, p*n) = max(9, 24, 24) = 24
    # But for matrix multiplication, the lower bound is typically higher.
    # For 3x3x3, rank 7 is known. For 3x3x8, let's try from 24 upward.
    
    best_rank = n * m * p  # Default to schoolbook
    best_decomp = (U_school, V_school, W_school)
    
    # Try a few ranks below schoolbook
    # We'll try ranks that are multiples of some structure
    candidate_ranks = []
    for r in range(24, 72):
        candidate_ranks.append(r)
    
    # Try from lower to higher
    for test_rank in candidate_ranks:
        # Try multiple random seeds
        for seed in range(3):
            U_test, V_test, W_test, success = try_rank(test_rank, max_iter=300, seed=seed)
            if success:
                best_rank = test_rank
                best_decomp = (U_test, V_test, W_test)
                break
        if best_rank < best_decomp[0].shape[1]:
            break
    
    # Return the best decomposition found
    U_final, V_final, W_final = best_decomp
    return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
# EVOLVE-BLOCK-END
