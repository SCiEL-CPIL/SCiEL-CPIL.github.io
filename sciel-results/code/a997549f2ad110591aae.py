"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Build the target multiplication tensor as a 3D array T[a, b, c]
    # where a = i*m+j (input), b = j*p+k (input), c = k*n+i (output)
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Best solution: start with schoolbook
    best_rank = rank_schoolbook
    best_U = np.zeros((n * m, rank_schoolbook))
    best_V = np.zeros((m * p, rank_schoolbook))
    best_W = np.zeros((p * n, rank_schoolbook))
    for idx in range(rank_schoolbook):
        i, j, k = divmod(idx, m * p)
        j, k = divmod(j, p)
        a = i * m + j
        b = j * p + k
        c = k * n + i
        best_U[a, idx] = 1
        best_V[b, idx] = 1
        best_W[c, idx] = 1
    
    # Try to find better decomposition using randomized SVD-based approach
    # The idea: randomly sample subsets of the tensor and try low-rank approximations
    
    rng = np.random.RandomState(42)
    
    # Try various target ranks below schoolbook
    for target_rank in range(rank_schoolbook - 1, 20, -1):
        if time.time() > deadline - 10:
            break
            
        # Try multiple random initializations
        for attempt in range(5):
            if time.time() > deadline - 10:
                break
                
            # Randomly initialize factor matrices
            U_rand = rng.randn(n * m, target_rank)
            V_rand = rng.randn(m * p, target_rank)
            W_rand = rng.randn(p * n, target_rank)
            
            # Scale to unit norm columns
            for r in range(target_rank):
                nU = np.linalg.norm(U_rand[:, r])
                nV = np.linalg.norm(V_rand[:, r])
                nW = np.linalg.norm(W_rand[:, r])
                if nU > 0 and nV > 0 and nW > 0:
                    U_rand[:, r] /= nU
                    V_rand[:, r] /= nV
                    W_rand[:, r] /= nW
            
            # Iterate: update each factor matrix by projecting the residual
            for iteration in range(10):
                if time.time() > deadline - 5:
                    break
                    
                # Compute current approximation
                # Residual = T - sum_r U[:,r] * V[:,r]^T * W[:,r]^T
                # For efficiency, compute residuals for each factor update
                
                # Update U: for each a, residual_a = T[a,:,:] - sum_r U[a,r] * (V[:,r] * W[:,r]^T)
                # We want U_new[a,r] = <residual_a, V[:,r] * W[:,r]^T> / ||V[:,r] * W[:,r]^T||^2
                
                # Precompute V * W for each r
                VW = np.zeros((m * p, p * n, target_rank))
                for r in range(target_rank):
                    VW[:, :, r] = np.outer(V_rand[:, r], W_rand[:, r])
                
                # Update U
                U_new = np.zeros((n * m, target_rank))
                for a in range(n * m):
                    for r in range(target_rank):
                        num = np.sum(T[a, :, :] * VW[:, :, r])
                        den = np.sum(VW[:, :, r] ** 2)
                        if den > 1e-12:
                            U_new[a, r] = num / den
                U_rand = U_new
                
                # Update V
                UW = np.zeros((n * m, p * n, target_rank))
                for r in range(target_rank):
                    UW[:, :, r] = np.outer(U_rand[:, r], W_rand[:, r])
                
                V_new = np.zeros((m * p, target_rank))
                for b in range(m * p):
                    for r in range(target_rank):
                        num = np.sum(T[:, b, :] * UW[:, :, r])
                        den = np.sum(UW[:, :, r] ** 2)
                        if den > 1e-12:
                            V_new[b, r] = num / den
                V_rand = V_new
                
                # Update W
                UV = np.zeros((n * m, m * p, target_rank))
                for r in range(target_rank):
                    UV[:, :, r] = np.outer(U_rand[:, r], V_rand[:, r])
                
                W_new = np.zeros((p * n, target_rank))
                for c in range(p * n):
                    for r in range(target_rank):
                        num = np.sum(T[:, :, c] * UV[:, :, r])
                        den = np.sum(UV[:, :, r] ** 2)
                        if den > 1e-12:
                            W_new[c, r] = num / den
                W_rand = W_new
            
            # Check if we have an exact (or very close) solution
            # Compute the error
            err = 0.0
            for r in range(target_rank):
                for a in range(n * m):
                    for b in range(m * p):
                        for c in range(p * n):
                            approx = U_rand[a, r] * V_rand[b, r] * W_rand[c, r]
                            err += (T[a, b, c] - approx) ** 2
            
            if err < 1e-6:
                # Found a better solution
                best_rank = target_rank
                best_U = U_rand
                best_V = V_rand
                best_W = W_rand
                break  # No need to try lower ranks for this attempt
        
        if time.time() > deadline - 10:
            break
    
    # Convert to lists
    def to_list(matrix):
        return [[float(matrix[i, j]) for j in range(matrix.shape[1])] for i in range(matrix.shape[0])]
    
    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
# EVOLVE-BLOCK-END
