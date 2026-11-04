import numpy as np
import time

n, m, p = 2, 5, 6
nm, mp, pn = n * m, m * p, p * n


def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    # Schoolbook decomposition is the baseline
    rank = n * m * p
    U = np.zeros((nm, rank), dtype=np.float64)
    V = np.zeros((mp, rank), dtype=np.float64)
    W = np.zeros((pn, rank), dtype=np.float64)

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1

    best_U = U
    best_V = V
    best_W = W
    best_rank = rank

    # Attempt rank reduction using a structured approach:
    # We try to find if the tensor has rank < 60 by checking for linear
    # dependencies among the rank-1 components using randomized methods.
    
    if time.time() < deadline - 5.0:
        # Try to reduce rank by finding linear combinations that preserve the tensor
        # Strategy: try random linear transformations of the factor matrices
        # and check if the resulting tensor still matches
        
        # First, let's try a simple approach: check if we can merge pairs of columns
        # by finding if T = sum_u U[:,u] ⊗ V[:,u] ⊗ W[:,u] can be written with fewer terms
        
        # Use a different strategy: try to find a rank-59 decomposition by
        # perturbing the schoolbook and checking if it still works
        
        # For 2x5x6, we know the lower bound is max(2*5, 5*6, 6*2) = 30
        # and the upper bound is 60. Let's try to find if rank < 60 is achievable.
        
        # Attempt: use random linear combinations of columns to see if we can
        # reduce the effective rank while preserving the tensor
        
        # Create a random test matrix to verify decompositions
        test_A = np.random.randn(n, m)
        test_B = np.random.randn(m, p)
        test_C = test_A @ test_B
        
        def verify_decomposition(U_mat, V_mat, W_mat, rank_test):
            """Verify that the decomposition correctly computes matrix multiplication"""
            # Reconstruct the multiplication using the factorization
            # C[i,k] = sum_j A[i,j] * B[j,k] = sum_u U[i*m+j, u] * V[j*p+k, u] * W[k*n+i, u]
            # This is complex, so let's use a simpler check:
            # For each (i,j,k), the coefficient should match A[i,j]*B[j,k]
            
            # Build the tensor from the factorization
            # T[i,j,k] = sum_u U[i*m+j, u] * V[j*p+k, u] * W[k*n+i, u]
            # This should equal 1 for all (i,j,k) in the schoolbook case
            
            # Actually, the standard decomposition has:
            # C[i,k] = sum_j sum_u A[i,j] * U[i*m+j,u] * V[j*p+k,u] * B[j,k] * W[k*n+i,u] / (something)
            
            # Let me reconsider. The matrix multiplication tensor M_{n,m,p} has:
            # M_{i,j,k} = 1 if we're computing C[i,k] from A[i,j]*B[j,k]
            # The rank-1 decomposition is: sum_u U_u ⊗ V_u ⊗ W_u where
            # U_u[i*m+j] = 1 if (i,j) corresponds to term u
            # V_u[j*p+k] = 1 if (j,k) corresponds to term u  
            # W_u[k*n+i] = 1 if (k,i) corresponds to term u
            
            # For verification, check that the reconstructed tensor matches the original
            # Original tensor: T[i,j,k] = 1 for all valid (i,j,k)
            # Reconstructed: sum_u U[i*m+j, u] * V[j*p+k, u] * W[k*n+i, u]
            
            recon = np.zeros((n, m, p), dtype=np.float64)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        recon[i, j, k] = np.sum(
                            U_mat[i * m + j, :] * V_mat[j * p + k, :] * W_mat[k * n + i, :]
                        )
            
            # The original tensor has 1s everywhere
            original = np.ones((n, m, p), dtype=np.float64)
            return np.allclose(recon, original, atol=1e-10)
        
        # Try to reduce rank by finding linear dependencies
        # Method: try random rank-r factorizations and check if they work
        
        # Start with rank 59 and go down
        for target_rank in range(59, max(30, 59 - 20), -1):
            if time.time() > deadline - 5.0:
                break
            
            # Try a few random attempts at this rank
            for attempt in range(20):
                if time.time() > deadline - 5.0:
                    break
                
                # Create random factor matrices
                U_rand = np.random.randn(nm, target_rank)
                V_rand = np.random.randn(mp, target_rank)
                W_rand = np.random.randn(pn, target_rank)
                
                # Normalize to improve conditioning
                for u in range(target_rank):
                    norm_u = np.linalg.norm(U_rand[:, u])
                    norm_v = np.linalg.norm(V_rand[:, u])
                    norm_w = np.linalg.norm(W_rand[:, u])
                    if norm_u > 1e-10 and norm_v > 1e-10 and norm_w > 1e-10:
                        U_rand[:, u] /= norm_u
                        V_rand[:, u] /= norm_v
                        W_rand[:, u] /= norm_w
                
                # Verify if this decomposition works
                if verify_decomposition(U_rand, V_rand, W_rand, target_rank):
                    best_U = U_rand
                    best_V = V_rand
                    best_W = W_rand
                    best_rank = target_rank
                    print(f"Found rank {target_rank} decomposition!")
                    break
            
            if best_rank < rank:
                break

    # Return the best result found
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
