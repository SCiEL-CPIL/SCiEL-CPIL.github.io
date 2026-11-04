"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    n, m, p = DIMENSIONS
    # The tensor T[i,j,k] = 1 for the multiplication: C[i,k] = sum_j A[i,j]B[j,k]
    # We want to find U (n*m x r), V (m*p x r), W (p*n x r) such that
    # T[i*m+j, k*n+i] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Wait, the convention is W[k*n+i, r], so W is indexed by (k, i) with k in [0,p), i in [0,n)
    
    # Let me re-read: U (n*m, rank), V (m*p, rank), W (p*n, rank)
    # U[i*m + j, c], V[j*p + k, c], W[k*n + i, c]
    # The product should satisfy: for all i,j,k: delta = U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] summed over c
    # equals 1 if this is the "diagonal" term... 
    
    # Actually, the matrix multiplication tensor T has T[i,j,k] = 1 for all i,j,k in the appropriate ranges,
    # and the decomposition is: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # where the indices are: U indexed by (i,j), V by (j,k), W by (k,i)
    
    # So we need: for all i in [0,n), j in [0,m), k in [0,p):
    #   sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # This is a rank-r decomposition of the 3-tensor T with T[i,j,k] = 1.
    
    # The schoolbook solution uses r = n*m*p = 48, with U[i*m+j, i*m*j*... ] 
    
    # Let me try to find a lower rank decomposition.
    
    # For now, let's try rank 32 and see if we can solve it.
    # We can set up a system of equations.
    
    # Let's try a few ranks and see which ones are feasible.
    # We'll use a randomized approach: fix U and V randomly, then solve for W, and check if it works.
    # Or better: use the fact that this is a bilinear problem.
    
    # Actually, let me just return the schoolbook for now and add a search.
    
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to find a lower rank decomposition using a simple approach
    # We'll try ranks from 24 up to 47
    
    best_result = None
    
    for rank in range(24, 48):
        if time.time() > deadline:
            break
        
        # Try to find a decomposition of this rank
        # Use a simple iterative approach: initialize U, V, W randomly, then optimize
        # This is a non-convex problem, so we'll try multiple random restarts
        
        success = False
        for restart in range(50):
            if time.time() > deadline:
                break
            
            # Random initialization
            U = np.random.randn(n * m, rank)
            V = np.random.randn(m * p, rank)
            W = np.random.randn(p * n, rank)
            
            # Normalize to avoid scale issues
            for c in range(rank):
                norm_u = np.linalg.norm(U[:, c])
                norm_v = np.linalg.norm(V[:, c])
                norm_w = np.linalg.norm(W[:, c])
                if norm_u > 1e-10 and norm_v > 1e-10 and norm_w > 1e-10:
                    scale = (norm_u * norm_v * norm_w) ** (1.0 / 3.0)
                    U[:, c] /= norm_u * scale
                    V[:, c] /= norm_v * scale
                    W[:, c] /= norm_w * scale
            
            # Now we want to solve: for all i,j,k: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
            # This is equivalent to: the tensor product should equal the all-ones tensor
            
            # Let's check the current error
            def compute_error(U, V, W):
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(rank))
                            err += (val - 1.0) ** 2
                return err
            
            # Simple gradient-free approach: just check if random init happens to work (unlikely)
            # Instead, let's use a smarter approach: fix U and V, solve for W linearly
            
            # For fixed U and V, the constraint is:
            # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i,j,k
            # This is linear in W. Let's set up the system.
            
            # Number of constraints: n*m*p = 48
            # Number of unknowns: p*n*rank = 8*2*rank = 16*rank
            # For rank >= 3, we have more unknowns than constraints, so it's underdetermined.
            
            # Let's build the linear system A @ W_vec = b
            # where W_vec is the vectorization of W
            
            A = np.zeros((n * m * p, p * n * rank))
            b = np.ones(n * m * p)
            
            idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        for c in range(rank):
                            coeff = U[i*m+j, c] * V[j*p+k, c]
                            if abs(coeff) > 1e-15:
                                # W[k*n+i, c] contributes
                                w_idx = (k * n + i) * rank + c
                                A[idx, w_idx] = coeff
                        idx += 1
            
            # Solve A @ W_vec = b using least squares
            try:
                W_vec, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
                W = W_vec.reshape(p * n, rank)
            except:
                continue
            
            # Check if the solution is accurate
            err = compute_error(U, V, W)
            if err < 1e-10:
                success = True
                break
        
        if success:
            # Convert to lists
            result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            if best_result is None or rank < len(best_result["U"][0]):
                best_result = result
            
            if time.time() > deadline:
                break
    
    if best_result is not None:
        return best_result
    else:
        return schoolbook()
# EVOLVE-BLOCK-END

