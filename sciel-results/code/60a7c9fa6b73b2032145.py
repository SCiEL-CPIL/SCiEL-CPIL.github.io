"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 6]

import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill

    n, m, p = DIMENSIONS
    # Schoolbook rank
    rank_schoolbook = n * m * p

    # Initialize with schoolbook
    def schoolbook():
        r = n * m * p
        U = np.zeros((n * m, r))
        V = np.zeros((m * p, r))
        W = np.zeros((p * n, r))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    U_best, V_best, W_best = schoolbook()
    best_rank = rank_schoolbook

    # Try to find lower rank via random initialization + iterative refinement
    # We use the identity: the multiplication tensor T[i,j,k] = delta
    # T = sum_r U[:,r] * V[:,r] * W[:,r] (outer product)
    # We want to minimize r such that T = sum_r u_r * v_r * w_r

    # Random search with iterative refinement
    def try_rank(r, max_iters=500, seed=0):
        """Try to find a rank-r decomposition via random init + gradient-like refinement."""
        if r >= rank_schoolbook:
            return None
        
        rng = np.random.RandomState(seed)
        
        # Random initialization
        U = rng.randn(n * m, r)
        V = rng.randn(m * p, r)
        W = rng.randn(p * n, r)
        
        # The target tensor is T[i,j,k] = 1 if we're looking at the multiplication tensor
        # Actually, the multiplication tensor for C = A*B where A is n x m, B is m x p:
        # C[i,k] = sum_j A[i,j] B[j,k]
        # The tensor T[i,j,k] has T[i,j,k] = 1 for all i,j,k (it's the all-ones tensor in the relevant subspace)
        # Wait, no. The multiplication tensor is T[i,j,k] = delta_{i,j} * delta_{j,k}... 
        # Actually for matrix multiplication, the tensor is T[i,j,k] = 1 for all (i,j,k) where i in [0,n), j in [0,m), k in [0,p)
        # No wait. Let me think again.
        
        # The tensor for multiplying (n x m) by (m x p) matrix:
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # So the tensor T has entries T[i,j,k] = 1 for all i,j,k
        # And we need T = sum_r U[:,r] * V[:,r] * W[:,r]
        # where U is (n*m) x r, V is (m*p) x r, W is (p*n) x r
        # Wait, the indexing is:
        # U[i*m + j, r], V[j*p + k, r], W[k*n + i, r]
        # So T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # And we need T[i,j,k] = 1 for all valid i,j,k
        
        # So the target is the all-ones tensor of shape (n, m, p)
        # Reshaped: U index is i*m+j, V index is j*p+k, W index is k*n+i
        
        # Let's verify: for the schoolbook, U[i*m+j, i*m+j] = 1 (column i*m+j), 
        # V[j*p+k, i*m+j] = 1 only when... wait no.
        
        # In schoolbook: column c corresponds to (i,j,k) where c = i*m*p + j*p + k
        # U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1
        # So sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 if there exists c where all three are 1
        # U[i*m+j,c]=1 means c corresponds to some (i',j',k') with i'=i, j'=j
        # V[j*p+k,c]=1 means c corresponds to (i',j',k') with j'=j, k'=k  
        # W[k*n+i,c]=1 means c corresponds to (i',j',k') with k'=k, i'=i
        # So all three are 1 iff i'=i, j'=j, k'=k, which is exactly one column.
        # So the product sums to 1. Good.
        
        # So we need: for all i in [0,n), j in [0,m), k in [0,p):
        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        
        # This is a system of n*m*p equations.
        
        # Let's use a simpler approach: random search with local optimization
        
        # Build the constraint matrix
        # For each (i,j,k), we have a constraint
        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        
        # This is nonlinear. Let's try a different approach.
        
        # Actually, let's just do random search with verification.
        # For small ranks, we can try random integer/half-integer entries.
        
        # For rank r, each of U, V, W has entries that should be "nice" (integers or half-integers)
        # Let's try random small integers
        
        for attempt in range(max_iters):
            if time.time() > deadline:
                return None
            
            # Random small integer entries
            scale = 2
            U = rng.randint(-scale, scale + 1, size=(n * m, r)) * 0.5
            V = rng.randint(-scale, scale + 1, size=(m * p, r)) * 0.5
            W = rng.randint(-scale, scale + 1, size=(p * n, r)) * 0.5
            
            # Check if this works
            # Compute the tensor
            error = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for r_idx in range(r):
                            val += U[i * m + j, r_idx] * V[j * p + k, r_idx] * W[k * n + i, r_idx]
                        if abs(val - 1.0) > 1e-9:
                            error = 1.0
                            break
                    if error:
                        break
                if error:
                    break
            
            if not error:
                return U, V, W
        
        return None

    # Try ranks from low to high
    # Known lower bounds: for n x m x p matrix multiplication, rank >= max(n*m, m*p, p*n, n+m+p-2, ...)
    # For 3x5x6: lower bound is at least max(15, 30, 18, 3+5+6-2=12) = 30
    # Actually the trivial lower bound is max(n*m, m*p, p*n) = max(15, 30, 18) = 30
    # But we can do better with known results.
    
    # Let's start from a reasonable lower bound
    lower_bound = max(n * m, m * p, p * n)
    
    # Try a few random seeds for each rank
    max_attempts_per_rank = 200
    
    for r in range(lower_bound, rank_schoolbook):
        if time.time() > deadline:
            break
        
        result = try_rank(r, max_iters=max_attempts_per_rank, seed=r * 1000)
        if result is not None:
            U_best, V_best, W_best = result
            best_rank = r
            break
    
    # Convert to lists
    U_list = U_best.tolist()
    V_list = V_best.tolist()
    W_list = W_best.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
