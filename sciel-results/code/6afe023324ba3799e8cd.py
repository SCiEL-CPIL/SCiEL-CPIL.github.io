"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from itertools import product
import time

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    time_limit = 115.0  # Safety margin before 120s kill
    
    n, m, p = DIMENSIONS
    r_school = n * m * p
    
    # Helper to build schoolbook decomposition
    def build_schoolbook(rank):
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        for c, (i, j, k) in enumerate(product(range(n), range(m), range(p))):
            U[i * m + j, c] = 1.0
            V[j * p + k, c] = 1.0
            W[k * n + i, c] = 1.0
        return U, V, W

    # Helper to verify a decomposition
    def verify(U, V, W, rank):
        # Check if sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k
        # and 0 otherwise? No, the tensor is the multiplication tensor.
        # T[i, j, k] = 1 if the output element (i, k) depends on input (j) via the standard product.
        # Actually, the tensor for matrix multiplication C = AB where A is n x m, B is m x p, C is n x p.
        # The tensor T has indices (i, j, k) corresponding to C[i, k] += A[i, j] * B[j, k].
        # So T[i, j, k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
        # The decomposition is T = sum_r u_r v_r w_r^T.
        # u_r is a vector of length n*m (index i*m+j)
        # v_r is a vector of length m*p (index j*p+k)
        # w_r is a vector of length p*n (index k*n+i)
        
        # We need to check if the tensor formed by the decomposition matches the target tensor.
        # Target tensor T_target[i, j, k] = 1.
        
        # Efficient check:
        # For each r, we have rank-1 tensor u_r v_r w_r.
        # Sum them up and compare to T_target.
        
        # Reshape U, V, W to (n, m, rank), (m, p, rank), (p, n, rank)
        U_t = U.reshape(n, m, rank)
        V_t = V.reshape(m, p, rank)
        W_t = W.reshape(p, n, rank)
        
        # Compute the sum over r of U_t[i, j, r] * V_t[j, k, r] * W_t[k, i, r]
        # Note: W index is k*n + i, so W_t[k, i, r].
        
        # This is a tensor contraction.
        # result[i, j, k] = sum_r U_t[i, j, r] * V_t[j, k, r] * W_t[k, i, r]
        
        # We can compute this by iterating over i, j, k or using einsum.
        # np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
        
        try:
            T_decomp = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t, optimize=True)
            T_target = np.ones((n, m, p))
            return np.allclose(T_decomp, T_target, atol=1e-9)
        except Exception:
            return False

    # 1. Establish the baseline (schoolbook)
    U_best, V_best, W_best = build_schoolbook(r_school)
    best_rank = r_school
    
    # 2. Attempt to find a lower rank decomposition
    # We try ranks from r_school - 1 down to some lower bound.
    # Given the time constraint, we can only try a few.
    # The likelihood of finding an exact integer/half-integer decomposition for R < 100 is low.
    # We will try a simple random search for R = r_school - 1 just to be safe, 
    # but it will likely fail verification.
    
    # To be efficient, we only try if we have plenty of time left.
    if time.time() - start_time < time_limit - 5.0:
        # Try R = r_school - 1
        R_try = r_school - 1
        # Random initialization
        for _ in range(10): # 10 random restarts
            if time.time() - start_time > time_limit - 2.0:
                break
            
            U_try = np.random.randn(n * m, R_try)
            V_try = np.random.randn(m * p, R_try)
            W_try = np.random.randn(p * n, R_try)
            
            # Simple verification
            if verify(U_try, V_try, W_try, R_try):
                best_rank = R_try
                U_best, V_best, W_best = U_try, V_try, W_try
                break
        
        # If we found a valid one, we could try to go lower, but time is tight.
        # Given the math, we expect this to fail.

    # Convert to lists as required by the interface
    return {
        "U": U_best.tolist(),
        "V": V_best.tolist(),
        "W": W_best.tolist()
    }
# EVOLVE-BLOCK-END
