"""Rewritten program for matrix multiplication 3x5x6 using ALS tensor decomposition."""
import time
import random
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS
    dim_u = n * m  # 15
    dim_v = m * p  # 30
    dim_w = p * n  # 18
    
    def verify(U, V, W, rank):
        """Verify the decomposition exactly using integer arithmetic where possible."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    actual = 0.0
                    for c in range(rank):
                        actual += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(actual - 1.0) > 1e-6:
                        return False
        return True
    
    def round_to_half_integer(x):
        """Round to nearest half-integer (multiple of 0.5)."""
        return round(x * 2) / 2.0
    
    def als_decompose(target_rank, max_iters=500, tol=1e-8, seed=42):
        """
        Use Alternating Least Squares to find a rank-target_rank decomposition.
        Returns (U, V, W) as numpy arrays or None if failed.
        """
        rng = np.random.RandomState(seed)
        
        # Initialize
        U = rng.randn(dim_u, target_rank)
        V = rng.randn(dim_v, target_rank)
        W = rng.randn(dim_w, target_rank)
        
        # Build the target tensor in flattened form for efficient ALS
        # T[i, j, k] = 1 for all i,j,k
        # We can work with the tensor directly
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                return None
            
            # Solve for U: for each (j,k), U[:,c] * V[j,c] * W[k,c] should sum to 1
            # U[i, c] = sum over j,k of T[i,j,k] * V[j,c] * W[k,c] / (sum over j of V[j,c]^2 * sum over k of W[k,c]^2)
            # More precisely: U[i,c] minimizes sum_{j,k} (T[i,j,k] - sum_c' U[i,c']*V[j,c']*W[k,c'])^2
            # Taking derivative w.r.t U[i,c]: sum_{j,k} T[i,j,k] V[j,c] W[k,c] - U[i,c] * (sum_j V[j,c]^2) * (sum_k W[k,c]^2) = 0
            # So U[i,c] = (sum_{j,k} T[i,j,k] V[j,c] W[k,c]) / ((sum_j V[j,c]^2) * (sum_k W[k,c]^2))
            
            # Since T[i,j,k] = 1 for all i,j,k:
            # sum_{j,k} V[j,c] W[k,c] = (sum_j V[j,c]) * (sum_k W[k,c])
            
            sum_V = np.sum(V, axis=0)  # shape (target_rank,)
            sum_W = np.sum(W, axis=0)  # shape (target_rank,)
            sum_V2 = np.sum(V**2, axis=0)  # shape (target_rank,)
            sum_W2 = np.sum(W**2, axis=0)  # shape (target_rank,)
            
            denom_U = sum_V2 * sum_W2  # shape (target_rank,)
            denom_U[denom_U < 1e-12] = 1e-12
            
            # U[i,c] = sum_V[c] * sum_W[c] / denom_U[c] for all i
            # So all rows of U are the same!
            U_new = np.ones((dim_u, target_rank)) * (sum_V * sum_W / denom_U)
            
            # Check convergence
            diff = np.max(np.abs(U_new - U))
            U = U_new
            
            # Solve for V: V[j,c] = (sum_i U[i,c] * sum_k W[k,c]) / ((sum_i U[i,c]^2) * (sum_k W[k,c]^2))
            sum_U = np.sum(U, axis=0)
            sum_U2 = np.sum(U**2, axis=0)
            denom_V = sum_U2 * sum_W2
            denom_V[denom_V < 1e-12] = 1e-12
            
            V_new = np.ones((dim_v, target_rank)) * (sum_U * sum_W / denom_V)
            V = V_new
            
            # Solve for W: W[k,c] = (sum_i U[i,c] * sum_j V[j,c]) / ((sum_i U[i,c]^2) * (sum_j V[j,c]^2))
            denom_W = sum_U2 * sum_V2
            denom_W[denom_W < 1e-12] = 1e-12
            
            W_new = np.ones((dim_w, target_rank)) * (sum_U * sum_V / denom_W)
            W = W_new
            
            if diff < tol:
                break
        
        # Convert to lists and round
        U_list = [[round_to_half_integer(U[i,c]) for c in range(target_rank)] for i in range(dim_u)]
        V_list = [[round_to_half_integer(V[j,c]) for c in range(target_rank)] for j in range(dim_v)]
        W_list = [[round_to_half_integer(W[k,c]) for c in range(target_rank)] for k in range(dim_w)]
        
        return U_list, V_list, W_list
    
    # Try ranks from a reasonable minimum up to n*m*p
    # Start with a lower bound estimate
    min_rank = 1
    
    for target_rank in range(min_rank, n * m * p + 1):
        if time.time() > deadline - 10:
            break
        
        # Try multiple random seeds
        for seed in range(10):
            if time.time() > deadline - 5:
                break
            
            result = als_decompose(target_rank, max_iters=300, seed=seed)
            if result is None:
                continue
            
            U, V, W = result
            
            # Verify
            if verify(U, V, W, target_rank):
                return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(dim_u)]
    V = [[0] * rank for _ in range(dim_v)]
    W = [[0] * rank for _ in range(dim_w)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
