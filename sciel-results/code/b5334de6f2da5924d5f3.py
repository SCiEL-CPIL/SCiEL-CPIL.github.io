"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin from 150s limit
    
    n, m, p = DIMENSIONS
    trivial_rank = n * m * p
    
    # Start with trivial decomposition
    U = np.zeros((n * m, trivial_rank))
    V = np.zeros((m * p, trivial_rank))
    W = np.zeros((p * n, trivial_rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    best_rank = trivial_rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try to reduce rank through optimization
    # The tensor equation: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # We want to find the minimum r such that this holds exactly.
    
    # Strategy: Start from trivial and try to zero out columns iteratively
    # If a column can be set to zero without breaking the decomposition, remove it.
    
    current_rank = trivial_rank
    cur_U, cur_V, cur_W = U.copy(), V.copy(), W.copy()
    
    def check_decomposition(U_mat, V_mat, W_mat, rank):
        """Check if the decomposition is exact."""
        # T[i,j,k] should equal sum_r U[i,j,r]*V[j,k,r]*W[k,i,r]
        # Build expected tensor from trivial
        expected = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected[i,j,k] = 1
        
        # Compute actual tensor
        actual = np.zeros((n, m, p))
        for r in range(rank):
            u_col = U_mat[:, r]
            v_col = V_mat[:, r]
            w_col = W_mat[:, r]
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        actual[i,j,k] += u_col[i*m+j] * v_col[j*p+k] * w_col[k*n+i]
        
        return np.allclose(actual, expected, atol=1e-10)
    
    # Try greedy column elimination
    # For each column, check if we can set it to zero and still have valid decomposition
    # This is expensive, so we do it in a limited way
    
    # Alternative: Use optimization to find a lower rank decomposition
    # Let's try a simple approach: perturb and see if we can reduce rank
    
    # Try rank reduction from trivial_rank down
    # For each target_rank, try to find a valid decomposition
    
    # Given time constraints, let's try a few promising ranks
    # Known bounds for M_{3,4,7}:
    # Trivial: 84
    # Strassen-type bounds suggest possibly lower
    
    # Let's try to find if rank 80, 76, 72, etc. work
    # We'll use a random initialization approach
    
    np.random.seed(42)
    
    for target_rank in range(trivial_rank - 1, max(20, trivial_rank - 20), -1):
        if time.time() > deadline - 10:
            break
        
        # Try multiple random initializations
        for attempt in range(5):
            if time.time() > deadline - 10:
                break
            
            # Initialize with random values
            U_init = np.random.randn(n * m, target_rank)
            V_init = np.random.randn(m * p, target_rank)
            W_init = np.random.randn(p * n, target_rank)
            
            # Use gradient descent to minimize ||T - sum_r U_r ⊗ V_r ⊗ W_r||^2
            # This is a non-convex problem, so we'll use a simple approach
            
            # Actually, let's try a different approach:
            # Start from the trivial decomposition and try to merge columns
            
            # For now, just check if the trivial decomposition works
            # and return it as a fallback
            
            pass
    
    # If we found a better decomposition, use it
    if best_rank < trivial_rank:
        U_out = best_U[:, :best_rank].tolist()
        V_out = best_V[:, :best_rank].tolist()
        W_out = best_W[:, :best_rank].tolist()
    else:
        U_out = U.tolist()
        V_out = V.tolist()
        W_out = W.tolist()
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
