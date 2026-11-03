"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition - this is the baseline
    # We'll try to find a better one, but fall back to schoolbook if needed
    rank_schoolbook = n * m * p
    
    # Build schoolbook first as a safe fallback
    U_sb = np.zeros((n * m, rank_schoolbook))
    V_sb = np.zeros((m * p, rank_schoolbook))
    W_sb = np.zeros((p * n, rank_schoolbook))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    # Convert to list format for return
    def to_list(mat):
        return mat.tolist()
    
    # Try to improve using known fast matrix multiplication algorithms
    # For 4x5x6, we can try block decomposition
    
    # Strategy: Use Strassen for 2x2 blocks where possible
    # The tensor T[i,j,k] for multiplying A(n x m) by B(m x p) to get C(n x p)
    # T[i*m+j, j*p+k, k*n+i] = 1
    
    # Let's try a rank minimization approach using random initialization and gradient descent
    # on the tensor decomposition, but this might be slow.
    
    # Alternative: Try known decompositions for rectangular matrices
    # For 4x5x6, one approach is to use the fact that we can embed into larger known decompositions
    
    # Let's try a direct approach: use the rank of the tensor via numerical methods
    # But this is computationally expensive.
    
    # For now, let's try a few specific strategies:
    # 1. Schoolbook (rank = 120)
    # 2. Try to find a better decomposition
    
    # Let's implement a simple iterative refinement starting from schoolbook
    # and trying to merge/eliminate terms
    
    # Actually, let's try a different approach: use the Laderman or Strassen-based
    # block decomposition
    
    # For 4x5x6:
    # We can think of A as 4x5, B as 5x6, C as 4x6
    # Split A into 2 blocks of 2x5, B into 2 blocks of 5x3, C into 2x3 blocks
    # C_11 = A_1*B_1 (2x2 * 2x3)
    # C_12 = A_1*B_2 (2x2 * 2x3)  
    # C_21 = A_2*B_1 (2x2 * 2x3)
    # C_22 = A_2*B_2 (2x2 * 2x3)
    
    # For 2x2 * 2x3, the schoolbook rank is 2*2*3 = 12
    # But we can use Strassen-like methods
    
    # Actually, let's try a more systematic approach using the tensor's structure
    
    # The key insight: for matrix multiplication, the tensor is "bilinear" and has
    # a specific structure that can be exploited
    
    # Let's try to find a decomposition using random linear combinations and then
    # try to reduce the rank
    
    # Given the time constraint, let's implement a practical approach:
    # Start with schoolbook and try to find linear dependencies in the rank-1 terms
    
    # For each term (i,j,k), we have U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1
    # Two terms (i,j,k) and (i',j',k') can potentially be merged if the resulting
    # decomposition still satisfies all constraints
    
    # This is complex. Let's instead try a numerical optimization approach with a time limit
    
    deadline = time.time() + 100  # Leave 20s buffer for the 120s limit
    
    # Try to find a better decomposition using iterative methods
    best_rank = rank_schoolbook
    best_U, best_V, best_W = U_sb.copy(), V_sb.copy(), W_sb.copy()
    
    # Try a few random initializations and gradient descent
    np.random.seed(42)
    
    for rank_attempt in [100, 90, 85, 80, 75, 70, 65, 60]:
        if time.time() > deadline:
            break
            
        # Initialize randomly
        U_init = np.random.randn(n * m, rank_attempt) * 0.1
        V_init = np.random.randn(m * p, rank_attempt) * 0.1
        W_init = np.random.randn(p * n, rank_attempt) * 0.1
        
        # Try to optimize: minimize sum over all (i,j,k) of (sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] - 1)^2
        # Use a simple iterative approach
        
        lr = 0.01
        for it in range(500):
            if time.time() > deadline:
                break
                
            # Compute current values and errors
            errors = np.zeros((n, m, p))
            dU = np.zeros_like(U_init)
            dV = np.zeros_like(V_init)
            dW = np.zeros_like(W_init)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U_init[i*m+j], V_init[j*p+k] * W_init[k*n+i])
                        err = val - 1
                        errors[i, j, k] = err
                        # Gradient for U[i*m+j, c] = err * V[j*p+k, c] * W[k*n+i, c]
                        dU[i*m+j] += err * V_init[j*p+k] * W_init[k*n+i]
                        dV[j*p+k] += err * U_init[i*m+j] * W_init[k*n+i]
                        dW[k*n+i] += err * U_init[i*m+j] * V_init[j*p+k]
            
            # Update
            U_init -= lr * dU
            V_init -= lr * dV
            W_init -= lr * dW
            
            # Check convergence
            if np.max(np.abs(errors)) < 1e-6:
                break
        
        if time.time() > deadline:
            break
        
        # Check if we found a good solution
        max_err = np.max(np.abs(errors))
        if max_err < 1e-4:
            # Round to rational numbers if possible
            # For now, just check if it's valid
            best_rank = rank_attempt
            best_U, best_V, best_W = U_init, V_init, W_init
            break
    
    # If we didn't find a better one, use schoolbook
    if best_rank >= rank_schoolbook:
        best_U, best_V, best_W = U_sb, V_sb, W_sb
        best_rank = rank_schoolbook
    
    # Convert to lists
    U_list = to_list(best_U)
    V_list = to_list(best_V)
    W_list = to_list(best_W)
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
