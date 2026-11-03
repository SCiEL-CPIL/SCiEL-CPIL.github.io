import time
import numpy as np

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the rank-minimal exact tensor decomposition for multiplying
    a 3x4 matrix by a 4x5 matrix.
    
    The schoolbook rank for M_{3,4,5} is 3*4*5 = 60.
    We attempt to find a sub-schoolbook decomposition (rank 58 or 59) 
    using a time-limited optimization search. If no better solution is found,
    we return the canonical schoolbook decomposition.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit

    # 1. Build the schoolbook decomposition as a fallback
    rank_schoolbook = n * m * p
    U_school = np.zeros((n * m, rank_schoolbook))
    V_school = np.zeros((m * p, rank_schoolbook))
    W_school = np.zeros((p * n, rank_schoolbook))

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1

    # Build the target tensor for verification
    # T[a, b, c] where a = i*m+j, b = j*p+k, c = k*n+i
    T_target = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1.0

    def verify(U, V, W, tol=1e-9):
        """Verify decomposition using numpy."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        
        # Compute the tensor decomposition
        # T_computed[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        T_computed = np.einsum('ar,br,cr->abc', U, V, W)
        
        return np.allclose(T_computed, T_target, atol=tol)

    def clean_matrix(M, tol=1e-7):
        """Clean a matrix by snapping near-zero and near-half-integer values."""
        M = np.array(M, dtype=float)
        M_clean = np.zeros_like(M)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                val = M[i, j]
                if abs(val) < tol:
                    M_clean[i, j] = 0.0
                else:
                    val2 = round(val * 2)
                    if abs(val * 2 - val2) < tol * 10:
                        M_clean[i, j] = val2 / 2.0
                    else:
                        M_clean[i, j] = val
        return M_clean

    best_rank = rank_schoolbook
    best_U = U_school.tolist()
    best_V = V_school.tolist()
    best_W = W_school.tolist()
    
    # Try to find a sub-schoolbook decomposition
    # Focus on ranks 58 and 59, as 58 is a known theoretical upper bound
    # and 59 is a close alternative.
    
    # We use a simple gradient descent with random restarts.
    # This is unlikely to find a global optimum for rank 58 from scratch,
    # but might find a rank 59 if one exists and is "close" to schoolbook,
    # or if we use a good initialization.
    
    # However, given the difficulty, we will primarily rely on the schoolbook
    # unless we find a solution quickly.
    
    for target_rank in [59, 58]:
        if time.time() > deadline - 10: # Leave 10s for cleanup
            break
            
        # Limit search time per rank
        rank_start_time = time.time()
        rank_deadline = rank_start_time + 5 # 5 seconds per rank
        
        found = False
        for trial in range(50): # 50 random restarts
            if time.time() > rank_deadline:
                break
            
            np.random.seed(target_rank * 100 + trial)
            
            # Initialize with small random values
            U = np.random.randn(n * m, target_rank) * 0.5
            V = np.random.randn(m * p, target_rank) * 0.5
            W = np.random.randn(p * n, target_rank) * 0.5
            
            # Optimize using gradient descent
            lr = 0.01
            for iteration in range(500):
                if time.time() > rank_deadline:
                    break
                
                T_computed = np.einsum('ar,br,cr->abc', U, V, W)
                residual = T_computed - T_target
                
                dU = np.einsum('abc,br,cr->ar', residual, V, W)
                dV = np.einsum('abc,ar,cr->br', residual, U, W)
                dW = np.einsum('abc,ar,br->cr', residual, U, V)
                
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
                
                lr *= 0.995
                
                # Early stopping if residual is small
                if np.linalg.norm(residual) < 1e-6:
                    break
            
            U_clean = clean_matrix(U)
            V_clean = clean_matrix(V)
            W_clean = clean_matrix(W)
            
            if verify(U_clean, V_clean, W_clean):
                best_rank = target_rank
                best_U = U_clean.tolist()
                best_V = V_clean.tolist()
                best_W = W_clean.tolist()
                found = True
                break
        
        if found:
            break

    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
