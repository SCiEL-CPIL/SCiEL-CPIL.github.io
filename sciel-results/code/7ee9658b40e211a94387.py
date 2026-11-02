"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill
    
    # Trivial decomposition: rank = n*m*p = 75
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    # Verify trivial decomposition
    def verify(U, V, W, n, m, p, tol=1e-9):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    if not verify(U, V, W, n, m, p):
        raise ValueError("Trivial decomposition failed")
    
    # Attempt rank reduction via column elimination
    # This is a greedy approach: try removing columns one at a time
    def check_valid(cols, U, V, W, n, m, p, tol=1e-8):
        """Check if decomposition with only specified columns is valid."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in cols:
                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = rank
    active_cols = list(range(rank))
    
    # Multiple passes of greedy elimination
    for _pass in range(5):
        if time.time() - start_time > deadline - 15:
            break
        removed_any = False
        for c in list(active_cols):
            if time.time() - start_time > deadline - 15:
                break
            test_cols = [x for x in active_cols if x != c]
            if check_valid(test_cols, U, V, W, n, m, p):
                active_cols = test_cols
                removed_any = True
                best_rank = len(active_cols)
                best_U = U[:, active_cols].copy()
                best_V = V[:, active_cols].copy()
                best_W = W[:, active_cols].copy()
        if not removed_any:
            break
    
    # Final safety check
    if time.time() - start_time > deadline:
        # Return trivial decomposition as fallback
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
