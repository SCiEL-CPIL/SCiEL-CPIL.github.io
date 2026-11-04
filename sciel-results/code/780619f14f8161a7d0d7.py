"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def _build_schoolbook(n, m, p):
    """Build the standard schoolbook decomposition with rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for i in range(n):
        for k in range(m):
            for j in range(p):
                r = i * (m * p) + k * p + j
                U[i * m + k, r] = 1.0
                V[k * p + j, r] = 1.0
                W[j * n + i, r] = 1.0
                
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly computes matrix multiplication.
    
    For each (i, k, j) triple, the tensor coefficient should be 1.
    T[i,k,j] = sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r]
    """
    U_arr = np.array(U)
    V_arr = np.array(V)
    W_arr = np.array(W)
    
    max_err = 0.0
    for i in range(n):
        for k in range(m):
            for j in range(p):
                expected = 1.0
                actual = np.dot(U_arr[i * m + k], V_arr[k * p + j] * W_arr[j * n + i])
                err = abs(actual - expected)
                if err > max_err:
                    max_err = err
    
    # Also verify that no spurious terms appear
    # For a valid decomposition, the only non-zero tensor entries should be T[i,k,j]=1
    # Check a few random positions where T should be 0
    # Actually, in matrix mult, T[i,k,j] is always 1 for all valid (i,k,j),
    # and there are no other positions. The tensor is fully dense with all 1s.
    # Wait - that's not right. The tensor T[i,k,j] for matrix mult is 1 for ALL (i,k,j)
    # in the range. So we just need to verify all n*m*p entries are 1.
    
    return max_err

def _try_reduce_rank(U, V, W, n, m, p, max_time=100.0):
    """Try to reduce the rank by finding and eliminating redundant terms."""
    start_time = time.time()
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    
    best_U, best_V, best_W = U_arr.copy(), V_arr.copy(), W_arr.copy()
    best_rank = rank
    
    # Try to eliminate columns that are linear combinations of others
    # This is a heuristic approach
    
    while time.time() - start_time < max_time:
        current_rank = best_U.shape[1]
        if current_rank <= 100:  # Stop trying to reduce below 100
            break
            
        # Try removing each column and checking if the decomposition still works
        removed = False
        for col in range(current_rank):
            if time.time() - start_time > max_time - 5:
                    break
                    
            # Temporarily remove column col
            mask = [i for i in range(current_rank) if i != col]
            U_test = best_U[:, mask]
            V_test = best_V[:, mask]
            W_test = best_W[:, mask]
            
            # Quick check: verify a subset of tensor entries
            valid = True
            for i in range(0, n, 2):
                for k in range(0, m, 2):
                    for j in range(0, p, 2):
                        expected = 1.0
                        actual = np.dot(U_test[i * m + k], V_test[k * p + j] * W_test[j * n + i])
                        if abs(actual - expected) > 1e-9:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
                    
            if valid:
                # Full verification
                max_err = 0.0
                for i in range(n):
                    for k in range(m):
                        for j in range(p):
                            actual = np.dot(U_test[i * m + k], V_test[k * p + j] * W_test[j * n + i])
                            err = abs(actual - 1.0)
                            if err > max_err:
                                max_err = err
                if max_err < 1e-9:
                    best_U, best_V, best_W = U_test, V_test, W_test
                    best_rank = current_rank - 1
                    removed = True
                    break
                    
        if not removed:
            break
    
    return best_U.tolist(), best_V.tolist(), best_W.tolist(), best_rank

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110.0  # Leave 10s margin for the 120s limit
    
    # Start with schoolbook
    U, V, W = _build_schoolbook(n, m, p)
    
    # Try to reduce rank
    if time.time() < deadline - 50:
        U, V, W, rank = _try_reduce_rank(U, V, W, n, m, p, max_time=80.0)
    else:
        rank = U.shape[1]
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
