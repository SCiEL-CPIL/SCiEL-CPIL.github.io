"""Rank-minimal tensor decomposition for 3x3x4 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def _target_tensor(n, m, p):
    """Build the 3D target tensor T[i, j, k] = 1 for all i, j, k."""
    T = np.zeros((n, m, p), dtype=np.int64)
    T[:, :, :] = 1
    return T

def _verify(U, V, W, n, m, p):
    """Check if U, V, W form a valid rank decomposition of the target tensor."""
    T = _target_tensor(n, m, p)
    T_flat = T.reshape(-1)
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    # Decomposition: sum_c U[:, c] * V[:, c] * W[:, c]
    # Reshape to 3D and check
    R = U.shape[1]
    recon = np.zeros((n, m, p), dtype=np.float64)
    for c in range(R):
        uc = U[:, c].reshape(n, m)
        vc = V[:, c].reshape(m, p)
        wc = W[:, c].reshape(p, n)
        # term = uc * vc * wc
        # Need to be careful about dimensions
        # uc is (n, m), vc is (m, p), wc is (p, n)
        # The term is a 3D tensor where T[i,j,k] += uc[i,j] * vc[j,k] * wc[k,i]
        for i in range(n):
            for j in range(m):
                if uc[i, j] != 0:
                    for k in range(p):
                        recon[i, j, k] += uc[i, j] * vc[j, k] * wc[k, i]
    # Check exactness
    return np.allclose(recon, T_flat.reshape(n, m, p), atol=1e-9)

def _try_rank(n, m, p, target_rank, max_iter=50, time_limit=10.0):
    """
    Attempt to find a decomposition of the given rank using iterative randomized search.
    This is a heuristic and may not always succeed.
    """
    start = time.time()
    T = _target_tensor(n, m, p)
    
    # Initialize random matrices
    # Using half-integers: multiply by 2 to get integers, then divide by 2
    U = np.random.randint(-10, 11, (n * m, target_rank)) / 2.0
    V = np.random.randint(-10, 11, (m * p, target_rank)) / 2.0
    W = np.random.randint(-10, 11, (p * n, target_rank)) / 2.0
    
    best_err = np.inf
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for _ in range(max_iter):
        if time.time() - start > time_limit:
            break
        
        # Check current error
        recon = np.zeros((n, m, p), dtype=np.float64)
        for c in range(target_rank):
            uc = U[:, c].reshape(n, m)
            vc = V[:, c].reshape(m, p)
            wc = W[:, c].reshape(p, n)
            for i in range(n):
                for j in range(m):
                    if uc[i, j] != 0:
                        for k in range(p):
                            recon[i, j, k] += uc[i, j] * vc[j, k] * wc[k, i]
        err = np.linalg.norm(recon - T)
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if err < 1e-9:
            # Found a valid decomposition
            # Convert to exact half-integers if possible
            # For simplicity, return as is if close enough
            return best_U.tolist(), best_V.tolist(), best_W.tolist()
        
        # Random perturbation
        noise = np.random.normal(0, 0.5, U.shape)
        U = U + noise
        V = V + np.random.normal(0, 0.5, V.shape)
        W = W + np.random.normal(0, 0.5, W.shape)
    
    # Return best found if it's close enough
    if best_err < 1e-6:
        return best_U.tolist(), best_V.tolist(), best_W.tolist()
    return None

def _schoolbook(n, m, p):
    """Standard schoolbook decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    deadline = 120.0
    start_time = time.time()
    
    # Try to find lower rank decompositions
    # Known lower bounds: rank >= max(n, m, p) * min(n, m, p) / something
    # For 3x3x4, schoolbook is 36.
    # Let's try ranks from 20 to 35
    for target_rank in range(20, 36):
        if time.time() - start_time > deadline - 5:
            break
        result = _try_rank(n, m, p, target_rank, max_iter=30, time_limit=deadline - 5 - (time.time() - start_time))
        if result is not None:
            U, V, W = result
            # Verify
            if _verify(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    U, V, W = _schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
