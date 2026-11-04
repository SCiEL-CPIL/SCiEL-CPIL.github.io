import numpy as np
import time

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n
    
    T = np.zeros((dim_U, dim_V, dim_W), dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    def try_find_decomposition(rank, max_restarts=1000, deadline=None):
        """
        Tries to find a rank 'rank' decomposition by randomly initializing U and V,
        then solving for W using least squares and checking for exactness.
        """
        for r in range(max_restarts):
            if deadline and time.time() > deadline:
                return None
            
            # Initialize U and V with random small integers
            # Using {-1, 0, 1} to keep numbers small
            U = np.random.choice([-1, 0, 1], size=(dim_U, rank))
            V = np.random.choice([-1, 0, 1], size=(dim_V, rank))
            
            # Construct the matrix M where columns are vec(U[:,l] V[:,l]^T)
            # M shape: (dim_U * dim_V, rank)
            M = np.zeros((dim_U * dim_V, rank))
            for l in range(rank):
                M[:, l] = np.outer(U[:, l], V[:, l]).flatten()
            
            # We want to solve M @ W_flat = T_flat
            # T is (dim_U, dim_V, dim_W)
            # We can solve for each k in dim_W separately
            # T[:, :, k] flattened is b_k
            # M @ W[:, k] = b_k
            
            W = np.zeros((dim_W, rank))
            valid = True
            
            for k in range(dim_W):
                b_k = T[:, :, k].flatten()
                # Solve M @ x = b_k
                try:
                    x, residuals, rank_m, s = np.linalg.lstsq(M, b_k, rcond=None)
                    # Check if residual is small
                    if residuals is not None and len(residuals) > 0:
                        if residuals[0] > 1e-9:
                            valid = False
                            break
                    else:
                        # Check residual directly
                        res = np.linalg.norm(M @ x - b_k)
                        if res > 1e-9:
                            valid = False
                            break
                    W[k, :] = x
                except np.linalg.LinAlgError:
                    valid = False
                    break
            
            if not valid:
                continue
                
            # We have a candidate W
            # Verify exactness with integer rounding
            # U, V are integers. W might be rational.
            # Check if W can be rounded to half-integers or integers
            # And if the decomposition is exact.
            
            # First, check if the current float solution is exact
            T_recon = np.zeros_like(T)
            for l in range(rank):
                T_recon += np.outer(U[:, l], np.outer(V[:, l], W[:, l])).reshape(dim_U, dim_V, dim_W)
            
            if np.linalg.norm(T_recon - T) < 1e-9:
                # Try to round W to half-integers
                W_rounded = np.round(W * 2) / 2
                T_recon_rounded = np.zeros_like(T)
                for l in range(rank):
                    T_recon_rounded += np.outer(U[:, l], np.outer(V[:, l], W_rounded[:, l])).reshape(dim_U, dim_V, dim_W)
                
                if np.linalg.norm(T_recon_rounded - T) < 1e-9:
                    return U, V, W_rounded
            
            # If float W is not exact after rounding, try to find rational W?
            # For now, accept if float is exact (though problem requires int/half-int)
            # But the problem says "integer/half-integer entries".
            # If W is not half-integer, we need to find a better one or reject.
            # However, if U, V are integers, and T is integer, W from lstsq should be rational.
            # If the system is consistent, W is determined.
            # We can check if W is close to half-integers.
            
            # If not close, this random U, V might not lead to a valid half-integer decomposition.
            # But it is a valid float decomposition.
            # The problem requires integer/half-integer.
            # So we must reject if W is not close to half-integers.
            
            # Check distance to half-integers
            dist = np.abs(W - np.round(W * 2) / 2)
            if np.max(dist) < 1e-6:
                return U, V, np.round(W * 2) / 2

        return None

    start_time = time.time()
    deadline = start_time + 140
    
    # Try ranks from 20 to 30
    for rank in range(20, 31):
        if time.time() > deadline - 10:
            break
            
        result = try_find_decomposition(rank, max_restarts=200, deadline=deadline)
        if result:
            U, V, W = result
            # Final verification
            T_recon = np.zeros_like(T)
            for l in range(rank):
                T_recon += np.outer(U[:, l], np.outer(V[:, l], W[:, l])).reshape(dim_U, dim_V, dim_W)
            if np.linalg.norm(T_recon - T) < 1e-9:
                return {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
                
    # Fallback to schoolbook
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
                
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
