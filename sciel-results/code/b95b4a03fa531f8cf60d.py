"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 4
    deadline = time.time() + 100  # leave safety margin before 120s kill

    # Try to find a rank-15 decomposition via alternating least squares
    # The tensor T has entries T[i*m+j, j*p+k, k*n+i] = 1 for all valid indices
    # We want: sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] = 1
    
    def verify(U, V, W, n, m, p, rank):
        """Check if the decomposition is exact (within tolerance)."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for t in range(rank):
                        s += U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def verify_np(U, V, W, n, m, p, rank):
        """Vectorized verification."""
        for i in range(n):
            for j in range(m):
                # For fixed i,j: sum_k sum_t U[i*m+j,t] * V[j*p+k,t] * W[k*n+i,t] should be 1 for each k
                # Actually each (i,j,k) triple must equal 1
                for k in range(p):
                    s = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def try_als(rank, max_iter=500, tol=1e-12, seed=42):
        """Alternating least squares to find rank decomposition."""
        rng = np.random.RandomState(seed)
        
        # Initialize
        U = rng.randn(n * m, rank) * 0.1
        V = rng.randn(m * p, rank) * 0.1
        W = rng.randn(p * n, rank) * 0.1
        
        for iteration in range(max_iter):
            if time.time() > deadline:
                return None
            
            # Solve for W: fix U, V
            W_new = np.zeros((p * n, rank))
            for k in range(p):
                for i in range(n):
                    w = k * n + i
                    # Equations: for j in 0..m-1
                    # sum_t U[i*m+j,t] * V[j*p+k,t] * W[w,t] = 1
                    A = np.zeros((m, rank))
                    for j in range(m):
                        A[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    b = np.ones(m)
                    try:
                        W_new[w, :], _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    except np.linalg.LinAlgError:
                        W_new[w, :] = rng.randn(rank) * 0.1
            
            # Solve for U: fix V, W
            U_new = np.zeros((n * m, rank))
            for i in range(n):
                for j in range(m):
                    u = i * m + j
                    # Equations: for k in 0..p-1
                    # sum_t U[u,t] * V[j*p+k,t] * W[k*n+i,t] = 1
                    A = np.zeros((p, rank))
                    for k in range(p):
                        A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    b = np.ones(p)
                    try:
                        U_new[u, :], _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    except np.linalg.LinAlgError:
                        U_new[u, :] = rng.randn(rank) * 0.1
            
            # Solve for V: fix U, W
            V_new = np.zeros((m * p, rank))
            for j in range(m):
                for k in range(p):
                    v = j * p + k
                    # Equations: for i in 0..n-1
                    # sum_t U[i*m+j,t] * V[v,t] * W[k*n+i,t] = 1
                    A = np.zeros((n, rank))
                    for i in range(n):
                        A[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    b = np.ones(n)
                    try:
                        V_new[v, :], _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    except np.linalg.LinAlgError:
                        V_new[v, :] = rng.randn(rank) * 0.1
            
            # Check convergence
            diff = np.max(np.abs(W_new - W)) + np.max(np.abs(U_new - U)) + np.max(np.abs(V_new - V))
            W, U, V = W_new, U_new, V_new
            
            if diff < tol:
                # Verify
                if verify_np(U, V, W, n, m, p, rank):
                    return U, V, W
        
        # Final check even if not fully converged
        if verify_np(U, V, W, n, m, p, rank):
            return U, V, W
        
        return None

    # Try rank 15 with multiple seeds
    for seed in range(20):
        if time.time() > deadline:
            break
        result = try_als(15, max_iter=300, tol=1e-12, seed=seed)
        if result is not None:
            U, V, W = result
            # Convert to lists with rounded values (should be near-integer or half-integer)
            U_list = [[round(float(x), 10) for x in row] for row in U]
            V_list = [[round(float(x), 10) for x in row] for row in V]
            W_list = [[round(float(x), 10) for x in row] for row in W]
            return {"U": U_list, "V": V_list, "W": W_list}

    # Fallback: schoolbook rank 24
    rank = n * m * p  # 24
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
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
