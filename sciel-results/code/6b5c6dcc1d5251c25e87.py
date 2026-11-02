import numpy as np
import itertools

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    def schoolbook():
        r = n * m * p
        U = np.zeros((n * m, r), dtype=np.int32)
        V = np.zeros((m * p, r), dtype=np.int32)
        W = np.zeros((p * n, r), dtype=np.int32)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, idx] = 1
                    V[j * p + k, idx] = 1
                    W[k * n + i, idx] = 1
                    idx += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    def verify(U, V, W):
        # U, V, W are numpy arrays
        r = U.shape[1]
        # Check if U, V, W are half-integers
        for mat in [U, V, W]:
            if not np.allclose(mat * 2, np.round(mat * 2)):
                return False
        
        # Check exact decomposition
        # Condition: sum_t W[k*n+i, t] U[i*m+j, t] V[j'*p+k, t] == delta(j, j')
        for i in range(n):
            for j in range(m):
                for jp in range(m):
                    for k in range(p):
                        s = 0.0
                        for t in range(r):
                            s += W[k * n + i, t] * U[i * m + j, t] * V[jp * p + k, t]
                        target = 1.0 if j == jp else 0.0
                        if not np.isclose(s, target, atol=1e-9):
                            return False
        return True

    def try_rank(r, seed=0):
        np.random.seed(seed)
        # Initialize with small random integers or floats
        # Using floats for optimization, then rounding
        U = np.random.randn(n * m, r)
        V = np.random.randn(m * p, r)
        W = np.random.randn(p * n, r)
        
        # Normalize to prevent overflow/underflow
        for t in range(r):
            U[:, t] /= (np.linalg.norm(U[:, t]) + 1e-10)
            V[:, t] /= (np.linalg.norm(V[:, t]) + 1e-10)
            W[:, t] /= (np.linalg.norm(W[:, t]) + 1e-10)

        for _ in range(50): # 50 iterations of ALS
            # Update U
            for i in range(n):
                for j in range(m):
                    # Build A matrix (m*p, r)
                    A = np.zeros((m * p, r))
                    for jp in range(m):
                        for k in range(p):
                            for t in range(r):
                                A[jp * p + k, t] = W[k * n + i, t] * V[jp * p + k, t]
                    # Build b vector (m*p,)
                    b = np.zeros(m * p)
                    for jp in range(m):
                        if jp == j:
                            for k in range(p):
                                b[jp * p + k] = 1.0
                    # Solve A x = b
                    try:
                        U[i * m + j, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                    except:
                        pass

            # Update V
            for jp in range(m):
                for k in range(p):
                    A = np.zeros((n * m, r))
                    for i in range(n):
                        for j in range(m):
                            for t in range(r):
                                A[i * m + j, t] = W[k * n + i, t] * U[i * m + j, t]
                    b = np.zeros(n * m)
                    for i in range(n):
                        for j in range(m):
                            if j == jp:
                                b[i * m + j] = 1.0
                    try:
                        V[jp * p + k, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                    except:
                        pass

            # Update W
            for k in range(p):
                for i in range(n):
                    A = np.zeros((n * m, r))
                    for jp in range(m):
                        for j in range(m):
                            for t in range(r):
                                A[i * m + j, t] = U[i * m + j, t] * V[jp * p + k, t]
                    b = np.zeros(n * m)
                    for jp in range(m):
                        for j in range(m):
                            if j == jp:
                                b[i * m + j] = 1.0
                    try:
                        W[k * n + i, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                    except:
                        pass
        
        # Round to nearest 0.5
        U_rounded = np.round(U * 2) / 2
        V_rounded = np.round(V * 2) / 2
        W_rounded = np.round(W * 2) / 2
        
        if verify(U_rounded, V_rounded, W_rounded):
            return {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}
        return None

    # Search for minimal rank
    # Lower bound is 23 (Laderman for 3x3x3)
    # Upper bound is 45 (Schoolbook)
    for r in range(23, 45):
        # Try a few seeds for each rank
        for seed in range(10):
            res = try_rank(r, seed)
            if res:
                return res
        
    return schoolbook()
# EVOLVE-BLOCK-END
