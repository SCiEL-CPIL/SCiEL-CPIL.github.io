import time
import numpy as np

DIMENSIONS = [3, 3, 7]

def _schoolbook(n, m, p):
    """Construct the standard rank n*m*p tensor decomposition for M_{n,m,p}."""
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1.0
                V[j * p + k][c] = 1.0
                W[k * n + i][c] = 1.0
                c += 1
    return {"U": U, "V": V, "W": W}

def _verify(n, m, p, U, V, W, tol=1e-9):
    """
    Verify that U, V, W form a valid tensor decomposition of M_{n,m,p}.
    """
    r = len(U[0]) if U else 0
    if r != len(V[0]) or r != len(W[0]):
        return False
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    
    U_mat = np.array(U)
    V_mat = np.array(V)
    W_mat = np.array(W)

    # Check all (i,j,k) on-diagonal terms
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                # sum_r U[a,r] V[b,r] W[c,r]
                val = np.dot(U_mat[a], V_mat[b] * W_mat[c])
                if abs(val - 1.0) > tol:
                    return False

    # Check a subset of off-diagonal terms to ensure they are 0
    # We check all combinations to be safe, as n*m, m*p, p*n are small enough
    # 9 * 21 * 21 = 3969 checks. Each check is a dot product of length R.
    # R is at most 63. 3969 * 63 ops is ~250k ops. Very fast.
    valid_combos = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                valid_combos.add((i*m+j, j*p+k, k*n+i))

    for a in range(n * m):
        for b in range(m * p):
            for c in range(p * n):
                if (a, b, c) in valid_combos:
                    continue
                val = np.dot(U_mat[a], V_mat[b] * W_mat[c])
                if abs(val) > tol:
                    return False
    return True

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    best_result = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try to find a lower rank decomposition using ALS
    # We iterate rank R from best_rank - 1 down to a lower bound
    # Lower bound heuristic: max(n*p, m*p, n*m) is not tight.
    # For 3x3x7, let's try down to 40.
    
    lower_bound = 40
    
    for R in range(best_rank - 1, lower_bound - 1, -1):
        if time.time() > deadline:
            break
            
        # Try a few random initializations for this rank
        for trial in range(5):
            if time.time() > deadline:
                break
                
            np.random.seed(42 + trial + R)
            
            # Initialize with small random values
            U = np.random.randn(n * m, R) * 0.1
            V = np.random.randn(m * p, R) * 0.1
            W = np.random.randn(p * n, R) * 0.1
            
            # ALS iterations
            converged = False
            prev_err = float('inf')
            
            for iteration in range(100):
                if time.time() > deadline:
                    break
                
                # Update U
                for i in range(n):
                    for j in range(m):
                        a = i * m + j
                        # Build matrix M (p x R)
                        # M[k, r] = V[j*p+k, r] * W[k*n+i, r]
                        M = np.zeros((p, R))
                        for k in range(p):
                            b = j * p + k
                            c = k * n + i
                            M[k, :] = V[b, :] * W[c, :]
                        # Solve M @ u = ones(p)
                        try:
                            u, _, _, _ = np.linalg.lstsq(M, np.ones(p), rcond=None)
                            U[a, :] = u
                        except:
                            pass
                
                # Update V
                for j in range(m):
                    for k in range(p):
                        b = j * p + k
                        # Build matrix M (n x R)
                        # M[i, r] = U[i*m+j, r] * W[k*n+i, r]
                        M = np.zeros((n, R))
                        for i in range(n):
                            a = i * m + j
                            c = k * n + i
                            M[i, :] = U[a, :] * W[c, :]
                        # Solve M @ v = ones(n)
                        try:
                            v, _, _, _ = np.linalg.lstsq(M, np.ones(n), rcond=None)
                            V[b, :] = v
                        except:
                            pass
                
                # Update W
                for k in range(p):
                    for i in range(n):
                        c = k * n + i
                        # Build matrix M (m x R)
                        # M[j, r] = U[i*m+j, r] * V[j*p+k, r]
                        M = np.zeros((m, R))
                        for j in range(m):
                            a = i * m + j
                            b = j * p + k
                            M[j, :] = U[a, :] * V[b, :]
                        # Solve M @ w = ones(m)
                        try:
                            w, _, _, _ = np.linalg.lstsq(M, np.ones(m), rcond=None)
                            W[c, :] = w
                        except:
                            pass
                
                # Check convergence / error
                # Simple error check: sum of squares of residuals
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        a = i * m + j
                        for k in range(p):
                            b = j * p + k
                            c = k * n + i
                            val = np.dot(U[a], V[b] * W[c])
                            err += (val - 1.0) ** 2
                
                if abs(err - prev_err) < 1e-8:
                    converged = True
                    break
                prev_err = err
                
                if err < 1e-6:
                    converged = True
                    break
            
            if converged or err < 1e-5:
                # Round to half-integers
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                # Verify exactness
                if _verify(n, m, p, U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist()):
                    return {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}
                    
    return best_result
# EVOLVE-BLOCK-END
