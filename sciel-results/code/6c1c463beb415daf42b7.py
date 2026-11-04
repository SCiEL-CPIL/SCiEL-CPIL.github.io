import numpy as np
import time

DIMENSIONS = [4, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Leave 20s safety margin

    # Build the target tensor T[i*m+j, j*p+k, k*n+i] = 1 for valid (i,j,k), 0 otherwise
    nonzero_entries = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                nonzero_entries.append((a, b, c))

    num_nonzero = len(nonzero_entries)
    nz_set = set(nonzero_entries)

    # Schoolbook decomposition (fallback)
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    fallback = schoolbook()

    # Verify a decomposition
    def verify(U, V, W, tol=1e-8):
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        
        # Check non-zero entries
        max_err = 0.0
        for (a, b, c) in nonzero_entries:
            val = np.dot(U[a, :], V[b, :] * W[c, :])
            max_err = max(max_err, abs(val - 1.0))
        
        if max_err > tol:
            return False, max_err
        
        # Check that invalid entries are zero
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    if (a, b, c) not in nz_set:
                        val = np.dot(U[a, :], V[b, :] * W[c, :])
                        max_err = max(max_err, abs(val))
        
        return max_err <= tol, max_err

    # Try to find a lower rank decomposition using ALS
    def try_als(r, max_iters=200, tol=1e-10, seed=0):
        if time.time() > deadline:
            return None
        
        rng = np.random.RandomState(seed)
        
        # Initialize with random structure
        U = rng.randn(n*m, r) * 0.5
        V = rng.randn(m*p, r) * 0.5
        W = rng.randn(p*n, r) * 0.5
        
        # ALS: For each mode, solve a linear system
        for iteration in range(max_iters):
            if time.time() > deadline:
                return None
            
            # Update U
            for a in range(n * m):
                i = a // m
                j = a % m
                # Equations: for k in 0..p-1
                # U[a, :] dot (V[j*p+k, :] * W[k*n+i, :]) = 1
                A = np.array([V[j*p+k, :] * W[k*n+i, :] for k in range(p)])
                b_vec = np.ones(p)
                # Solve least squares
                if A.shape[0] >= A.shape[1]:
                    U[a, :], residuals, rank, sv = np.linalg.lstsq(A, b_vec, rcond=None)
                else:
                    # Underdetermined: use pseudo-inverse
                    U[a, :] = np.linalg.pinv(A) @ b_vec
            
            # Update V
            for b in range(m * p):
                j = b // p
                k = b % p
                # Equations: for i in 0..n-1
                # V[b, :] dot (U[i*m+j, :] * W[k*n+i, :]) = 1
                A = np.array([U[i*m+j, :] * W[k*n+i, :] for i in range(n)])
                b_vec = np.ones(n)
                if A.shape[0] >= A.shape[1]:
                    V[b, :], residuals, rank, sv = np.linalg.lstsq(A, b_vec, rcond=None)
                else:
                    V[b, :] = np.linalg.pinv(A) @ b_vec
            
            # Update W
            for c in range(p * n):
                k = c // n
                i = c % n
                # Equations: for j in 0..m-1
                # W[c, :] dot (U[i*m+j, :] * V[j*p+k, :]) = 1
                A = np.array([U[i*m+j, :] * V[j*p+k, :] for j in range(m)])
                b_vec = np.ones(m)
                if A.shape[0] >= A.shape[1]:
                    W[c, :], residuals, rank, sv = np.linalg.lstsq(A, b_vec, rcond=None)
                else:
                    W[c, :] = np.linalg.pinv(A) @ b_vec
            
            # Check convergence
            if iteration > 0:
                # Quick check on a few entries
                err = 0.0
                for (a, b, c) in nonzero_entries[:10]:
                    val = np.dot(U[a, :], V[b, :] * W[c, :])
                    err += abs(val - 1.0)
                if err < tol:
                    break
        
        # Check if we found a valid solution
        valid, err = verify(U, V, W, tol=1e-6)
        if valid:
            # Convert to list format
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        return None

    # Try ranks from a reasonable lower bound upward
    best_result = None
    best_rank = n * m * p  # schoolbook rank

    # Try some candidate ranks
    # For 4x5x6, schoolbook is 120
    # Let's try to find something better
    candidate_ranks = list(range(30, 121))
    candidate_ranks.sort()

    for r in candidate_ranks:
        if time.time() > deadline - 10:
            break
        
        # Try ALS with this rank, using multiple seeds for robustness
        found = False
        for seed in range(3):
            if time.time() > deadline - 10:
                break
            result = try_als(r, max_iters=50, seed=seed)
            if result is not None:
                best_result = result
                best_rank = r
                found = True
                break
        
        if found:
            break  # Found a valid solution, stop

    if best_result is None:
        best_result = fallback
    
    return best_result
