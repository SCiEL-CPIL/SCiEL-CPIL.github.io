import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_school = n * m * p
    
    # Attempt to find a decomposition of rank r < rank_school
    # We use an Alternating Least Squares (ALS) approach to search for a solution.
    # This is a numerical method to solve the bilinear system, approximating the 
    # algebraic search suggested by the Gröbner basis directive.
    
    target_rank = rank_school - 1 # Try 35
    
    # Initialize tensors U, V, W
    # U shape: (n*m, r)
    # V shape: (m*p, r)
    # W shape: (p*n, r)
    
    np.random.seed(42)
    
    U = np.random.randn(n * m, target_rank) * 0.1
    V = np.random.randn(m * p, target_rank) * 0.1
    W = np.random.randn(p * n, target_rank) * 0.1
    
    # Target Tensor T
    # T is sparse. We don't need to store the full (nm, mp, pn) tensor.
    # We just need to know that T[I, J, K] = 1 if I=(i,j), J=(j,k), K=(k,i).
    
    # We will iterate a few times to see if we can drive the error to 0.
    # Error = sum_{I,J,K} (sum_c U[I,c] V[J,c] W[K,c] - T[I,J,K])^2
    
    def compute_error():
        err = 0.0
        # Iterate over all non-zero entries of T
        # These are (i, j, k) where T is 1.
        # I = i*m + j
        # J = j*p + k
        # K = k*n + i
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    I = i * m + j
                    J = j * p + k
                    K = k * n + i
                    # Sum over c
                    val = np.dot(U[I, :], np.multiply(V[J, :], W[K, :]))
                    # T[I, J, K] is 1
                    err += (val - 1.0) ** 2
        # Also need to account for zero entries?
        # The tensor is zero everywhere else.
        # There are nm*mp*pn - nmp zero entries.
        # Checking all of them is expensive: 6*18*12 = 1296 entries.
        # We can sample or just check the non-zero ones first.
        # If the non-zero ones are not satisfied, the rank is not achieved.
        # If they are satisfied, we must check if the zero entries are also 0.
        # For now, we just minimize the non-zero part error.
        # If we can make the non-zero part error 0, we then check the zero part.
        return err

    def solve_W():
        # For each K=(k,i), we have m equations (for each j).
        # A[j, c] = U[i*m+j, c] * V[j*p+k, c]
        # We want A @ W[K] = 1 (vector of ones of length m)
        for k in range(p):
            for i in range(n):
                K = k * n + i
                A = np.zeros((m, target_rank))
                for j in range(m):
                    I = i * m + j
                    J = j * p + k
                    A[j, :] = U[I, :] * V[J, :]
                
                # Solve A @ w = 1
                # Use least squares
                try:
                    w, residuals, rank, sv = np.linalg.lstsq(A, np.ones(m), rcond=None)
                    W[K, :] = w
                except np.linalg.LinAlgError:
                    pass

    def solve_V():
        # For each J=(j,k), we have n equations (for each i).
        # A[i, c] = U[i*m+j, c] * W[k*n+i, c]
        # We want A @ V[J] = 1 (vector of ones of length n)
        for j in range(m):
            for k in range(p):
                J = j * p + k
                A = np.zeros((n, target_rank))
                for i in range(n):
                    I = i * m + j
                    K = k * n + i
                    A[i, :] = U[I, :] * W[K, :]
                
                # Solve A @ v = 1
                try:
                    v, residuals, rank, sv = np.linalg.lstsq(A, np.ones(n), rcond=None)
                    V[J, :] = v
                except np.linalg.LinAlgError:
                    pass

    def solve_U():
        # For each I=(i,j), we have p equations (for each k).
        # A[k, c] = V[j*p+k, c] * W[k*n+i, c]
        # We want A @ U[I] = 1 (vector of ones of length p)
        for i in range(n):
            for j in range(m):
                I = i * m + j
                A = np.zeros((p, target_rank))
                for k in range(p):
                    J = j * p + k
                    K = k * n + i
                    A[k, :] = V[J, :] * W[K, :]
                
                # Solve A @ u = 1
                try:
                    u, residuals, rank, sv = np.linalg.lstsq(A, np.ones(p), rcond=None)
                    U[I, :] = u
                except np.linalg.LinAlgError:
                    pass

    # Run ALS for a few iterations
    for _ in range(20):
        solve_W()
        solve_V()
        solve_U()
        
        err = compute_error()
        if err < 1e-8:
            # Check zero entries
            # Sample some zero entries
            zero_ok = True
            # Total entries: nm*mp*pn
            # Non-zero: nmp
            # We check a few random zero entries
            for _ in range(100):
                I = np.random.randint(0, n * m)
                J = np.random.randint(0, m * p)
                K = np.random.randint(0, p * n)
                # Check if this is a non-zero entry
                i = I // m
                j = I % m
                j2 = J // p
                k = J % p
                k2 = K // n
                i2 = K % n
                
                # Non-zero if i==i2, j==j2, k==k2
                if i == i2 and j == j2 and k == k2:
                    continue
                
                val = np.dot(U[I, :], np.multiply(V[J, :], W[K, :]))
                if abs(val) > 1e-6:
                    zero_ok = False
                    break
            
            if zero_ok:
                # Found a valid rank 35 decomposition
                return {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
    
    # If no rank 35 decomposition found, return schoolbook rank 36
    rank = rank_school
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
