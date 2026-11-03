import numpy as np
import time

DIMENSIONS = [4, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 145.0  # 5 seconds margin

    # Schoolbook decomposition
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n * m, rank_schoolbook))
    V_sb = np.zeros((m * p, rank_schoolbook))
    W_sb = np.zeros((p * n, rank_schoolbook))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1

    # Function to verify a decomposition
    def verify(U, V, W, tol=1e-9):
        # U: (n*m, R), V: (m*p, R), W: (p*n, R)
        R = U.shape[1]
        if R > rank_schoolbook:
            return False
        # Check if entries are integer or half-integer
        for mat in [U, V, W]:
            if not np.allclose(mat * 2, np.round(mat * 2), atol=tol):
                return False
        # Check tensor identity
        # Tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # We need T[i,j,k] = 1 for all i,j,k
        # We can do this efficiently by iterating
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                    if not np.isclose(val, 1.0, atol=tol):
                        return False
        return True

    # Try to find a lower rank decomposition
    # We will try ranks from 119 down to, say, 110
    # Using a simple gradient descent / ALS approach
    
    def attempt_rank(R):
        if R >= rank_schoolbook:
            return None
            
        # Initialize with schoolbook columns + random noise
        # We need R columns. We can take the first R-1 schoolbook columns and a random one?
        # Better: Initialize U, V, W with random values, or schoolbook truncated.
        # Truncating schoolbook won't work directly.
        
        # Let's use a random initialization scaled appropriately
        np.random.seed(42)
        U = np.random.randn(n * m, R) * 0.1
        V = np.random.randn(m * p, R) * 0.1
        W = np.random.randn(p * n, R) * 0.1
        
        # ALS: Alternate fixing two matrices and solving for the third
        # This is a linear least squares problem for the fixed third matrix.
        # For fixed V, W, the equation for U is:
        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        # This is not linear in U if V and W are fixed?
        # U[i*m+j, r] * (V[j*p+k, r] * W[k*n+i, r]) = 1
        # For a fixed (i,j,k), this is linear in the vector U[i*m+j, :].
        # Let A_{(i,j), (j,k), (i,k), r} = V[j*p+k, r] * W[k*n+i, r].
        # Then sum_r U[i*m+j, r] * A_{(i,j), (j,k), (i,k), r} = 1.
        # This is a system of linear equations for each (i,j) row of U?
        # No, for each (i,j), the variable is U[i*m+j, :] (length R).
        # The equations involving U[i*m+j, :] are for all k.
        # There are p equations for each (i,j).
        # Variables: R.
        # If R > p, it's underdetermined.
        
        # Let's solve for U given V, W.
        # For each i, j:
        #   Let u = U[i*m+j, :]
        #   We need sum_{k} (sum_{r} u[r] * V[j*p+k, r] * W[k*n+i, r]) = ?
        #   Wait, the target is 1 for each k.
        #   So for each k, sum_r u[r] * (V[j*p+k, r] * W[k*n+i, r]) = 1.
        #   This is a linear system A u = b, where A is p x R.
        #   A[k, r] = V[j*p+k, r] * W[k*n+i, r].
        #   b[k] = 1.
        
        # We can solve this least squares problem for each (i,j).
        
        for it in range(50):
            if time.time() > deadline:
                return None
            
            # Update U
            for i in range(n):
                for j in range(m):
                    # Construct A for this i,j
                    A = np.zeros((p, R))
                    for k in range(p):
                        A[k, :] = V[j * p + k, :] * W[k * n + i, :]
                    b = np.ones(p)
                    # Solve A u = b
                    u, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                    U[i * m + j, :] = u
            
            # Update V
            for j in range(m):
                for k in range(p):
                    A = np.zeros((n, R)) # Equations for each i
                    for i in range(n):
                        A[i, :] = U[i * m + j, :] * W[k * n + i, :]
                    b = np.ones(n)
                    v, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                    V[j * p + k, :] = v
            
            # Update W
            for k in range(p):
                for i in range(n):
                    A = np.zeros((m, R)) # Equations for each j
                    for j in range(m):
                        A[j, :] = U[i * m + j, :] * V[j * p + k, :]
                    b = np.ones(m)
                    w, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                    W[k * n + i, :] = w
            
            # Check convergence / validity
            # Calculate error
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                        err += (val - 1.0)**2
            
            if err < 1e-8:
                # Try to snap to integers/half-integers
                U_int = np.round(U * 2) / 2.0
                V_int = np.round(V * 2) / 2.0
                W_int = np.round(W * 2) / 2.0
                
                if verify(U_int, V_int, W_int):
                    return U_int, V_int, W_int
            
            if time.time() > deadline:
                return None

    # Try ranks 119 down to 115
    for R in range(119, 114, -1):
        res = attempt_rank(R)
        if res:
            U, V, W = res
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback to schoolbook
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
