import numpy as np
import time

DIMENSIONS = [2, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Target tensor T is 1 for all i,j,k
    # We want sum_c U[i,j,c] V[j,k,c] W[k,i,c] = 1
    
    # Try ranks from lower bound upwards
    lower_bound = max(n*m, m*p, p*n)
    
    for r in range(lower_bound, n*m*p + 1):
        if time.time() > deadline - 5:
            break
            
        # Initialize U, V, W
        # U: (n*m, r), V: (m*p, r), W: (p*n, r)
        # Use small random initialization
        np.random.seed(42)
        U = np.random.randn(n*m, r) * 0.1
        V = np.random.randn(m*p, r) * 0.1
        W = np.random.randn(p*n, r) * 0.1
        
        converged = False
        for it in range(1000):
            if time.time() > deadline - 5:
                break
                
            # Update U
            # For each i, j, solve A u = t
            for i in range(n):
                for j in range(m):
                    # A is (p, r)
                    # A[k, c] = V[j*p+k, c] * W[k*n+i, c]
                    A = np.zeros((p, r))
                    for k in range(p):
                        A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    # t is (p,)
                    t = np.ones(p)
                    # Solve A u = t
                    # Least squares
                    try:
                        u, residuals, rank, s = np.linalg.lstsq(A, t, rcond=None)
                        U[i*m+j, :] = u
                    except:
                        pass
                        
            # Update V
            for j in range(m):
                for k in range(p):
                    # A is (n, r)
                    # A[i, c] = U[i*m+j, c] * W[k*n+i, c]
                    A = np.zeros((n, r))
                    for i in range(n):
                        A[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    t = np.ones(n)
                    try:
                        v, residuals, rank, s = np.linalg.lstsq(A, t, rcond=None)
                        V[j*p+k, :] = v
                    except:
                        pass
                        
            # Update W
            for k in range(p):
                for i in range(n):
                    # A is (m, r)
                    # A[j, c] = U[i*m+j, c] * V[j*p+k, c]
                    A = np.zeros((m, r))
                    for j in range(m):
                        A[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    t = np.ones(m)
                    try:
                        w, residuals, rank, s = np.linalg.lstsq(A, t, rcond=None)
                        W[k*n+i, :] = w
                    except:
                        pass
            
            # Check residual
            # Calculate error
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                        err += (val - 1.0)**2
            if err < 1e-10:
                converged = True
                break
        
        if converged:
            # Round to half-integers
            U_rounded = np.round(U * 2) / 2
            V_rounded = np.round(V * 2) / 2
            W_rounded = np.round(W * 2) / 2
            
            # Verify rounded version
            err_rounded = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U_rounded[i*m+j, :], V_rounded[j*p+k, :] * W_rounded[k*n+i, :])
                        err_rounded += (val - 1.0)**2
                        
            if err_rounded < 1e-10:
                # Convert to lists of lists
                U_list = U_rounded.tolist()
                V_list = V_rounded.tolist()
                W_list = W_rounded.tolist()
                return {"U": U_list, "V": V_list, "W": W_list}
            
            # If rounding failed, maybe try without rounding if allowed?
            # Prompt says "integer/half-integer entries".
            # So we must round.
            
    # If no better rank found, return schoolbook
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
    return {"U": U, "V": V, "W": W}
