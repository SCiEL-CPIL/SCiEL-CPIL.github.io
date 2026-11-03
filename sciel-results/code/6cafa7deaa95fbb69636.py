import numpy as np

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Trivial rank
    rank_trivial = n * m * p
    
    # Target tensor is all ones
    # T[i, j, k] = 1
    
    # We search for minimal rank
    # Start from a reasonable guess. 
    # For 3x4x4, let's try from 20 up to 48.
    # To be safe and fast, we might just return trivial if we don't find better quickly.
    
    best_solution = None
    best_rank = rank_trivial
    
    # Generate trivial solution
    U_triv = np.zeros((n * m, rank_trivial))
    V_triv = np.zeros((m * p, rank_trivial))
    W_triv = np.zeros((p * n, rank_trivial))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U_triv[u_idx, col] = 1.0
                V_triv[v_idx, col] = 1.0
                W_triv[w_idx, col] = 1.0
                col += 1
                
    best_solution = (U_triv, V_triv, W_triv)
    
    # Try to find lower rank
    # Using ALS (Alternating Least Squares)
    # Equation: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    def check_error(U, V, W):
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_val = U[i*m+j, :]
                    v_val = V[j*p+k, :]
                    w_val = W[k*n+i, :]
                    # sum of element-wise product
                    val = np.dot(u_val, v_val * w_val)
                    err += (val - 1.0)**2
        return np.sqrt(err)

    def round_to_half(X):
        return np.round(X * 2) / 2.0

    # Search ranks
    # Limit search to save time
    max_rank_to_try = 48
    min_rank_to_try = 20
    
    # Time limit check is hard inside, so keep iterations low
    for R in range(min_rank_to_try, max_rank_to_try + 1):
        found = False
        # A few restarts
        for restart in range(5):
            U = np.random.randn(n * m, R) * 0.1
            V = np.random.randn(m * p, R) * 0.1
            W = np.random.randn(p * n, R) * 0.1
            
            # ALS iterations
            for it in range(100):
                # Update U
                for i in range(n):
                    for j in range(m):
                        # Build matrix A for least squares: A[k, r] = V[j*p+k, r] * W[k*n+i, r]
                        # We want sum_r U[i*m+j, r] * A[k, r] = 1 for all k
                        A = np.zeros((p, R))
                        for k in range(p):
                            A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                        b = np.ones(p)
                        # Solve A x = b
                        try:
                            U[i*m+j, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                        except:
                            pass
                            
                # Update V
                for j in range(m):
                    for k in range(p):
                        # Build matrix B: B[i, r] = U[i*m+j, r] * W[k*n+i, r]
                        # sum_r V[j*p+k, r] * B[i, r] = 1 for all i
                        B = np.zeros((n, R))
                        for i in range(n):
                            B[i, :] = U[i*m+j, :] * W[k*n+i, :]
                        b = np.ones(n)
                        try:
                            V[j*p+k, :] = np.linalg.lstsq(B, b, rcond=None)[0]
                        except:
                            pass
                            
                # Update W
                for k in range(p):
                    for i in range(n):
                        # Build matrix C: C[j, r] = U[i*m+j, r] * V[j*p+k, r]
                        # sum_r W[k*n+i, r] * C[j, r] = 1 for all j
                        C = np.zeros((m, R))
                        for j in range(m):
                            C[j, :] = U[i*m+j, :] * V[j*p+k, :]
                        b = np.ones(m)
                        try:
                            W[k*n+i, :] = np.linalg.lstsq(C, b, rcond=None)[0]
                        except:
                            pass
                            
            # Check error
            err = check_error(U, V, W)
            if err < 1e-5:
                # Try rounding
                U_r = round_to_half(U)
                V_r = round_to_half(V)
                W_r = round_to_half(W)
                err_r = check_error(U_r, V_r, W_r)
                if err_r < 1e-5:
                    best_solution = (U_r, V_r, W_r)
                    best_rank = R
                    found = True
                    break
                else:
                    # If rounding fails, maybe keep floats? 
                    # Prompt says integer/half-integer. 
                    # If we can't round, we might need to continue or accept float if allowed? 
                    # "integer/half-integer entries" is a hard constraint.
                    # We will not accept float solution.
                    pass
            
        if found:
            break
            
    U, V, W = best_solution
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
