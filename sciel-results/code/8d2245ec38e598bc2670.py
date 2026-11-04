import numpy as np
import time

DIMENSIONS = [4, 4, 9]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    # Precompute
    num_constraints = n * p * m * m
    
    # Try different ranks
    for rank in range(64, 145):
        if time.time() > deadline - 15:
            break
        result = try_alternating(n, m, p, rank, deadline)
        if result is not None:
            U, V, W = result
            U_list = [[float(U[i][j]) for j in range(rank)] for i in range(n*m)]
            V_list = [[float(V[i][j]) for j in range(rank)] for i in range(m*p)]
            W_list = [[float(W[i][j]) for j in range(rank)] for i in range(p*n)]
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback to schoolbook
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    U_list = [[float(U[i][j]) for j in range(rank)] for i in range(n*m)]
    V_list = [[float(V[i][j]) for j in range(rank)] for i in range(m*p)]
    W_list = [[float(W[i][j]) for j in range(rank)] for i in range(p*n)]
    return {"U": U_list, "V": V_list, "W": W_list}

def try_alternating(n, m, p, rank, deadline):
    """Try to find a rank-r decomposition using alternating least squares."""
    np.random.seed(999)
    
    U = np.random.randn(n*m, rank)
    V = np.random.randn(m*p, rank)
    W = np.random.randn(p*n, rank)
    
    for iteration in range(1000):
        if time.time() > deadline - 5:
            return None
        
        # Step 1: Fix U, V; solve for W
        for i in range(n):
            for k in range(p):
                # Build M matrix: M[(j1*m+j2), c] = U[i*m+j1, c] * V[j2*p+k, c]
                M = np.zeros((m*m, rank))
                for j1 in range(m):
                    for j2 in range(m):
                        M[j1*m+j2, :] = U[i*m+j1, :] * V[j2*p+k, :]
                b = np.zeros(m*m)
                for j in range(m):
                    b[j*m+j] = 1.0
                # Solve M * x = b (least squares)
                x, _, _, _ = np.linalg.lstsq(M, b, rcond=None)
                W[k*n+i, :] = x
        
        # Step 2: Fix U, W; solve for V
        for j2 in range(m):
            for k in range(p):
                M = np.zeros((m*m, rank))
                for i in range(n):
                    for j1 in range(m):
                        M[i*m+j1, :] = W[k*n+i, :] * U[i*m+j1, :]
                b = np.zeros(m*m)
                for i in range(n):
                    b[i*m+j2] = 1.0  # Wait, this isn't right
                # Actually, the constraint is:
                # sum_c W[k*n+i, c] * U[i*m+j1, c] * V[j2*p+k, c] = delta_{j1,j2}
                # For fixed j2, k: sum_c (W[k*n+i, c] * U[i*m+j1, c]) * V[j2*p+k, c] = delta_{j1,j2}
                # This is a system in V[j2*p+k, :]
                # For each i, j1: sum_c M[i*m+j1, c] * V[j2*p+k, c] = delta_{j1,j2}
                # But this needs to hold for ALL i, j1 simultaneously
                # So it's a system of n*m equations in rank unknowns
                # M has n*m rows
                M = np.zeros((n*m, rank))
                b = np.zeros(n*m)
                for i in range(n):
                    for j1 in range(m):
                        M[i*m+j1, :] = W[k*n+i, :] * U[i*m+j1, :]
                        b[i*m+j1] = 1.0 if j1 == j2 else 0.0
                x, _, _, _ = np.linalg.lstsq(M, b, rcond=None)
                V[j2*p+k, :] = x
        
        # Step 3: Fix V, W; solve for U
        for i in range(n):
            for j1 in range(m):
                M = np.zeros((n*p, rank))
                b = np.zeros(n*p)
                for k in range(p):
                    for j2 in range(m):
                        M[k*m+j2, :] = W[k*n+i, :] * V[j2*p+k, :]
                        b[k*m+j2] = 1.0 if j1 == j2 else 0.0
                x, _, _, _ = np.linalg.lstsq(M, b, rcond=None)
                U[i*m+j1, :] = x
        
        # Check residual
        max_res = 0.0
        for i in range(n):
            for k in range(p):
                for j1 in range(m):
                    for j2 in range(m):
                        val = np.sum(W[k*n+i, :] * U[i*m+j1, :] * V[j2*p+k, :])
                        target = 1.0 if j1 == j2 else 0.0
                        max_res = max(max_res, abs(val - target))
        
        if max_res < 1e-8:
            # Round to half-integers
            U_r = np.round(U * 2) / 2
            V_r = np.round(V * 2) / 2
            W_r = np.round(W * 2) / 2
            
            max_res2 = 0.0
            for i in range(n):
                for k in range(p):
                    for j1 in range(m):
                        for j2 in range(m):
                            val = np.sum(W_r[k*n+i, :] * U_r[i*m+j1, :] * V_r[j2*p+k, :])
                            target = 1.0 if j1 == j2 else 0.0
                            max_res2 = max(max_res2, abs(val - target))
            
            if max_res2 < 1e-9:
                return (U_r, V_r, W_r)
    
    return None
