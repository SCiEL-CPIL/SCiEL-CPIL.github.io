import numpy as np
import time
import random

DIMENSIONS = [3, 3, 3]

def verify_decomposition(U, V, W, n, m, p):
    """
    Verify if U, V, W exactly decompose the matrix multiplication tensor.
    U: (n*m, R)
    V: (m*p, R)
    W: (p*n, R)
    """
    R = U.shape[1]
    # Reshape U, V for easier manipulation
    # U[i*m+j, c] -> U_mat[i, j, c]
    U_mat = np.array(U).reshape(n, m, R)
    V_mat = np.array(V).reshape(m, p, R)
    W_mat = np.array(W).reshape(p, n, R)
    
    # Test with random integer matrices
    random.seed(42)
    for _ in range(20):
        A = np.array(random.randint(-2, 2) for _ in range(n*m)).reshape(n, m)
        B = np.array(random.randint(-2, 2) for _ in range(m*p)).reshape(m, p)
        C = A @ B
        
        # Compute decomposition
        C_dec = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                total = 0
                for c in range(R):
                    # Term for rank c
                    # sum_j A[i,j] U[i*m+j, c]
                    term_u = 0
                    for j in range(m):
                        term_u += A[i, j] * U_mat[i, j, c]
                    # sum_j B[j,k] V[j*p+k, c]
                    term_v = 0
                    for j in range(m):
                        term_v += B[j, k] * V_mat[j, k, c]
                    total += W_mat[k, i, c] * term_u * term_v
                C_dec[i, k] = total
                
                if C[i, k] != C_dec[i, k]:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 140s limit
    
    # Fallback: Schoolbook rank 27
    def get_schoolbook():
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
    
    # Try to find lower rank via optimization
    # We'll try ranks 19, 20, 21
    # Note: Exact integer/half-integer solutions are rare from random optimization.
    # We will attempt to find a solution and snap to 0.5/1.0.
    
    target_ranks = [19, 20, 21]
    
    for R in target_ranks:
        if time.time() > deadline - 20: # Need 20s for verification/fallback
            break
            
        # Initialize randomly
        U = np.random.randint(0, 2, (n*m, R)) * 0.5
        V = np.random.randint(0, 2, (m*p, R)) * 0.5
        W = np.random.randint(0, 2, (p*n, R)) * 0.5
        
        # Optimization
        # We need to minimize error for a fixed set of A, B
        random.seed(123)
        test_As = []
        test_Bs = []
        for _ in range(5):
            A = np.array(random.randint(-1, 1) for _ in range(n*m)).reshape(n, m)
            B = np.array(random.randint(-1, 1) for _ in range(m*p)).reshape(m, p)
            test_As.append(A)
            test_Bs.append(B)
            
        def cost_function(params):
            U_flat, V_flat, W_flat = np.split(params, [n*m*R, n*m*R + m*p*R])
            U_c = U_flat.reshape(n*m, R)
            V_c = V_flat.reshape(m*p, R)
            W_c = W_flat.reshape(p*n, R)
            
            U_mat = U_c.reshape(n, m, R)
            V_mat = V_c.reshape(m, p, R)
            W_mat = W_c.reshape(p, n, R)
            
            total_err = 0.0
            for idx, (A, B) in enumerate(zip(test_As, test_Bs)):
                C = A @ B
                for i in range(n):
                    for k in range(p):
                        target = C[i, k]
                        calc = 0.0
                        for c in range(R):
                            term_u = 0.0
                            for j in range(m):
                                term_u += A[i, j] * U_mat[i, j, c]
                            term_v = 0.0
                            for j in range(m):
                                term_v += B[j, k] * V_mat[j, k, c]
                            calc += W_mat[k, i, c] * term_u * term_v
                        total_err += (target - calc) ** 2
            return total_err

        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        # Use a fast local optimizer
        try:
            from scipy.optimize import minimize
            res = minimize(cost_function, x0, method='L-BFGS-B', options={'maxiter': 50, 'maxfun': 200})
        except Exception:
            continue
            
        if time.time() > deadline - 10:
            continue
            
        # Check result
        if res.success and res.fun < 1e-6:
            U_res, V_res, W_res = np.split(res.x, [n*m*R, n*m*R + m*p*R])
            # Snap to 0.5
            U_snap = np.round(U_res.flatten().reshape(n*m, R) * 2) / 2
            V_snap = np.round(V_res.flatten().reshape(m*p, R) * 2) / 2
            W_snap = np.round(W_res.flatten().reshape(p*n, R) * 2) / 2
            
            if verify_decomposition(U_snap, V_snap, W_snap, n, m, p):
                return {
                    "U": U_snap.tolist(),
                    "V": V_snap.tolist(),
                    "W": W_snap.tolist()
                }

    # Fallback
    return get_schoolbook()
