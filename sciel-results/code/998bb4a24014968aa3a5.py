import numpy as np
import time
import random

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    target_rank = 12  # Known minimal rank for 2x3x4 is 12
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # Function to check if U, V, W form a valid decomposition
    def is_valid(U, V, W):
        # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
        rank = U.shape[1]
        # Check all i, j, k
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    val = 0
                    for r in range(rank):
                        val += U[u_idx, r] * V[v_idx, r] * W[w_idx, r]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    # Function to round to half-integers
    def round_half(x):
        return np.round(x * 2) / 2

    # Search for rank R decomposition
    def search_rank(R, max_iters=1000):
        for _ in range(max_iters):
            if time.time() > deadline:
                return None
            
            # Random initialization with small values
            U = np.random.choice([-1, 0, 1, 0.5, -0.5], size=(n*m, R))
            V = np.random.choice([-1, 0, 1, 0.5, -0.5], size=(m*p, R))
            W = np.random.choice([-1, 0, 1, 0.5, -0.5], size=(p*n, R))
            
            # Local search / Gradient descent
            # We want to minimize sum (sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] - 1)^2
            # This is non-convex.
            # Simple approach: random restarts with checking.
            
            # Since the space is discrete (half-integers), we can just random sample and check.
            # But the probability of hitting the exact solution is low.
            # Better: Use a solver if available? No, standard lib + numpy/scipy.
            # We can use scipy.optimize.minimize to find a real-valued solution, then round.
            
            try:
                # Flatten U, V, W
                U_flat = U.flatten()
                V_flat = V.flatten()
                W_flat = W.flatten()
                
                def objective(x):
                    U_x = x[:n*m*R].reshape(n*m, R)
                    V_x = x[n*m*R:n*m*R + m*p*R].reshape(m*p, R)
                    W_x = x[n*m*R + m*p*R:].reshape(p*n, R)
                    err = 0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                u_idx = i * m + j
                                v_idx = j * p + k
                                w_idx = k * n + i
                                val = np.dot(U_x[u_idx, :], V_x[v_idx, :] * W_x[w_idx, :])
                                err += (val - 1.0)**2
                    return err
                
                # Use scipy to minimize
                from scipy.optimize import minimize
                x0 = np.concatenate([U_flat, V_flat, W_flat])
                
                # We need to be careful with time.
                # If we do this for many iterations, it might be slow.
                # Let's just do a few random restarts with a quick check.
                
                # Actually, for such a small system, a smarter search might be needed.
                # Let's try a few random integer/half-integer guesses.
                
            except Exception:
                pass
            
            # Check the random guess
            if is_valid(U, V, W):
                return U.tolist(), V.tolist(), W.tolist()
                
            # Try a slightly modified guess?
            # For now, just random restarts.
            
        return None

    # Try rank 12
    for R in range(12, 25):
        if time.time() > deadline:
            break
        result = search_rank(R, max_iters=50)
        if result:
            return {"U": result[0], "V": result[1], "W": result[2]}
            
    # Fallback to schoolbook
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
