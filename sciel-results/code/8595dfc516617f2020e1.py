import numpy as np
import time
from scipy.optimize import minimize

def construct():
    n, m, p = 3, 3, 3
    rank_target = 19
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 140  # 10s margin

    # Function to verify a decomposition
    def verify(U, V, W):
        R = U.shape[1]
        for i in range(n):
            for k in range(p):
                for j1 in range(m):
                    for j2 in range(m):
                        s = 0
                        for r in range(R):
                            s += W[k*n + i, r] * U[i*m + j1, r] * V[j2*p + k, r]
                        target = 1 if j1 == j2 else 0
                        if abs(s - target) > 1e-9:
                            return False
        return True

    # Schoolbook decomposition (Rank 27)
    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
        return U, V, W

    def error_function(params):
        R = 19
        U = params[0:R*n*m].reshape(n*m, R)
        V = params[R*n*m:2*R*n*m].reshape(m*p, R)
        W = params[2*R*n*m:3*R*n*m].reshape(p*n, R)
        
        err = 0.0
        for i in range(n):
            for k in range(p):
                for j1 in range(m):
                    for j2 in range(m):
                        s = np.dot(W[k*n+i], U[i*m+j1] * V[j2*p+k])
                        target = 1 if j1 == j2 else 0
                        err += (s - target)**2
        return err

    # Try to find Rank 19 using optimization
    # We will try a few random initializations
    best_solution = None
    best_error = float('inf')
    
    # To speed up, we can fix some symmetries or use a structured initialization
    # But random search with polish is a good bet.
    
    # We use a smaller rank if 19 fails? No, target is 19.
    # If 19 fails, we might try 20, 21... up to 27.
    
    for rank in [19, 20, 21, 22, 23]:
        if time.time() > deadline - 20:
            break
            
        R = rank
        # Define error function for this rank
        def err_func(params, R=R):
            U = params[0:R*n*m].reshape(n*m, R)
            V = params[R*n*m:2*R*n*m].reshape(m*p, R)
            W = params[2*R*n*m:3*R*n*m].reshape(p*n, R)
            
            err = 0.0
            for i in range(n):
                for k in range(p):
                    for j1 in range(m):
                        for j2 in range(m):
                            s = np.dot(W[k*n+i], U[i*m+j1] * V[j2*p+k])
                            target = 1 if j1 == j2 else 0
                            err += (s - target)**2
            return err

        # Try a few random restarts
        for restart in range(5):
            if time.time() > deadline - 10:
                break
                
            # Random initialization with small integers
            np.random.seed(42 + rank + restart)
            # Initialize with -1, 0, 1
            init_U = np.random.choice([-1, 0, 1], size=(n*m, R))
            init_V = np.random.choice([-1, 0, 1], size=(m*p, R))
            init_W = np.random.choice([-1, 0, 1], size=(p*n, R))
            
            # Flatten
            x0 = np.concatenate([init_U.flatten(), init_V.flatten(), init_W.flatten()])
            
            # Optimize
            # Use L-BFGS-B for speed
            result = minimize(err_func, x0, method='L-BFGS-B', options={'maxiter': 100, 'ftol': 1e-9})
            
            if result.fun < best_error:
                best_error = result.fun
                # Extract U, V, W
                U_opt = result.x[0:R*n*m].reshape(n*m, R)
                V_opt = result.x[R*n*m:2*R*n*m].reshape(m*p, R)
                W_opt = result.x[2*R*n*m:3*R*n*m].reshape(p*n, R)
                
                # Round to nearest integer or half-integer
                # The Laser method often yields integers.
                # Let's try rounding to nearest 0.5
                U_round = np.round(U_opt * 2) / 2
                V_round = np.round(V_opt * 2) / 2
                W_round = np.round(W_opt * 2) / 2
                
                if verify(U_round, V_round, W_round):
                    best_solution = (U_round, V_round, W_round)
                    return {"U": U_round.tolist(), "V": V_round.tolist(), "W": W_round.tolist()}
                
                # Also check pure integer rounding
                U_int = np.round(U_opt)
                V_int = np.round(V_opt)
                W_int = np.round(W_opt)
                if verify(U_int, V_int, W_int):
                    best_solution = (U_int, V_int, W_int)
                    return {"U": U_int.tolist(), "V": V_int.tolist(), "W": W_int.tolist()}

    # Fallback to schoolbook
    U, V, W = get_schoolbook()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
