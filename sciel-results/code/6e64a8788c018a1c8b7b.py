import numpy as np
import time
import sys

DIMENSIONS = [2, 2, 3]

def construct():
    n, m, p = DIMENSIONS
    # Target rank search
    best_rank = n * m * p
    best_solution = None
    
    # Deadline guard
    start_time = time.time()
    deadline = 140.0 # 10s margin
    
    # We will try ranks from 7 up to 12
    # Known: 2x2x2 is 7. 2x2x3 might be 8 or 7.
    # Let's try 7, 8, 9, 10.
    
    for R in range(7, 13):
        if time.time() > deadline:
            break
            
        # Try to find a solution for rank R
        solution = _find_decomposition(n, m, p, R, time_limit=10.0)
        if solution is not None:
            best_rank = R
            best_solution = solution
            break # Found minimal rank, stop
            
    if best_solution is None:
        # Fallback to schoolbook
        return _schoolbook(n, m, p)
        
    U, V, W = best_solution
    return {"U": U, "V": V, "W": W}

def _schoolbook(n, m, p):
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

def _find_decomposition(n, m, p, R, time_limit=10.0):
    start_time = time.time()
    num_equations = n * m * p
    
    # Variables: U (n*m*R), V (m*p*R), W (p*n*R)
    # We will use random restarts
    max_restarts = 50
    
    for restart in range(max_restarts):
        if time.time() - start_time > time_limit:
            break
            
        # Initialize
        U = np.random.randn(n * m, R)
        V = np.random.randn(m * p, R)
        W = np.random.randn(p * n, R)
        
        # Flatten
        x = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        # Objective function
        def objective(x):
            U_flat = x[:n*m*R]
            V_flat = x[n*m*R : n*m*R + m*p*R]
            W_flat = x[n*m*R + m*p*R :]
            
            U_mat = U_flat.reshape(n*m, R)
            V_mat = V_flat.reshape(m*p, R)
            W_mat = W_flat.reshape(p*n, R)
            
            errors = []
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        # Compute sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                        val = np.sum(U_mat[i*m+j, :] * V_mat[j*p+k, :] * W_mat[k*n+i, :])
                        errors.append(val - 1.0)
            return np.array(errors)
            
        # Use least squares
        try:
            from scipy.optimize import least_squares
            res = least_squares(objective, x, method='lm', max_nfev=1000)
        except Exception:
            continue
            
        if res.cost < 1e-6:
            # Extract solution
            U_flat = res.x[:n*m*R]
            V_flat = res.x[n*m*R : n*m*R + m*p*R]
            W_flat = res.x[n*m*R + m*p*R :]
            
            U_sol = U_flat.reshape(n*m, R)
            V_sol = V_flat.reshape(m*p, R)
            W_sol = W_flat.reshape(p*n, R)
            
            # Try to snap to half-integers
            U_half = np.round(U_sol * 2) / 2.0
            V_half = np.round(V_sol * 2) / 2.0
            W_half = np.round(W_sol * 2) / 2.0
            
            # Verify exactness
            if _verify(n, m, p, U_half, V_half, W_half):
                # Convert to lists of floats (integers or half-integers)
                return U_half.tolist(), V_half.tolist(), W_half.tolist()
            else:
                # If snapping fails, maybe try integers?
                U_int = np.round(U_sol)
                V_int = np.round(V_sol)
                W_int = np.round(W_sol)
                if _verify(n, m, p, U_int, V_int, W_int):
                    return U_int.tolist(), V_int.tolist(), W_int.tolist()
                
                # If no exact snap, we might accept a very small error if we can't do better?
                # But task requires "exactly decompose". So we must find exact.
                # If numerical solution is close, we can try local search around half-integers.
                # But for now, if snap fails, we continue searching.
                pass

    return None

def _verify(n, m, p, U, V, W):
    # U, V, W are numpy arrays
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if not np.isclose(val, 1.0, atol=1e-9):
                    return False
    return True
