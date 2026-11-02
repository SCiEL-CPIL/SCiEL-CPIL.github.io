import numpy as np
import time
import sys
from scipy.optimize import least_squares

DIMENSIONS = [3, 3, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    n, m, p = DIMENSIONS
    # Indices:
    # U: (n*m, R)
    # V: (m*p, R)
    # W: (p*n, R)
    
    # The tensor equation:
    # Sum_{r=1}^R U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # for all i in [0,n), j in [0,m), k in [0,p)
    
    def residual(x, R):
        # x is flattened [U, V, W]
        size_u = n * m * R
        size_v = m * p * R
        size_w = p * n * R
        
        U = x[:size_u].reshape(n * m, R)
        V = x[size_u:size_u+size_v].reshape(m * p, R)
        W = x[size_u+size_v:].reshape(p * n, R)
        
        res = np.zeros(n * m * p)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Term: U[i*m+j, :] dot (V[j*p+k, :] * W[k*n+i, :])
                    # Note: element-wise product of V row and W row, then dot with U row
                    val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                    res[idx] = val - 1.0
                    idx += 1
        return res

    def check_solution(U, V, W):
        # Check if entries are close to half-integers and equation holds exactly
        for mat in [U, V, W]:
            for val in mat:
                # Check if 2*val is close to an integer
                if abs(2*val - round(2*val)) > 1e-6:
                    return False
        # Verify equations
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(U.shape[1]):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(s - 1.0) > 1e-6:
                        return False
        return True

    def snap_to_half_integer(mat):
        return np.round(mat * 2) / 2.0

    # Try ranks from low to high
    # Start with 25, go up to 36
    for R in range(25, 37):
        if time.time() > deadline:
            break
            
        # Try a few random restarts for this rank
        for attempt in range(20):
            if time.time() > deadline:
                break
                
            # Initialize with small random values
            np.random.seed(attempt + R) # Reproducible-ish
            U = np.random.choice([-1, 0, 1], size=(n*m, R)) * 0.1
            V = np.random.choice([-1, 0, 1], size=(m*p, R)) * 0.1
            W = np.random.choice([-1, 0, 1], size=(p*n, R)) * 0.1
            
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            # Use least_squares to refine
            try:
                sol = least_squares(residual, x0, args=(R,), method='lm', max_nfev=1000)
            except Exception:
                continue
                
            if sol.success or sol.cost < 1e-10:
                # Extract solution
                size_u = n * m * R
                size_v = m * p * R
                U_sol = sol.x[:size_u].reshape(n * m, R)
                V_sol = sol.x[size_u:size_u+size_v].reshape(m * p, R)
                W_sol = sol.x[size_u+size_v:].reshape(p * n, R)
                
                # Snap to half-integers
                U_sol = snap_to_half_integer(U_sol)
                V_sol = snap_to_half_integer(V_sol)
                W_sol = snap_to_half_integer(W_sol)
                
                if check_solution(U_sol, V_sol, W_sol):
                    # Convert to lists of lists as per interface
                    return {
                        "U": U_sol.tolist(),
                        "V": V_sol.tolist(),
                        "W": W_sol.tolist()
                    }
        
    # Fallback: Schoolbook decomposition
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
