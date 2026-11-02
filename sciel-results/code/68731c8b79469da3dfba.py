import numpy as np
from scipy.optimize import differential_evolution
import time

DIMENSIONS = [3, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    total_dims = n * m * p
    
    def compute_error(params, R):
        U = params[:R*n*m].reshape(n*m, R)
        V = params[R*n*m:2*R*n*m].reshape(m*p, R)
        W = params[2*R*n*m:3*R*n*m].reshape(p*n, R)
        
        errors = np.zeros(n*m*p)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    val = np.sum(U[u_idx, :] * V[v_idx, :] * W[w_idx, :])
                    errors[idx] = val - 1.0
                    idx += 1
        return np.sum(errors**2)

    def try_find_solution(R, time_budget):
        start_time = time.time()
        bounds = [(-5, 5)] * (R * n * m * 3)
        
        # Differential evolution is slow, so we limit the population and generations
        # We use a time check inside the callback? No, DE doesn't support time-based stopping easily.
        # We will run a few iterations and check time.
        
        # Instead, we use a random restart with LM but with better initialization.
        # We initialize with the schoolbook solution and then try to reduce rank.
        
        # This is too complex.
        # I will stick to the LM search but with a larger budget.
        
        for attempt in range(50):
            if time.time() - start_time > time_budget:
                return None
            
            params = np.random.uniform(-3, 3, size=R * n * m * 3)
            
            try:
                from scipy.optimize import least_squares
                res = least_squares(lambda x: compute_error(x, R), params, method='lm', max_nfev=2000)
                
                if res.cost < 1e-8:
                    U = res.x[:R*n*m].reshape(n*m, R)
                    V = res.x[R*n*m:2*R*n*m].reshape(m*p, R)
                    W = res.x[2*R*n*m:3*R*n*m].reshape(p*n, R)
                    
                    U_snap = np.round(U)
                    V_snap = np.round(V)
                    W_snap = np.round(W)
                    
                    snap_params = np.concatenate([U_snap.flatten(), V_snap.flatten(), W_snap.flatten()])
                    err = compute_error(snap_params, R)
                    
                    if np.max(np.abs(err)) < 1e-9:
                        return {
                            "U": U_snap.tolist(),
                            "V": V_snap.tolist(),
                            "W": W_snap.tolist()
                        }
            except Exception:
                continue
        
        return None

    # Try rank 23
    sol = try_find_solution(23, 80)
    if sol:
        return sol

    # Fallback to Schoolbook (Rank 27)
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
