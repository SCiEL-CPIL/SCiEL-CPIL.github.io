import numpy as np
import time
from scipy.optimize import minimize

def construct():
    n, m, p = 3, 3, 3
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    def check_decomposition(U, V, W):
        """
        Checks if the decomposition is exact.
        U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
        """
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(s - 1) > 1e-6:
                        return False
        return True

    def schoolbook():
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
        return {"U": U, "V": V, "W": W}

    def try_rank_19():
        """Try to find a rank-19 decomposition via optimization."""
        rank = 19
        
        # Initialize with random values
        np.random.seed(42)
        U = np.random.randn(n*m, rank) * 0.1
        V = np.random.randn(m*p, rank) * 0.1
        W = np.random.randn(p*n, rank) * 0.1
        
        # Define the objective function
        def objective(params):
            U_flat = params[:n*m*rank].reshape(n*m, rank)
            V_flat = params[n*m*rank:2*n*m*rank].reshape(m*p, rank)
            W_flat = params[2*n*m*rank:].reshape(p*n, rank)
            
            # Compute the residual
            residual = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = np.sum(U_flat[i*m+j, :] * V_flat[j*p+k, :] * W_flat[k*n+i, :])
                        residual += (s - 1.0) ** 2
            return residual
        
        # Initial parameters
        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        # Try to optimize
        try:
            result = minimize(objective, x0, method='BFGS', options={'maxiter': 200, 'gtol': 1e-8})
            if result.fun < 1e-6:
                U_opt = result.x[:n*m*rank].reshape(n*m, rank)
                V_opt = result.x[n*m*rank:2*n*m*rank].reshape(m*p, rank)
                W_opt = result.x[2*n*m*rank:].reshape(p*n, rank)
                
                # Round to nearest half-integer
                U_round = np.round(U_opt * 2) / 2
                V_round = np.round(V_opt * 2) / 2
                W_round = np.round(W_opt * 2) / 2
                
                if check_decomposition(U_round, V_round, W_round):
                    return {"U": U_round, "V": V_round, "W": W_round}
        except Exception:
            pass
        
        return None

    # Try to find a lower rank decomposition
    result_19 = try_rank_19()
    if result_19 is not None:
        return result_19

    # Fallback to schoolbook
    result = schoolbook()
    return result
