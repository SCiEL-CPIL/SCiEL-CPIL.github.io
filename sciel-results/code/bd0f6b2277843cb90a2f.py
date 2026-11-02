"""Find rank-minimal exact tensor decomposition for 2x3 * 3x9 matrix multiplication."""
import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [2, 3, 9]

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    def verify(U, V, W, rank):
        """Check if the decomposition is exact."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(rank))
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    best_result = schoolbook()
    best_rank = schoolbook_rank
    
    def try_optimization(rank, num_restarts=50, max_nfev=5000):
        """Try to find a decomposition of given rank using optimization."""
        if time.time() > deadline - 20:
            return None
        
        U_size = n * m
        V_size = m * p
        W_size = p * n
        total_size = (U_size + V_size + W_size) * rank
        
        def residual(params):
            U = params[:U_size*rank].reshape(U_size, rank)
            V = params[U_size*rank:(U_size+V_size)*rank].reshape(V_size, rank)
            W = params[(U_size+V_size)*rank:].reshape(W_size, rank)
            
            res = np.zeros(n*m*p)
            idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(
                            U[i*m+j], 
                            V[j*p+k] * W[k*n+i]
                        )
                        res[idx] = val - 1.0
                        idx += 1
            return res
        
        best_solution = None
        best_cost = float('inf')
        
        for restart in range(num_restarts):
            if time.time() > deadline - 20:
                break
            
            # Random initialization
            x0 = np.random.randn(total_size) * 0.5
            
            try:
                sol = least_squares(residual, x0, method='lm', max_nfev=max_nfev)
                if sol.cost < best_cost:
                    best_cost = sol.cost
                    if sol.cost < 1e-10:
                        best_solution = sol.x
                        break
            except Exception:
                continue
        
        if best_solution is not None:
            U = best_solution[:U_size*rank].reshape(U_size, rank)
            V = best_solution[U_size*rank:(U_size+V_size)*rank].reshape(V_size, rank)
            W = best_solution[(U_size+V_size)*rank:].reshape(W_size, rank)
            
            # Try to round to half-integers
            U2 = U * 2
            V2 = V * 2
            W2 = W * 2
            
            if (np.allclose(U2, np.round(U2), atol=1e-4) and
                np.allclose(V2, np.round(V2), atol=1e-4) and
                np.allclose(W2, np.round(W2), atol=1e-4)):
                
                U_rounded = np.round(U2) / 2.0
                V_rounded = np.round(V2) / 2.0
                W_rounded = np.round(W2) / 2.0
                
                if verify(U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist(), rank):
                    return {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}
        
        return None
    
    # Try ranks from 18 down to 10
    for rank in range(18, 9, -1):
        if time.time() > deadline - 30:
            break
        
        result = try_optimization(rank, num_restarts=30, max_nfev=3000)
        if result is not None:
            best_result = result
            best_rank = rank
            break
    
    return best_result
