import numpy as np
from scipy.optimize import least_squares
import time

def construct():
    n, m, p = 2, 3, 3
    deadline = time.time() + 140  # safety margin
    
    # Precompute the target tensor T
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T[a,b,c] = 1

    T_flat = T.flatten()
    
    def residual(x, rank):
        U = x[:n*m*rank].reshape(n*m, rank)
        V = x[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
        W = x[n*m*rank + m*p*rank:].reshape(p*n, rank)
        
        R = np.einsum('as,bs,cs->abc', U, V, W)
        return (R - T).flatten()
    
    def polish(x, rank, max_nfev=500):
        """High-precision polish pass"""
        try:
            sol = least_squares(lambda v: residual(v, rank), x, max_nfev=max_nfev, 
                               ftol=1e-12, xtol=1e-12, gtol=1e-12)
            return sol.x, np.sum(sol.fun**2)
        except Exception:
            return x, np.sum(residual(x, rank)**2)
    
    # Store best solution found so far
    best_overall = None
    best_overall_rank = 999
    best_overall_cost = np.inf
    
    # Try ranks from theoretical lower bound upwards
    for rank in range(10, 19):
        if time.time() > deadline - 30:
            break
        
        best_sol = None
        best_cost = np.inf
        
        # More trials for smaller ranks, fewer for larger
        num_trials = 5 if rank <= 11 else 3 if rank <= 13 else 2
        
        for trial in range(num_trials):
            if time.time() > deadline - 15:
                break
            
            np.random.seed(42 + rank * 100 + trial)
            
            U0 = np.random.randn(n*m, rank)
            V0 = np.random.randn(m*p, rank)
            W0 = np.random.randn(p*n, rank)
            
            x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
            
            try:
                sol = least_squares(lambda v: residual(v, rank), x0, max_nfev=300, 
                                   ftol=1e-10, xtol=1e-10, gtol=1e-10)
                cost = np.sum(sol.fun**2)
                if cost < best_cost:
                    best_cost = cost
                    best_sol = sol.x
                if cost < 1e-10:
                    break
            except Exception:
                pass
        
        # If we found a good solution, try to polish it
        if best_sol is not None and best_cost < 1e-6:
            polished_x, polished_cost = polish(best_sol, rank)
            if polished_cost < best_cost:
                best_cost = polished_cost
                best_sol = polished_x
        
        # Track overall best
        if best_sol is not None and best_cost < best_overall_cost:
            best_overall_cost = best_cost
            best_overall = best_sol
            best_overall_rank = rank
        
        # Check if we found an exact solution
        if best_sol is not None and best_cost < 1e-8:
            U = best_sol[:n*m*rank].reshape(n*m, rank)
            V = best_sol[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
            W = best_sol[n*m*rank + m*p*rank:].reshape(p*n, rank)
            
            # Try rounding to half-integers
            U_out = [[round(x*2)/2 for x in row] for row in U]
            V_out = [[round(x*2)/2 for x in row] for row in V]
            W_out = [[round(x*2)/2 for x in row] for row in W]
            
            U_np = np.array(U_out)
            V_np = np.array(V_out)
            W_np = np.array(W_out)
            R = np.einsum('as,bs,cs->abc', U_np, V_np, W_np)
            if np.allclose(R, T, atol=1e-9):
                return {"U": U_out, "V": V_out, "W": W_out}
            
            # If rounding didn't work, try integer rounding
            U_out_int = [[round(x) for x in row] for row in U]
            V_out_int = [[round(x) for x in row] for row in V]
            W_out_int = [[round(x) for x in row] for row in W]
            
            U_np_int = np.array(U_out_int)
            V_np_int = np.array(V_out_int)
            W_np_int = np.array(W_out_int)
            R_int = np.einsum('as,bs,cs->abc', U_np_int, V_np_int, W_np_int)
            if np.allclose(R_int, T, atol=1e-9):
                return {"U": U_out_int, "V": V_out_int, "W": W_out_int}
    
    # Fallback: schoolbook
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
