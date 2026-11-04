import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 6]
DEADLINE = 140  # seconds, safety margin before 150s kill

def _verify(U, V, W):
    """
    Verify that the decomposition is correct.
    U: (n*m, rank)
    V: (m*p, rank)  
    W: (p*n, rank)
    """
    n, m, p = DIMENSIONS
    rank = len(U[0]) if U else 0
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U_arr[i * m + j, :]
                v_val = V_arr[j * p + k, :]
                w_val = W_arr[k * n + i, :]
                tensor_val = np.sum(u_val * v_val * w_val)
                if abs(tensor_val - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook():
    """Return the schoolbook decomposition with rank n*m*p."""
    n, m, p = DIMENSIONS
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

def _try_rank(r, max_iters=200, tol=1e-8, time_budget=10.0):
    """
    Try to find a rank-r decomposition using iterative optimization.
    Returns (U, V, W) if found, else None.
    """
    n, m, p = DIMENSIONS
    start = time.time()
    
    for attempt in range(5):
        if time.time() - start > time_budget:
            break
        
        np.random.seed(attempt * 12345 + r * 678)
        
        # Initialize
        U = np.random.randn(n * m, r) * 0.3
        V = np.random.randn(m * p, r) * 0.3
        W = np.random.randn(p * n, r) * 0.3
        
        # Iterative refinement using alternating least squares
        for it in range(max_iters):
            if time.time() - start > time_budget:
                break
            
            # Update U
            for i in range(n):
                for j in range(m):
                    for s in range(r):
                        # Compute gradient-like update
                        num = 0
                        den = 0
                        for k in range(p):
                            v_val = V[j * p + k, s]
                            w_val = W[k * n + i, s]
                            if abs(v_val * w_val) > 1e-12:
                                # Target contribution from this term should help reach 1
                                # Current contribution of all terms
                                current = 0
                                for s2 in range(r):
                                    current += U[i*m+j, s2] * V[j*p+k, s2] * W[k*n+i, s2]
                                # Residual
                                residual = 1.0 - current
                                # Simple update
                                num += residual * V[j*p+k, s] * W[k*n+i, s]
                                den += (V[j*p+k, s] * W[k*n+i, s])**2
                        if den > 1e-12:
                            U[i*m+j, s] = num / den
            
            # Update V
            for j in range(m):
                for k in range(p):
                    for s in range(r):
                        num = 0
                        den = 0
                        for i in range(n):
                            u_val = U[i * m + j, s]
                            w_val = W[k * n + i, s]
                            if abs(u_val * w_val) > 1e-12:
                                current = 0
                                for s2 in range(r):
                                    current += U[i*m+j, s2] * V[j*p+k, s2] * W[k*n+i, s2]
                                residual = 1.0 - current
                                num += residual * U[i*m+j, s] * W[k*n+i, s]
                                den += (U[i*m+j, s] * W[k*n+i, s])**2
                        if den > 1e-12:
                            V[j*p+k, s] = num / den
            
            # Update W
            for k in range(p):
                for i in range(n):
                    for s in range(r):
                        num = 0
                        den = 0
                        for j in range(m):
                            u_val = U[i * m + j, s]
                            v_val = V[j * p + k, s]
                            if abs(u_val * v_val) > 1e-12:
                                current = 0
                                for s2 in range(r):
                                    current += U[i*m+j, s2] * V[j*p+k, s2] * W[k*n+i, s2]
                                residual = 1.0 - current
                                num += residual * U[i*m+j, s] * V[j*p+k, s]
                                den += (U[i*m+j, s] * V[j*p+k, s])**2
                        if den > 1e-12:
                            W[k*n+i, s] = num / den
            
            # Check convergence
            max_err = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        tensor_val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                        err = abs(tensor_val - 1.0)
                        max_err = max(max_err, err)
            
            if max_err < tol:
                return U, V, W
    
    return None

def construct():
    start_time = time.time()
    
    # Start with schoolbook as fallback
    best = _schoolbook()
    best_rank = 54
    
    n, m, p = DIMENSIONS
    
    # Try to find lower rank decompositions
    for target_rank in range(53, 30, -1):
        if time.time() - start_time > DEADLINE - 20:
            break
        
        result = _try_rank(target_rank, max_iters=100, tol=1e-8, time_budget=8.0)
        
        if result is not None:
            U, V, W = result
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            
            if _verify(U_list, V_list, W_list):
                best = {"U": U_list, "V": V_list, "W": W_list}
                best_rank = target_rank
                break  # Found a valid lower rank, stop
    
    return best
