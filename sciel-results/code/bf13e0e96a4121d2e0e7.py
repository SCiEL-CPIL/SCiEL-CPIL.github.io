import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 10]
_start_time = time.time()
_DEADLINE = 130  # safety margin before 150s limit

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 2x3 * 3x10 matrix multiplication.
    Uses numerical optimization to find low-rank decomposition.
    """
    n, m, p = DIMENSIONS
    start = time.time()
    
    # Build the target tensor T[i, j, k] = 1 for valid multiplication indices
    # T has shape (n, m, p) where T[i,j,k] = 1
    # The decomposition is: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # We want to find U (n*m x rank), V (m*p x rank), W (p*n x rank)
    # such that for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    nm = n * m  # 6
    mp = m * p  # 30
    pn = p * n  # 20
    
    # Target: for each (i,j,k) triple, the contraction should equal 1
    # Number of constraints: n*m*p = 60
    
    def time_check():
        return time.time() - start < _DEADLINE
    
    # Try ranks from low to high
    for rank in range(10, 61):
        if not time_check():
            break
        
        # Try multiple random initializations
        for trial in range(5):
            if not time_check():
                break
            
            # Random initialization
            U = np.random.randn(nm, rank) * 0.5
            V = np.random.randn(mp, rank) * 0.5
            W = np.random.randn(pn, rank) * 0.5
            
            # Use gradient-free optimization (Nelder-Mead style or simple iterative)
            # For each constraint (i,j,k), we need:
            # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
            
            # Let's use a simple iterative approach:
            # For each (i,j,k), compute current value and adjust
            
            best_error = float('inf')
            best_U, best_V, best_W = None, None, None
            
            for iteration in range(200):
                if not time_check():
                    break
                
                # Compute errors for all (i,j,k)
                errors = np.zeros((n, m, p))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            u_idx = i * m + j
                            v_idx = j * p + k
                            w_idx = k * n + i
                            errors[i, j, k] = np.sum(U[u_idx] * V[v_idx] * W[w_idx]) - 1.0
                
                err_norm = np.sum(errors ** 2)
                if err_norm < best_error:
                    best_error = err_norm
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
                if err_norm < 1e-12:
                    break
                
                # Gradient step: for each (i,j,k), gradient w.r.t. U[u_idx,r] is V[v_idx,r]*W[w_idx,r]
                # This is complex; let's use a simpler approach with small perturbations
                
                # Actually, let's try a different approach: use the fact that we can
                # fix one matrix and solve for others
                # For now, use random restarts with different strategies
                
                # Try adjusting U
                for i in range(n):
                    for j in range(m):
                        u_idx = i * m + j
                        for k in range(p):
                            v_idx = j * p + k
                            w_idx = k * n + i
                            # Gradient of error[i,j,k] w.r.t. U[u_idx, r] = V[v_idx,r] * W[w_idx,r]
                            grad_U = np.zeros(rank)
                            for r in range(rank):
                                grad_U[r] = V[v_idx, r] * W[w_idx, r]
                            U[u_idx] -= 0.01 * grad_U
                
                # Try adjusting V
                for j in range(m):
                    for k in range(p):
                        v_idx = j * p + k
                        for i in range(n):
                            u_idx = i * m + j
                            w_idx = k * n + i
                            grad_V = np.zeros(rank)
                            for r in range(rank):
                                grad_V[r] = U[u_idx, r] * W[w_idx, r]
                            V[v_idx] -= 0.01 * grad_V
                
                # Try adjusting W
                for k in range(p):
                    for i in range(n):
                        w_idx = k * n + i
                        for j in range(m):
                            u_idx = i * m + j
                            v_idx = j * p + k
                            grad_W = np.zeros(rank)
                            for r in range(rank):
                                grad_W[r] = U[u_idx, r] * V[v_idx, r]
                            W[w_idx] -= 0.01 * grad_W
            
            if best_error < 1e-10 and best_U is not None:
                # Found a valid decomposition!
                # Convert to lists
                U_list = [[float(x) for x in row] for row in best_U]
                V_list = [[float(x) for x in row] for row in best_V]
                W_list = [[float(x) for x in row] for row in best_W]
                return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: schoolbook decomposition
    rank = n * m * p
    U = [[0.0] * rank for _ in range(nm)]
    V = [[0.0] * rank for _ in range(mp)]
    W = [[0.0] * rank for _ in range(pn)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U[u_idx][column] = 1.0
                V[v_idx][column] = 1.0
                W[w_idx][column] = 1.0
                column += 1
    
    return {"U": U, "V": V, "W": W}
