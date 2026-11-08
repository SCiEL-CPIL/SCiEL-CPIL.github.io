import numpy as np
from scipy.optimize import minimize
import time

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    # Tensor entries: T[i,j,k] for i in range(n), j in range(m), k in range(p)
    # T[i,j,k] = 1 for all valid (i,j,k) - actually no.
    
    # The multiplication tensor: C[i,k] = sum_j A[i,j] B[j,k]
    # In the decomposition: C[i,k] = sum_c W[k*n+i,c] * (sum_j A[i,j]*U[i*m+j,c]) * (sum_j B[j,k]*V[j*p+k,c])
    
    # So the tensor T is defined by: T[i,j,k] should equal the coefficient such that
    # sum_c W[k*n+i,c] * U[i*m+j,c] * V[j*p+k,c] = delta... 
    
    # Actually, the condition is that for ALL A and B:
    # C[i,k] = sum_j A[i,j] B[j,k] = sum_c W[k*n+i,c] * (sum_j A[i,j]*U[i*m+j,c]) * (sum_j B[j,k]*V[j*p+k,c])
    
    # This must hold for all A, B. So we need:
    # sum_c W[k*n+i,c] * U[i*m+j1,c] * V[j1*p+k1,c] ... 
    
    # Let me expand: RHS = sum_c W[k*n+i,c] * sum_{j1} A[i,j1]*U[i*m+j1,c] * sum_{j2} B[j2,k]*V[j2*p+k,c]
    # = sum_{j1,j2,c} A[i,j1]*B[j2,k]*W[k*n+i,c]*U[i*m+j1,c]*V[j2*p+k,c]
    
    # LHS = sum_j A[i,j]*B[j,k]
    
    # So we need: sum_c W[k*n+i,c]*U[i*m+j1,c]*V[j2*p+k,c] = delta_{j1,j2} for all i,j1,j2,k
    
    # This is the key constraint!
    
    def residual(params, r):
        """Compute residual given parameters and rank r."""
        # params: flat array of U (6*r), V (21*r), W (14*r)
        U = params[:6*r].reshape(n*m, r)
        V = params[6*r:6*r+21*r].reshape(m*p, r)
        W = params[6*r+21*r:].reshape(p*n, r)
        
        res = 0.0
        for i in range(n):
            for j1 in range(m):
                for j2 in range(m):
                    for k in range(p):
                        val = sum(W[k*n+i, c] * U[i*m+j1, c] * V[j2*p+k, c] for c in range(r))
                        target = 1.0 if j1 == j2 else 0.0
                        res += (val - target) ** 2
        return res
    
    best_rank = n*m*p  # schoolbook
    best_solution = None
    
    deadline = time.time() + 120
    
    for r in range(1, best_rank + 1):
        if time.time() > deadline - 30:
            break
        
        found = False
        max_restarts = 50
        for restart in range(max_restarts):
            if time.time() > deadline - 30:
                break
            
            # Random initialization
            np.random.seed(42 + restart * 1000 + r)
            params = np.random.randn(41 * r) * 0.1
            
            # Optimize
            result = minimize(residual, params, args=(r,), method='BFGS', 
                           options={'maxiter': 1000, 'gtol': 1e-10})
            
            if result.fun < 1e-10:
                U = result.x[:6*r].reshape(n*m, r)
                V = result.x[6*r:6*r+21*r].reshape(m*p, r)
                W = result.x[6*r+21*r:].reshape(p*n, r)
                
                # Try to snap to half-integers
                U_half = np.round(U * 2) / 2
                V_half = np.round(V * 2) / 2
                W_half = np.round(W * 2) / 2
                
                # Check if half-integer version also works
                def check_solution(U_c, V_c, W_c, r_c):
                    for i in range(n):
                        for j1 in range(m):
                            for j2 in range(m):
                                for k in range(p):
                                    val = sum(W_c[k*n+i, c] * U_c[i*m+j1, c] * V_c[j2*p+k, c] for c in range(r_c))
                                    target = 1.0 if j1 == j2 else 0.0
                                    if abs(val - target) > 1e-9:
                                        return False
                    return True
                
                if check_solution(U_half, V_half, W_half, r):
                    U_f = [[U_half[i, j] for j in range(r)] for i in range(n*m)]
                    V_f = [[V_half[i, j] for j in range(r)] for i in range(m*p)]
                    W_f = [[W_half[i, j] for j in range(r)] for i in range(p*n)]
                    return {"U": U_f, "V": V_f, "W": W_f}
                elif check_solution(U, V, W, r):
                    # Real solution works, try to find half-integer nearby
                    # For now, accept real if we can't snap
                    U_f = [[U[i, j] for j in range(r)] for i in range(n*m)]
                    V_f = [[V[i, j] for j in range(r)] for i in range(m*p)]
                    W_f = [[W[i, j] for j in range(r)] for i in range(p*n)]
                    # But problem requires integer/half-integer... let's keep searching
                    pass
            
            if time.time() > deadline - 30:
                break
        
        if found:
            break
    
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
