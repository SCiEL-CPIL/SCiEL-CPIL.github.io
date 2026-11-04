"""
Program to find a rank-minimal exact tensor decomposition for matrix multiplication 2x3 by 3x7.
"""

import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 7]

def construct():
    """
    Returns a dictionary with keys 'U', 'V', 'W' representing the factor matrices
    of the tensor decomposition.
    """
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition rank
    rank_schoolbook = n * m * p
    
    # Build the tensor: T[a, b, c] = 1 if a=(i,j), b=(j',k), c=(k',i') with j=j', i=i', k=k'
    # A index: i*m + j
    # B index: j*p + k
    # C index: k*n + i
    
    tensor = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                tensor[a, b, c] = 1.0
    
    # Try to find lower rank decompositions
    best_U = None
    best_V = None
    best_W = None
    best_rank = rank_schoolbook
    
    # Schoolbook solution as fallback
    U_school = np.zeros((n*m, rank_schoolbook))
    V_school = np.zeros((m*p, rank_schoolbook))
    W_school = np.zeros((p*n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m + j, col] = 1.0
                V_school[j*p + k, col] = 1.0
                W_school[k*n + i, col] = 1.0
                col += 1
    
    # Try ranks from lower to higher
    # Theoretical lower bound: max(n*m, m*p, p*n) = max(6, 21, 14) = 21
    # But we need to check if lower ranks are possible
    
    deadline = time.time() + 120
    
    for rank in range(21, rank_schoolbook + 1):
        if time.time() > deadline - 30:
            break
        
        # Try multiple random initializations
        found = False
        n_attempts = 5
        
        for attempt in range(n_attempts):
            if time.time() > deadline - 25:
                break
            
            # Random initialization
            np.random.seed(42 + attempt * 1000 + rank)
            U = np.random.randn(n*m, rank) * 0.5
            V = np.random.randn(m*p, rank) * 0.5
            W = np.random.randn(p*n, rank) * 0.5
            
            def residual(params):
                U_r = params[:n*m*rank].reshape(n*m, rank)
                V_r = params[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
                W_r = params[n*m*rank + m*p*rank:].reshape(p*n, rank)
                
                # Compute tensor from factors
                T = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
                return np.sum((T - tensor) ** 2)
            
            def residual_grad(params):
                U_r = params[:n*m*rank].reshape(n*m, rank)
                V_r = params[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
                W_r = params[n*m*rank + m*p*rank:].reshape(p*n, rank)
                
                T = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
                diff = T - tensor
                
                # Gradient with respect to U: d/dU[a,r] sum_{b,c} diff[a,b,c] * V[b,r] * W[c,r]
                grad_U = np.einsum('abc,br,cr->ar', diff, V_r, W_r)
                grad_V = np.einsum('abc,ar,cr->br', diff, U_r, W_r)
                grad_W = np.einsum('abc,ar,br->cr', diff, U_r, V_r)
                
                return np.concatenate([grad_U.ravel(), grad_V.ravel(), grad_W.ravel()])
            
            params0 = np.concatenate([U.ravel(), V.ravel(), W.ravel()])
            
            try:
                result = minimize(residual, params0, jac=residual_grad, method='L-BFGS-B',
                                  options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50})
                
                if result.success or result.fun < 1e-8:
                    U_r = result.x[:n*m*rank].reshape(n*m, rank)
                    V_r = result.x[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
                    W_r = result.x[n*m*rank + m*p*rank:].reshape(p*n, rank)
                    
                    # Verify
                    T = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
                    err = np.max(np.abs(T - tensor))
                    
                    if err < 1e-6:
                        # Try to round to nearest half-integers
                        U_rounded = np.round(U_r * 2) / 2
                        V_rounded = np.round(V_r * 2) / 2
                        W_rounded = np.round(W_r * 2) / 2
                        
                        T_rounded = np.einsum('ar,br,cr->abc', U_rounded, V_rounded, W_rounded)
                        err_rounded = np.max(np.abs(T_rounded - tensor))
                        
                        if err_rounded < 1e-6:
                            best_U = U_rounded
                            best_V = V_rounded
                            best_W = W_rounded
                            best_rank = rank
                            found = True
                            break
                        else:
                            # Use the unrounded version
                            best_U = U_r
                            best_V = V_r
                            best_W = W_r
                            best_rank = rank
                            found = True
                            break
                    
            except Exception as e:
                continue
        
        if found:
            break
    
    # Use best found or fallback to schoolbook
    if best_U is not None:
        U_out = best_U
        V_out = best_V
        W_out = best_W
    else:
        U_out = U_school
        V_out = V_school
        W_out = W_school
    
    return {"U": U_out.tolist(), "V": V_out.tolist(), "W": W_out.tolist()}

# EVOLVE-BLOCK-START
# The construct function is defined above.
# EVOLVE-BLOCK-END
