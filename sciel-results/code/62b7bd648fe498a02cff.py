"""Matrix multiplication 2x3x6 tensor decomposition; structural construction with rank reduction search."""
DIMENSIONS = [2, 3, 6]

import time
import numpy as np
from fractions import Fraction
from scipy.optimize import minimize
import itertools
import random

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 110.0
    n, m, p = DIMENSIONS
    
    # Trivial rank: n*m*p = 36
    rank_trivial = n * m * p
    
    # Build trivial decomposition
    U = [[0] * rank_trivial for _ in range(n * m)]
    V = [[0] * rank_trivial for _ in range(m * p)]
    W = [[0] * rank_trivial for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    # Define the target tensor
    # T[a,b,c] = 1 if there exist i,j,k such that a=i*m+j, b=j*p+k, c=k*n+i
    # and (i,j,k) is the unique triple, else 0
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Build target tensor as a dense array
    T = np.zeros((nm, mp, pn), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Try to find lower rank decomposition
    best_U, best_V, best_W = U, V, W
    best_rank = rank_trivial
    
    # Try rank 35
    r = rank_trivial - 1
    if time.time() < deadline - 60:
        try:
            # Parameterize U, V, W as flat vectors
            # U: nm x r, V: mp x r, W: pn x r
            # Total params: r*(nm + mp + pn)
            
            # Start from a perturbed version of the trivial solution
            # Take the first r columns of the trivial solution and add small perturbations
            np.random.seed(42)
            U_init = np.array(U[:, :r], dtype=float)
            V_init = np.array(V[:, :r], dtype=float)
            W_init = np.array(W[:, :r], dtype=float)
            
            # Add small random perturbations
            U_init += 0.01 * np.random.randn(nm, r)
            V_init += 0.01 * np.random.randn(mp, r)
            W_init += 0.01 * np.random.randn(pn, r)
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            def residual(x):
                U_f = x[:nm*r].reshape(nm, r)
                V_f = x[nm*r:nm*r+mp*r].reshape(mp, r)
                W_f = x[nm*r+mp*r:].reshape(pn, r)
                
                # Compute tensor: T_recon[a,b,c] = sum_l U[a,l]*V[b,l]*W[c,l]
                # Use einsum for efficiency
                T_recon = np.einsum('al,bl,cl->abc', U_f, V_f, W_f)
                return (T_recon - T).flatten()
            
            def objective(x):
                res = residual(x)
                return np.dot(res, res)
            
            def gradient(x):
                U_f = x[:nm*r].reshape(nm, r)
                V_f = x[nm*r:nm*r+mp*r].reshape(mp, r)
                W_f = x[nm*r+mp*r:].reshape(pn, r)
                
                T_recon = np.einsum('al,bl,cl->abc', U_f, V_f, W_f)
                diff = T_recon - T  # shape (nm, mp, pn)
                
                # Gradient w.r.t. U[a,l]: sum_{b,c} diff[a,b,c] * V[b,l] * W[c,l]
                grad_U = np.einsum('abc,bl,cl->al', diff, V_f, W_f)
                grad_V = np.einsum('abc,al,cl->bl', diff, U_f, W_f)
                grad_W = np.einsum('abc,al,bl->cl', diff, U_f, V_f)
                
                return np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
            
            result = minimize(objective, x0, jac=gradient, method='L-BFGS-B',
                            options={'maxiter': 5000, 'gtol': 1e-8, 'maxfun': 10000})
            
            if result.success or result.fun < 1e-6:
                U_f = result.x[:nm*r].reshape(nm, r)
                V_f = result.x[nm*r:nm*r+mp*r].reshape(mp, r)
                W_f = result.x[nm*r+mp*r:].reshape(pn, r)
                
                # Verify
                T_recon = np.einsum('al,bl,cl->abc', U_f, V_f, W_f)
                if np.max(np.abs(T_recon - T)) < 1e-6:
                    best_U = U_f.tolist()
                    best_V = V_f.tolist()
                    best_W = W_f.tolist()
                    best_rank = r
        
        except Exception as e:
            pass
    
    # Try rank 34 if we have time
    r = rank_trivial - 2
    if best_rank == rank_trivial - 1 and time.time() < deadline - 30:
        try:
            np.random.seed(43)
            U_init = np.array(U[:, :r], dtype=float)
            V_init = np.array(V[:, :r], dtype=float)
            W_init = np.array(W[:, :r], dtype=float)
            
            U_init += 0.01 * np.random.randn(nm, r)
            V_init += 0.01 * np.random.randn(mp, r)
            W_init += 0.01 * np.random.randn(pn, r)
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            def objective(x):
                U_f = x[:nm*r].reshape(nm, r)
                V_f = x[nm*r:nm*r+mp*r].reshape(mp, r)
                W_f = x[nm*r+mp*r:].reshape(pn, r)
                T_recon = np.einsum('al,bl,cl->abc', U_f, V_f, W_f)
                res = (T_recon - T).flatten()
                return np.dot(res, res)
            
            def gradient(x):
                U_f = x[:nm*r].reshape(nm, r)
                V_f = x[nm*r:nm*r+mp*r].reshape(mp, r)
                W_f = x[nm*r+mp*r:].reshape(pn, r)
                T_recon = np.einsum('al,bl,cl->abc', U_f, V_f, W_f)
                diff = T_recon - T
                grad_U = np.einsum('abc,bl,cl->al', diff, V_f, W_f)
                grad_V = np.einsum('abc,al,cl->bl', diff, U_f, W_f)
                grad_W = np.einsum('abc,al,bl->cl', diff, U_f, V_f)
                return np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
            
            result = minimize(objective, x0, jac=gradient, method='L-BFGS-B',
                            options={'maxiter': 5000, 'gtol': 1e-8, 'maxfun': 10000})
            
            if result.success or result.fun < 1e-6:
                U_f = result.x[:nm*r].reshape(nm, r)
                V_f = result.x[nm*r:nm*r+mp*r].reshape(mp, r)
                W_f = result.x[nm*r+mp*r:].reshape(pn, r)
                
                T_recon = np.einsum('al,bl,cl->abc', U_f, V_f, W_f)
                if np.max(np.abs(T_recon - T)) < 1e-6:
                    best_U = U_f.tolist()
                    best_V = V_f.tolist()
                    best_W = W_f.tolist()
                    best_rank = r
        
        except Exception as e:
            pass
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
