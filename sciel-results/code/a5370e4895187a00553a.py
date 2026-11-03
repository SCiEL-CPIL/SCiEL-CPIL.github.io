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
    
    # Strategy: Find rank 15 decomposition for 2x3x3, then extend to rank 30 for 2x3x6.
    # If that fails, return trivial rank 36.
    
    best_U, best_V, best_W = U, V, W
    best_rank = rank_trivial
    
    # Target rank for 2x3x3 is 15.
    # Target rank for 2x3x6 is 30.
    r_233 = 15
    r_236_target = 30
    
    # Define 2x3x3 tensor
    n2, m2, p2 = 2, 3, 3
    nm2 = n2 * m2 # 6
    mp2 = m2 * p2 # 9
    pn2 = p2 * n2 # 6
    
    T_233 = np.zeros((nm2, mp2, pn2), dtype=float)
    for i in range(n2):
        for j in range(m2):
            for k in range(p2):
                a = i * m2 + j
                b = j * p2 + k
                c = k * n2 + i
                T_233[a, b, c] = 1.0
                
    # Try to find rank 15 decomposition for 2x3x3
    found_233 = False
    U_233_sol = None
    V_233_sol = None
    W_233_sol = None
    
    if time.time() < deadline - 30:
        # Try multiple random restarts
        for seed in range(5):
            if time.time() > deadline - 10:
                break
            np.random.seed(seed)
            # Initialize with random vectors
            U_init = np.random.randn(nm2, r_233)
            V_init = np.random.randn(mp2, r_233)
            W_init = np.random.randn(pn2, r_233)
            
            # Scale to avoid large gradients initially?
            U_init /= np.linalg.norm(U_init)
            V_init /= np.linalg.norm(V_init)
            W_init /= np.linalg.norm(W_init)
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            def objective_233(x):
                U_f = x[:nm2*r_233].reshape(nm2, r_233)
                V_f = x[nm2*r_233:nm2*r_233+mp2*r_233].reshape(mp2, r_233)
                W_f = x[nm2*r_233+mp2*r_233:].reshape(pn2, r_233)
                T_recon = np.einsum('al,bl,cl->abc', U_f, V_f, W_f)
                res = (T_recon - T_233).flatten()
                return np.dot(res, res)
            
            def gradient_233(x):
                U_f = x[:nm2*r_233].reshape(nm2, r_233)
                V_f = x[nm2*r_233:nm2*r_233+mp2*r_233].reshape(mp2, r_233)
                W_f = x[nm2*r_233+mp2*r_233:].reshape(pn2, r_233)
                T_recon = np.einsum('al,bl,cl->abc', U_f, V_f, W_f)
                diff = T_recon - T_233
                grad_U = np.einsum('abc,bl,cl->al', diff, V_f, W_f)
                grad_V = np.einsum('abc,al,cl->bl', diff, U_f, W_f)
                grad_W = np.einsum('abc,al,bl->cl', diff, U_f, V_f)
                return np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
            
            try:
                result = minimize(objective_233, x0, jac=gradient_233, method='L-BFGS-B',
                                options={'maxiter': 2000, 'gtol': 1e-8, 'maxfun': 5000})
                
                if result.fun < 1e-6:
                    U_f = result.x[:nm2*r_233].reshape(nm2, r_233)
                    V_f = result.x[nm2*r_233:nm2*r_233+mp2*r_233].reshape(mp2, r_233)
                    W_f = result.x[nm2*r_233+mp2*r_233:].reshape(pn2, r_233)
                    
                    # Verify
                    T_recon = np.einsum('al,bl,cl->abc', U_f, V_f, W_f)
                    if np.max(np.abs(T_recon - T_233)) < 1e-6:
                        U_233_sol = U_f
                        V_233_sol = V_f
                        W_233_sol = W_f
                        found_233 = True
                        break
            except Exception as e:
                pass
    
    if found_233:
        # Construct 2x3x6 decomposition from 2x3x3
        # Rank 30
        U_236 = np.zeros((nm2, 2 * r_233)) # 6 x 30
        V_236 = np.zeros((m2 * p, 2 * r_233)) # 18 x 30
        W_236 = np.zeros((p * n2, 2 * r_233)) # 12 x 30
        
        # First block (k=0,1,2)
        U_236[:, :r_233] = U_233_sol
        # V indices for k=0,1,2 are j*3+k. In 2x3x6, these are j*6+k.
        for j in range(m2):
            for k in range(p2):
                v_idx_233 = j * p2 + k
                v_idx_236 = j * p + k
                V_236[v_idx_236, :r_233] = V_233_sol[v_idx_233, :]
        
        # W indices for k=0,1,2 are k*2+i. In 2x3x6, these are k*2+i.
        for k in range(p2):
            for i in range(n2):
                w_idx_233 = k * n2 + i
                w_idx_236 = k * n2 + i
                W_236[w_idx_236, :r_233] = W_233_sol[w_idx_233, :]
        
        # Second block (k=3,4,5)
        U_236[:, r_233:] = U_233_sol
        # V indices for k=3,4,5 correspond to k'=0,1,2 in 233, but shifted in 236.
        # In 233, v_idx = j*3 + k'.
        # In 236, v_idx = j*6 + (k'+3).
        for j in range(m2):
            for k_prime in range(p2):
                v_idx_233 = j * p2 + k_prime
                v_idx_236 = j * p + (k_prime + 3)
                V_236[v_idx_236, r_233:] = V_233_sol[v_idx_233, :]
        
        # W indices for k=3,4,5 correspond to k'=0,1,2 in 233, but shifted in 236.
        # In 233, w_idx = k'*2 + i.
        # In 236, w_idx = (k'+3)*2 + i.
        for k_prime in range(p2):
            for i in range(n2):
                w_idx_233 = k_prime * n2 + i
                w_idx_236 = (k_prime + 3) * n2 + i
                W_236[w_idx_236, r_233:] = W_233_sol[w_idx_233, :]
        
        # Verify the full 2x3x6 decomposition
        T_236 = np.zeros((n*m, m*p, p*n), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    T_236[a, b, c] = 1.0
        
        T_recon_236 = np.einsum('al,bl,cl->abc', U_236, V_236, W_236)
        if np.max(np.abs(T_recon_236 - T_236)) < 1e-6:
            best_U = U_236.tolist()
            best_V = V_236.tolist()
            best_W = W_236.tolist()
            best_rank = 30
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
