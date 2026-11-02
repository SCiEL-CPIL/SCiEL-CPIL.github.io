import numpy as np
import time
import random
import itertools
from scipy.optimize import minimize

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s limit to be safe
    
    n, m, p = DIMENSIONS
    U_dim = n * m  # 10
    V_dim = m * p  # 25
    W_dim = p * n  # 10
    
    # Build the target tensor T
    # T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i else 0
    T = np.zeros((U_dim, V_dim, W_dim))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0
    
    def snap_to_half_integers(arr):
        # Round to nearest 0.5
        return np.round(arr * 2) / 2
    
    def validate(U, V, W, atol=1e-4):
        # Check if entries are integer or half-integer
        U_snap = snap_to_half_integers(U)
        V_snap = snap_to_half_integers(V)
        W_snap = snap_to_half_integers(W)
        
        # Check if snapped matrices are close to original
        if not (np.allclose(U, U_snap, atol=atol) and 
                np.allclose(V, V_snap, atol=atol) and 
                np.allclose(W, W_snap, atol=atol)):
            return False, None, None, None
            
        # Compute tensor with snapped values
        prod = np.einsum('ar,br,cr->abc', U_snap, V_snap, W_snap, optimize=True)
        if np.allclose(prod, T, atol=atol):
            return True, U_snap, V_snap, W_snap
        return False, None, None
    
    def compute_loss(U, V, W):
        # Compute sum_r U[:,r] V[:,r] W[:,r]
        prod = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        diff = prod - T
        return np.sum(diff * diff)
    
    # Schoolbook decomposition as fallback
    def schoolbook():
        rank = n * m * p
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1.0
                    V[j * p + k][column] = 1.0
                    W[k * n + i][column] = 1.0
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    # Try to find rank 45 decomposition
    target_rank = 45
    best_solution = None
    
    # Strategy 1: Start from schoolbook and try to merge terms
    # The schoolbook has 50 terms. We need to reduce to 45.
    # Idea: Initialize with 45 random linear combinations of the schoolbook terms
    
    num_restarts = 3
    
    for restart in range(num_restarts):
        if time.time() > deadline - 15:
            break
        
        R = target_rank
        random.seed(42 + restart)
        
        # Initialize U, V, W by taking random linear combinations of schoolbook columns
        # Schoolbook columns: for each (i,j,k), U has 1 at (i*m+j), V has 1 at (j*p+k), W has 1 at (k*n+i)
        
        # Create random coefficients for merging
        # We have 50 schoolbook terms, we want 45 rank-1 terms
        # Initialize with random 50x45 matrix for each of U, V, W
        
        # Actually, let's initialize more carefully
        # U_init[i*m+j, r] = sum over k of alpha_{ijk,r}
        # V_init[j*p+k, r] = sum over i of alpha_{ijk,r}
        # W_init[k*n+i, r] = sum over j of alpha_{ijk,r}
        
        # For each rank position r, pick a random subset of schoolbook terms to combine
        U = np.zeros((U_dim, R))
        V = np.zeros((V_dim, R))
        W = np.zeros((W_dim, R))
        
        # For each rank position, randomly select 5-6 schoolbook terms to combine
        for r in range(R):
            # Select 5-6 random (i,j,k) triples
            num_terms = random.choice([5, 6, 7])
            selected = random.sample(range(n*m*p), num_terms)
            
            for idx in selected:
                i = idx // (m*p)
                j = (idx // p) % m
                k = idx % p
                
                coeff = random.uniform(0.5, 1.5)
                U[i*m+j, r] += coeff
                V[j*p+k, r] += coeff
                W[k*n+i, r] += coeff
        
        # Optimize using L-BFGS-B
        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        def objective(x):
            u = x[:U_dim*R].reshape(U_dim, R)
            v = x[U_dim*R:U_dim*R+V_dim*R].reshape(V_dim, R)
            w = x[U_dim*R+V_dim*R:].reshape(W_dim, R)
            return compute_loss(u, v, w)
        
        try:
            res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 200, 'disp': False})
            
            if res.fun < 1e-3:
                u_opt = res.x[:U_dim*R].reshape(U_dim, R)
                v_opt = res.x[U_dim*R:U_dim*R+V_dim*R].reshape(V_dim, R)
                w_opt = res.x[U_dim*R+V_dim*R:].reshape(W_dim, R)
                
                valid, U_snap, V_snap, W_snap = validate(u_opt, v_opt, w_opt)
                if valid:
                    best_solution = (U_snap, V_snap, W_snap)
                    break
        except Exception as e:
            pass
    
    # If rank 45 not found, try rank 46
    if not best_solution:
        target_rank = 46
        for restart in range(2):
            if time.time() > deadline - 10:
                break
            
            R = target_rank
            random.seed(100 + restart)
            
            U = np.zeros((U_dim, R))
            V = np.zeros((V_dim, R))
            W = np.zeros((W_dim, R))
            
            for r in range(R):
                num_terms = random.choice([5, 6])
                selected = random.sample(range(n*m*p), num_terms)
                
                for idx in selected:
                    i = idx // (m*p)
                    j = (idx // p) % m
                    k = idx % p
                    
                    coeff = random.uniform(0.5, 1.5)
                    U[i*m+j, r] += coeff
                    V[j*p+k, r] += coeff
                    W[k*n+i, r] += coeff
            
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            def objective(x):
                u = x[:U_dim*R].reshape(U_dim, R)
                v = x[U_dim*R:U_dim*R+V_dim*R].reshape(V_dim, R)
                w = x[U_dim*R+V_dim*R:].reshape(W_dim, R)
                return compute_loss(u, v, w)
            
            try:
                res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 200, 'disp': False})
                
                if res.fun < 1e-3:
                    u_opt = res.x[:U_dim*R].reshape(U_dim, R)
                    v_opt = res.x[U_dim*R:U_dim*R+V_dim*R].reshape(V_dim, R)
                    w_opt = res.x[U_dim*R+V_dim*R:].reshape(W_dim, R)
                    
                    valid, U_snap, V_snap, W_snap = validate(u_opt, v_opt, w_opt)
                    if valid:
                        best_solution = (U_snap, V_snap, W_snap)
                        break
            except Exception as e:
                pass
    
    # If still not found, return schoolbook
    if best_solution:
        U, V, W = best_solution
        U_list = [[float(x) for x in row] for row in U]
        V_list = [[float(x) for x in row] for row in V]
        W_list = [[float(x) for x in row] for row in W]
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
        return schoolbook()

# EVOLVE-BLOCK-END
