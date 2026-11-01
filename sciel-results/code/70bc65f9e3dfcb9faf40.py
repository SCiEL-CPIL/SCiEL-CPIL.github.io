"""Matrix multiplication 2,2,15 - rank-minimal tensor decomposition"""
DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0
    
    # The tensor T[i,j,k] = 1 for all i,j,k (since (A*B)[i,k] = sum_j A[i,j]*B[j,k])
    # We need U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    # such that for all i,j,k: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # Start with the known rank-53 solution and try to reduce rank
    # by finding a rank-r solution for r < 53
    
    # Build the tensor
    T = np.ones((n, m, p))
    
    # Known rank-53 solution as starting point
    def build_rank53():
        rank = 53
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        U_strassen = np.array([
            [1, 0, 1, 0, 1, -1, 0],
            [0, 0, 0, 0, 1, 0, 1],
            [0, 1, 0, 0, 0, 1, 0],
            [1, 1, 0, 1, 0, 0, -1]
        ])
        V_strassen = np.array([
            [1, 1, 0, -1, 0, 1, 0],
            [0, 0, 1, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 1],
            [1, 0, -1, 0, 1, 0, 1]
        ])
        W_strassen = np.array([
            [1, 0, 0, 1, -1, 0, 1],
            [0, 1, 0, 1, 0, 0, 0],
            [0, 0, 1, 0, 1, 0, 0],
            [1, -1, 1, 0, 0, 1, 0]
        ])
        U_school = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        V_school = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1]
        ])
        W_school = np.array([
            [1, 1, 0, 0],
            [0, 0, 1, 1]
        ])
        
        col_offset = 0
        for block in range(7):
            k0 = 2 * block
            k1 = 2 * block + 1
            for r in range(7):
                c = col_offset + r
                for i in range(2):
                    for j in range(2):
                        U[i*m+j, c] = U_strassen[i*2+j, r]
                for j in range(2):
                    V[j*p+k0, c] = V_strassen[j*2+0, r]
                    V[j*p+k1, c] = V_strassen[j*2+1, r]
                for k in range(2):
                    kk = k0 + k
                    for i in range(2):
                        W[kk*n+i, c] = W_strassen[k*2+i, r]
            col_offset += 7
        
        k_last = 14
        for r in range(4):
            c = col_offset + r
            for i in range(2):
                for j in range(2):
                    U[i*m+j, c] = U_school[i*2+j, r]
            for j in range(2):
                V[j*p+k_last, c] = V_school[j, r]
            for i in range(2):
                W[k_last*n+i, c] = W_school[i, r]
        
        return U, V, W
    
    U53, V53, W53 = build_rank53()
    
    # Verify rank-53 is correct
    def check_decomp(U, V, W):
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - T[i,j,k]) > 1e-6:
                        return False
        return True
    
    # Try to find lower rank solutions using a greedy/optimization approach
    # Strategy: start with rank-53, try to eliminate one factor at a time
    # by solving for a rank-(r-1) approximation
    
    best_rank = 53
    best_U, best_V, best_W = U53, V53, W53
    
    # Try ranks from 52 down
    for target_rank in range(52, 45, -1):
        if time.time() > deadline:
            break
        
        # Multiple random attempts
        for attempt in range(50):
            if time.time() > deadline:
                break
            
            # Initialize with perturbed rank-53 and project
            U_init = U53[:, :target_rank] + 0.01 * np.random.randn(n*m, target_rank)
            V_init = V53[:, :target_rank] + 0.01 * np.random.randn(m*p, target_rank)
            W_init = W53[:, :target_rank] + 0.01 * np.random.randn(p*n, target_rank)
            
            # Optimize to satisfy the 60 constraints
            # Minimize sum of squared errors
            from scipy.optimize import minimize
            
            def objective(params):
                U_r = params[:n*m*target_rank].reshape(n*m, target_rank)
                V_r = params[n*m*target_rank:n*m*target_rank+m*p*target_rank].reshape(m*p, target_rank)
                W_r = params[n*m*target_rank+m*p*target_rank:].reshape(p*n, target_rank)
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.sum(U_r[i*m+j] * V_r[j*p+k] * W_r[k*n+i])
                            err += (val - T[i,j,k])**2
                return err
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                result = minimize(objective, x0, method='BFGS', 
                                  options={'maxiter': 200, 'gtol': 1e-8})
                if result.fun < 1e-8:
                    U_r = result.x[:n*m*target_rank].reshape(n*m, target_rank)
                    V_r = result.x[n*m*target_rank:n*m*target_rank+m*p*target_rank].reshape(m*p, target_rank)
                    W_r = result.x[n*m*target_rank+m*p*target_rank:].reshape(p*n, target_rank)
                    if check_decomp(U_r, V_r, W_r):
                        best_rank = target_rank
                        best_U, best_V, best_W = U_r, V_r, W_r
                        break  # Found valid solution for this rank
            except Exception:
                pass
    
    # Convert to list of lists for return
    def to_list_of_lists(arr):
        return [[float(x) for x in row] for row in arr]
    
    return {"U": to_list_of_lists(best_U), 
            "V": to_list_of_lists(best_V), 
            "W": to_list_of_lists(best_W)}
# EVOLVE-BLOCK-END
