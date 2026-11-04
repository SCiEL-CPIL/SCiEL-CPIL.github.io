"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction
from itertools import product
import scipy.optimize

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a low-rank exact tensor decomposition for multiplying a 2x4 matrix by a 4x5 matrix.
    Uses optimization to find a low-rank approximation, then snaps to half-integers
    and verifies exactly.
    """
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS  # n=2, m=4, p=5
    
    def schoolbook_decomposition():
        rank_schoolbook = n * m * p
        U = np.zeros((n*m, rank_schoolbook))
        V = np.zeros((m*p, rank_schoolbook))
        W = np.zeros((p*n, rank_schoolbook))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
        return U, V, W
    
    U_school, V_school, W_school = schoolbook_decomposition()
    
    def verify_half_integer(U, V, W):
        """
        Verify that U, V, W with half-integer entries exactly decompose the tensor.
        """
        U_int = (2 * U).astype(int)
        V_int = (2 * V).astype(int)
        W_int = (2 * W).astype(int)
        
        if not np.allclose(U * 2, U_int) or not np.allclose(V * 2, V_int) or not np.allclose(W * 2, W_int):
            return False
            
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for r in range(U.shape[1]):
                        val += U_int[i*m+j, r] * V_int[j*p+k, r] * W_int[k*n+i, r]
                    if val != 8:
                        return False
        return True

    def loss_function(x, R):
        """
        Loss function for optimization.
        """
        U_size = n * m * R
        V_size = m * p * R
        W_size = p * n * R
        
        U = x[:U_size].reshape(n*m, R)
        V = x[U_size:U_size+V_size].reshape(m*p, R)
        W = x[U_size+V_size:].reshape(p*n, R)
        
        loss = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    loss += (val - 1.0)**2
        return loss

    def grad_function(x, R):
        """
        Gradient of the loss function.
        """
        U_size = n * m * R
        V_size = m * p * R
        W_size = p * n * R
        
        U = x[:U_size].reshape(n*m, R)
        V = x[U_size:U_size+V_size].reshape(m*p, R)
        W = x[U_size+V_size:].reshape(p*n, R)
        
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # residual = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] - 1
                    uvw = U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :]
                    residual = np.sum(uvw) - 1.0
                    
                    # dL/dU[i,j,r] = 2 * residual * V[j,k,r] * W[k,i,r]
                    dU[i*m+j, :] += 2 * residual * V[j*p+k, :] * W[k*n+i, :]
                    dV[j*p+k, :] += 2 * residual * U[i*m+j, :] * W[k*n+i, :]
                    dW[k*n+i, :] += 2 * residual * U[i*m+j, :] * V[j*p+k, :]
        
        return np.concatenate([dU.flatten(), dV.flatten(), dW.flatten()])

    best_rank = n * m * p
    best_decomp = (U_school, V_school, W_school)
    
    # Try ranks from lower bound up to schoolbook
    # Lower bound is max(nm, mp, pn) = 20
    candidate_ranks = [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38]
    
    for target_rank in candidate_ranks:
        if target_rank >= best_rank:
            continue
        if time.time() > deadline - 20:
            break
            
        found = False
        # Try more random initializations with gradient-based optimization
        num_restarts = 30
        for restart in range(num_restarts):
            if time.time() > deadline - 20:
                break
                
            # Random initialization
            U_init = np.random.randn(n*m, target_rank)
            V_init = np.random.randn(m*p, target_rank)
            W_init = np.random.randn(p*n, target_rank)
            
            # Scale to help optimization
            U_init /= np.sqrt(target_rank)
            V_init /= np.sqrt(target_rank)
            W_init /= np.sqrt(target_rank)
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                res = scipy.optimize.minimize(
                    loss_function, x0, args=(target_rank,), 
                    jac=grad_function,
                    method='L-BFGS-B', options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
                )
            except Exception:
                continue
                
            if res.fun < 1e-8:
                # Extract U, V, W
                U_opt = res.x[:n*m*target_rank].reshape(n*m, target_rank)
                V_opt = res.x[n*m*target_rank:n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
                W_opt = res.x[n*m*target_rank + m*p*target_rank:].reshape(p*n, target_rank)
                
                # Snap to half-integers
                U_snap = np.round(U_opt * 2) / 2
                V_snap = np.round(V_opt * 2) / 2
                W_snap = np.round(W_opt * 2) / 2
                
                if verify_half_integer(U_snap, V_snap, W_snap):
                    found = True
                    best_rank = target_rank
                    best_decomp = (U_snap, V_snap, W_snap)
                    break
        
        if found:
            break
    
    if best_rank == n * m * p:
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    else:
        U_best, V_best, W_best = best_decomp
        return {"U": U_best.tolist(), "V": V_best.tolist(), "W": W_best.tolist()}
# EVOLVE-BLOCK-END
