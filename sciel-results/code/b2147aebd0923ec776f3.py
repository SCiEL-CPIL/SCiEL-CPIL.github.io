"""Systematic rank-1 peeling for 2x3x5 matrix multiplication tensor."""
import numpy as np
from fractions import Fraction
import scipy.optimize as opt

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank = n*m*p = 30
    # Target rank = 28
    
    # Build the tensor T
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
                
    target_rank = 28
    
    # Function to compute the tensor from factors
    def compute_tensor(U, V, W):
        # U: (nm, r), V: (mp, r), W: (pn, r)
        # Result: (nm, mp, pn)
        # Einsum: 'ir,jr,kr->ijk'
        return np.einsum('ir,jr,kr->ijk', U, V, W)
    
    # Function to flatten factors
    def flatten_factors(U, V, W):
        return np.concatenate([U.flatten(), V.flatten(), W.flatten()])
    
    # Function to unflatten factors
    def unflatten_factors(x):
        size_U = n * m * target_rank
        size_V = m * p * target_rank
        size_W = p * n * target_rank
        
        U_flat = x[:size_U]
        V_flat = x[size_U:size_U+size_V]
        W_flat = x[size_U+size_V:size_U+size_V+size_W]
        
        U = U_flat.reshape(n*m, target_rank)
        V = V_flat.reshape(m*p, target_rank)
        W = W_flat.reshape(p*n, target_rank)
        return U, V, W
    
    # Objective function
    def objective(x):
        U, V, W = unflatten_factors(x)
        T_est = compute_tensor(U, V, W)
        diff = T - T_est
        return np.sum(diff**2)
    
    # Gradient function
    def gradient(x):
        U, V, W = unflatten_factors(x)
        T_est = compute_tensor(U, V, W)
        diff = T_est - T
        
        # d(T_est)/dU = einsum('jk,ir,jr,kr->ir', diff, V, W) ?? 
        # T_est[i,j,k] = sum_r U[i,r] V[j,r] W[k,r]
        # d(T_est[i,j,k])/dU[i',r'] = delta_{i,i'} V[j,r'] W[k,r']
        # grad_U[i',r'] = sum_{j,k} diff[i',j,k] V[j,r'] W[k,r']
        # grad_U = einsum('ijk,jr,kr->ir', diff, V, W)
        
        grad_U = np.einsum('ijk,jr,kr->ir', diff, V, W)
        grad_V = np.einsum('ijk,ir,kr->jr', diff, U, W)
        grad_W = np.einsum('ijk,ir,jr->kr', diff, U, V)
        
        return np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
    
    best_solution = None
    best_err = float('inf')
    
    # Try multiple random initializations
    for trial in range(10):
        np.random.seed(42 + trial)
        # Initialize with small random values
        U0 = np.random.randn(n*m, target_rank) * 0.1
        V0 = np.random.randn(m*p, target_rank) * 0.1
        W0 = np.random.randn(p*n, target_rank) * 0.1
        
        x0 = flatten_factors(U0, V0, W0)
        
        try:
            result = opt.minimize(objective, x0, jac=gradient, method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-10})
            if result.fun < best_err:
                best_err = result.fun
                best_solution = result.x
        except Exception:
            continue
            
    # Check if we found a good solution
    if best_solution is not None and best_err < 1e-4:
        U, V, W = unflatten_factors(best_solution)
        
        # Round to half-integers
        # Multiply by 2, round, divide by 2
        U_rounded = np.round(U * 2) / 2.0
        V_rounded = np.round(V * 2) / 2.0
        W_rounded = np.round(W * 2) / 2.0
        
        # Verify exactness
        T_check = compute_tensor(U_rounded, V_rounded, W_rounded)
        if np.allclose(T_check, T, atol=1e-9):
            # Convert to list of lists
            U_list = U_rounded.tolist()
            V_list = V_rounded.tolist()
            W_list = W_rounded.tolist()
            return {"U": U_list, "V": V_list, "W": W_list}
            
    # Fall back to schoolbook
    U = [[0] * (n*m*p) for _ in range(n*m)]
    V = [[0] * (n*m*p) for _ in range(m*p)]
    W = [[0] * (n*m*p) for _ in range(p*n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][column] = 1
                V[j*p+k][column] = 1
                W[k*n+i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
