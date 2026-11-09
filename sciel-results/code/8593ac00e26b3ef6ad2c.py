"""
Rank-minimal exact tensor decomposition for multiplying a 3x4 by 4x6 matrix.
Output factor follows the official trace convention W[k*n+i, r].
"""
import time
import numpy as np
from scipy.optimize import linprog

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def _build_tensor(n, m, p):
    """
    Build the matrix multiplication tensor T[i*m+j, j*p+k, k*n+i] = 1
    We need to find U, V, W such that:
    sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all valid (i,j,k)
    
    This is a bilinear constraint. We can linearize by fixing one matrix.
    For a given rank r, we can use an alternating minimization approach.
    """
    pass

def _try_rank_lp(n, m, p, r, timeout=30):
    """
    Try to find a rank-r decomposition using LP relaxation.
    
    Strategy: Fix W to random values, then solve for U and V via LP.
    The constraint is: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    For fixed W, this becomes: sum_c (U[i*m+j, c] * W[k*n+i, c]) * V[j*p+k, c] = 1
    Let A[i,j,k,c] = U[i*m+j, c] * W[k*n+i, c], then sum_c A[i,j,k,c] * V[j*p+k, c] = 1
    
    This is still bilinear in U and V. Instead, we use an alternating approach:
    1. Initialize W randomly
    2. For fixed W, the problem in U,V is: sum_c U[i*m+j,c] * W[k*n+i,c] * V[j*p+k,c] = 1
       This can be written as a linear system if we treat U*W as intermediate variables.
    """
    start_time = time.time()
    
    # Number of equations: n*m*p
    n_eq = n * m * p
    # Unknowns: U has n*m*r, V has m*p*r, W has p*n*r
    n_unknowns = (n * m + m * p + p * n) * r
    
    # We'll use a simpler approach: alternating minimization with least squares
    # Initialize all matrices randomly
    np.random.seed(42)
    U = np.random.randn(n * m, r) * 0.1
    V = np.random.randn(m * p, r) * 0.1
    W = np.random.randn(p * n, r) * 0.1
    
    best_error = float('inf')
    
    for iteration in range(200):
        if time.time() - start_time > timeout:
            break
        
        # Update U: for each (i,j), minimize sum_{k} (sum_c U[i*m+j,c] * W[k*n+i,c] * V[j*p+k,c] - 1)^2
        # This is equivalent to: for each i, solve for U[i*m:(i+1)*m, :]
        # The constraint involves: sum_{j,k} (sum_c U[i*m+j,c] * W[k*n+i,c] * V[j*p+k,c] - 1)^2
        
        # Let's build the linear system for U given V and W
        # For each i, the variables are U[i*m:(i+1)*m, :] which has m*r unknowns
        # The equations come from all (j,k) pairs
        
        # Actually, let's use a simpler iterative approach
        # For each factor, solve a linear least squares problem
        
        # Update U: Fix V and W. For each i, we have equations for all (j,k):
        # sum_c U[i*m+j,c] * (W[k*n+i,c] * V[j*p+k,c]) = 1
        # Let B[i,j,k,c] = W[k*n+i,c] * V[j*p+k,c]
        # Then sum_c U[i*m+j,c] * B[i,j,k,c] = 1 for all j,k
        # This is a linear system in U[i*m:(i+1)*m, :]
        
        for i in range(n):
            # Variables: U[i*m:(i+1)*m, :] shape (m, r)
            # Equations: for each (j,k), sum_c U[i*m+j,c] * W[k*n+i,c] * V[j*p+k,c] = 1
            # Build matrix A of shape (m*p, m*r) where A[(j,k), (j',c)] = 
            # W[k*n+i,c] * V[j*p+k,c] if j==j', else 0
            
            A = np.zeros((m * p, m * r))
            b = np.ones(m * p)
            for j in range(m):
                for k in range(p):
                    for c in range(r):
                        A[j * p + k, j * r + c] = W[k * n + i, c] * V[j * p + k, c]
            
            # Solve least squares
            if np.linalg.matrix_rank(A) > 0:
                U_new_i, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
                U[i * m:(i + 1) * m, :] = U_new_i.reshape(m, r)
        
        # Update V: Fix U and W. For each j, solve for V[j*p:(j+1)*p, :]
        for j in range(m):
            A = np.zeros((n * p, p * r))
            b = np.ones(n * p)
            for i in range(n):
                for k in range(p):
                    for c in range(r):
                        A[i * p + k, k * r + c] = U[i * m + j, c] * W[k * n + i, c]
            
            if np.linalg.matrix_rank(A) > 0:
                V_new_j, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
                V[j * p:(j + 1) * p, :] = V_new_j.reshape(p, r)
        
        # Update W: Fix U and V. For each k, solve for W[k*n:(k+1)*n, :]
        for k in range(p):
            A = np.zeros((n * m, n * r))
            b = np.ones(n * m)
            for i in range(n):
                for j in range(m):
                    for c in range(r):
                        A[i * m + j, i * r + c] = U[i * m + j, c] * V[j * p + k, c]
            
            if np.linalg.matrix_rank(A) > 0:
                W_new_k, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
                W[k * n:(k + 1) * n, :] = W_new_k.reshape(n, r)
        
        # Compute error
        error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :])
                    error += (val - 1.0) ** 2
        error = np.sqrt(error)
        
        if error < best_error:
            best_error = error
            U_best = U.copy()
            V_best = V.copy()
            W_best = W.copy()
        
        if error < 1e-10:
            break
    
    if best_error < 1e-8:
        # Snap to half-integers
        U_snap = np.round(U_best * 2) / 2
        V_snap = np.round(V_best * 2) / 2
        W_snap = np.round(W_best * 2) / 2
        
        # Verify
        error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U_snap[i * m + j, :] * V_snap[j * p + k, :] * W_snap[k * n + i, :])
                    error += (val - 1.0) ** 2
        error = np.sqrt(error)
        
        if error < 1e-8:
            return U_snap, V_snap, W_snap
    return None

def _construct_schoolbook(n, m, p):
    """Fallback schoolbook decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 120  # 120 second budget
    
    # Try ranks from low to high
    min_possible_rank = max(n * m, m * p, p * n)  # Lower bound
    schoolbook_rank = n * m * p
    
    for r in range(min_possible_rank, schoolbook_rank + 1):
        if time.time() > deadline - 10:
            break
        
        result = _try_rank_lp(n, m, p, r, timeout=20)
        if result is not None:
            U, V, W = result
            # Convert to lists with integer/half-integer entries
            U_list = [[float(x) for x in row] for row in U]
            V_list = [[float(x) for x in row] for row in V]
            W_list = [[float(x) for x in row] for row in W]
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback to schoolbook
    U, V, W = _construct_schoolbook(n, m, p)
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
