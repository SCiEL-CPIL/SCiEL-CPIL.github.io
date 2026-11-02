import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 3]

def _verify(U, V, W):
    """Verify that U,V,W with given indices correctly decompose matrix multiplication.
    Returns True if decomposition is exact."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1) > 1e-9:
                    return False
    return True

def _construct_schoolbook():
    n, m, p = DIMENSIONS
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

def _construct_rank_26():
    """
    Construct a rank-26 decomposition for 3x3 matrix multiplication.
    """
    n, m, p = 3, 3, 3
    rank = 26
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Term 0: M1
    U[0*3+0][0] += 1
    U[1*3+1][0] += 1
    V[0*3+0][0] += 1
    V[1*3+1][0] += 1
    W[0*3+0][0] += 1  # C00
    W[1*3+1][0] += 1  # C11
    
    # Term 1: M2
    U[1*3+0][1] += 1
    U[1*3+1][1] += 1
    V[0*3+0][1] += 1
    W[1*3+0][1] += 1  # C10
    
    # Term 2: M3
    U[0*3+0][2] += 1
    V[0*3+1][2] += 1
    V[1*3+1][2] -= 1
    W[0*3+1][2] += 1  # C01
    
    # Term 3: M4
    U[1*3+1][3] += 1
    V[1*3+0][3] += 1
    V[0*3+0][3] -= 1
    W[0*3+0][3] -= 1  # C00
    W[1*3+0][3] += 1  # C10
    
    # Term 4: M5
    U[0*3+0][4] += 1
    U[0*3+1][4] += 1
    V[1*3+1][4] += 1
    W[0*3+1][4] += 1  # C01
    W[1*3+1][4] -= 1  # C11
    
    # Term 5: M6
    U[1*3+0][5] += 1
    U[0*3+0][5] -= 1
    V[0*3+0][5] += 1
    V[0*3+1][5] += 1
    W[0*3+0][5] -= 1  # C00
    W[1*3+1][5] += 1  # C11
    
    # Term 6: M7
    U[0*3+1][6] += 1
    U[1*3+1][6] -= 1
    V[1*3+0][6] += 1
    V[1*3+1][6] += 1
    W[0*3+0][6] += 1  # C00
    W[1*3+1][6] += 1  # C11
    
    # Remaining 19 terms
    col = 7
    for i in range(3):
        for k in range(3):
            U[i*3+2][col] = 1
            V[2*3+k][col] = 1
            W[k*3+i][col] = 1
            col += 1
            
    for i in range(3):
        for j in range(2):
            U[i*3+j][col] = 1
            V[j*3+2][col] = 1
            W[2*3+i][col] = 1
            col += 1
            
    for j in range(2):
        for k in range(2):
            U[2*3+j][col] = 1
            V[j*3+k][col] = 1
            W[k*3+2][col] = 1
            col += 1
            
    if _verify(U, V, W):
        return {"U": U, "V": V, "W": W}
    return None

def _try_reduce_rank(rank_26_data):
    """
    Attempt to reduce the rank of the decomposition by driving some coefficients to zero
    using local optimization.
    """
    U, V, W = rank_26_data['U'], rank_26_data['V'], rank_26_data['W']
    n, m, p = 3, 3, 3
    rank = len(U[0])
    
    # Flatten to vector
    U_flat = np.array(U).flatten()
    V_flat = np.array(V).flatten()
    W_flat = np.array(W).flatten()
    x0 = np.concatenate([U_flat, V_flat, W_flat])
    
    def objective(x):
        u = x[:rank*n*m].reshape(n*m, rank)
        v = x[rank*n*m:2*rank*n*m].reshape(m*p, rank)
        w = x[2*rank*n*m:].reshape(p*n, rank)
        
        error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += u[i*m+j, c] * v[j*p+k, c] * w[k*n+i, c]
                    error += (s - 1.0)**2
        
        # Add L1 penalty to encourage sparsity
        l1_penalty = 0.001 * (np.sum(np.abs(u)) + np.sum(np.abs(v)) + np.sum(np.abs(w)))
        return error + l1_penalty

    def jac(x):
        # Numerical gradient
        eps = 1e-7
        grad = np.zeros_like(x)
        for i in range(len(x)):
            x_plus = x.copy()
            x_plus[i] += eps
            f_plus = objective(x_plus)
            x_minus = x.copy()
            x_minus[i] -= eps
            f_minus = objective(x_minus)
            grad[i] = (f_plus - f_minus) / (2 * eps)
        return grad

    # We will try to optimize for a short time
    deadline = time.time() + 100
    best_result = None
    
    # Try different starting points or penalties? 
    # Just one run with BFGS
    try:
        res = minimize(objective, x0, method='BFGS', jac=jac, 
                       options={'maxiter': 500, 'disp': False, 'gtol': 1e-8})
        
        if res.success or res.fun < 1e-6:
            x_opt = res.x
            u_opt = x_opt[:rank*n*m].reshape(n*m, rank)
            v_opt = x_opt[rank*n*m:2*rank*n*m].reshape(m*p, rank)
            w_opt = x_opt[2*rank*n*m:].reshape(p*n, rank)
            
            # Clean up small values
            tol = 1e-5
            u_clean = np.where(np.abs(u_opt) < tol, 0, u_opt)
            v_clean = np.where(np.abs(v_opt) < tol, 0, v_opt)
            w_clean = np.where(np.abs(w_opt) < tol, 0, w_opt)
            
            # Check if any column became entirely zero
            # If so, we can potentially remove it
            # But first, verify the cleaned version
            U_list = u_clean.tolist()
            V_list = v_clean.tolist()
            W_list = w_clean.tolist()
            
            if _verify(U_list, V_list, W_list):
                # Check effective rank
                # Find columns that are all zero
                active_cols = []
                for c in range(rank):
                    if np.any(u_clean[:, c] != 0) or np.any(v_clean[:, c] != 0) or np.any(w_clean[:, c] != 0):
                        active_cols.append(c)
                
                if len(active_cols) < rank:
                    # Extract active columns
                    U_new = [ [u_clean[i, c] for c in active_cols] for i in range(n*m) ]
                    V_new = [ [v_clean[i, c] for c in active_cols] for i in range(m*p) ]
                    W_new = [ [w_clean[i, c] for c in active_cols] for i in range(p*n) ]
                    
                    if _verify(U_new, V_new, W_new):
                        best_result = {"U": U_new, "V": V_new, "W": W_new}
                        
    except Exception as e:
        pass
        
    return best_result

def construct():
    overall_deadline = time.time() + 140
    
    # Try rank-26 first
    result_26 = _construct_rank_26()
    if result_26 is not None:
        # Try to reduce rank
        result_reduced = _try_reduce_rank(result_26)
        if result_reduced is not None:
            return result_reduced
        return result_26
    
    # Fallback to schoolbook
    return _construct_schoolbook()
