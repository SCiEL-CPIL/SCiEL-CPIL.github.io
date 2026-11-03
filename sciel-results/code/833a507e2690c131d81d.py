"""Matrix multiplication 3x3 by 3x6 via tensor rank minimization."""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
import itertools

DIMENSIONS = [3, 3, 6]

def _build_system(n, m, p, r):
    """
    Build the linear system for rank-r tensor decomposition.
    
    We want U (n*m x r), V (m*p x r), W (p*n x r) such that
    sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 if (i,j,k) is the "diagonal" term
    Actually, the tensor T[i,j,k] for matmul is:
    C[i,k] = sum_j A[i,j] * B[j,k]
    So T[i,j,k] = 1 for all (i,j,k) - wait no.
    
    The matrix multiplication tensor T has T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    A rank-r decomposition means T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    
    Wait, let me re-read the convention. The task says:
    "The output factor follows the official trace convention W[k*n+i,r]"
    
    So W is indexed by (k, i) -> row k*n+i, and the factor is W[k*n+i, c].
    
    The tensor for multiplying n x m matrix A by m x p matrix B to get n x p matrix C:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    The tensor T[i,j,k] = 1 for all valid indices (this is the "all-ones" tensor of size n x m x p).
    
    A rank-r decomposition: T[i,j,k] = sum_{c=0}^{r-1} U[i,j,c] * V[j,k,c] * W[k,i,c]
    
    Where:
    - U has shape (n*m, r), indexed U[i*m+j, c]
    - V has shape (m*p, r), indexed V[j*p+k, c]  
    - W has shape (p*n, r), indexed W[k*n+i, c]
    
    So we need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all (i,j,k)
    
    This is a system of n*m*p equations in r*(n*m + m*p + p*n) unknowns.
    """
    num_eqs = n * m * p
    num_vars = r * (n * m + m * p + p * n)
    
    # Each equation is: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # This is a cubic system, not linear!
    
    # For fixed U and V, it's linear in W. But jointly it's nonlinear.
    
    # Strategy: Use alternating minimization or direct nonlinear optimization.
    
    return num_eqs, num_vars

def _validate(U, V, W, n, m, p):
    """Check if the decomposition is exact."""
    r = len(U[0]) if U else 0
    # Check shapes
    assert len(U) == n * m
    assert len(V) == m * p
    assert len(W) == p * n
    assert len(U[0]) == r
    assert len(V[0]) == r
    assert len(W[0]) == r
    
    # Check all entries are integers or half-integers
    for name, mat in [('U', U), ('V', V), ('W', W)]:
        for row in mat:
            for val in row:
                # Check if val is integer or half-integer
                v = float(val)
                if abs(v * 2 - round(v * 2)) > 1e-9:
                    return False, f"{name} has non-integer/half-integer value {val}"
    
    # Check the decomposition
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                for c in range(r):
                    total += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(total - 1.0) > 1e-9:
                    return False, f"Decomposition failed at ({i},{j},{k}): got {total}"
    
    return True, "Valid"

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120.0  # 120 second deadline
    
    # Try increasing ranks starting from the known lower bound
    # The Strassen-like bound suggests rank can be less than n*m*p = 54
    
    # Start with a reasonable lower bound
    # For 3x3 * 3x6, the tensor is 3x3x6 = 54 entries
    # Rank lower bound: at least max(n*m, m*p, p*n) = max(9, 18, 18) = 18
    
    best_result = None
    
    for r in range(18, 55):
        if time.time() > deadline - 10:
            break
        
        # Set up the optimization
        num_U = n * m
        num_V = m * p
        num_W = p * n
        total_vars = r * (num_U + num_V + num_W)
        
        # Initial guess: spread out values
        np.random.seed(42)
        
        # Try multiple random initializations
        found = False
        for attempt in range(3):
            if time.time() > deadline - 5:
                break
            
            x0 = np.random.randn(total_vars) * 0.5
            
            def objective(x):
                # Unpack x into U, V, W
                U_flat = x[:r*num_U]
                V_flat = x[r*num_U:r*num_U+r*num_V]
                W_flat = x[r*num_U+r*num_V:]
                
                U_mat = U_flat.reshape(num_U, r)
                V_mat = V_flat.reshape(num_V, r)
                W_mat = W_flat.reshape(num_W, r)
                
                # Compute sum of squared errors
                error = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0.0
                            for c in range(r):
                                s += U_mat[i*m+j, c] * V_mat[j*p+k, c] * W_mat[k*n+i, c]
                            error += (s - 1.0) ** 2
                return error
            
            # Use Nelder-Mead for small r, L-BFGS-B for larger
            if total_vars < 200:
                result = minimize(objective, x0, method='Nelder-Mead', 
                                  options={'maxiter': 10000, 'xatol': 1e-12, 'fatol': 1e-12})
            else:
                result = minimize(objective, x0, method='L-BFGS-B',
                                  options={'maxiter': 5000, 'gtol': 1e-10})
            
            if result.fun < 1e-8:
                # Extract U, V, W
                U_flat = result.x[:r*num_U]
                V_flat = result.x[r*num_U:r*num_U+r*num_V]
                W_flat = result.x[r*num_U+r*num_V:]
                
                U_mat = U_flat.reshape(num_U, r)
                V_mat = V_flat.reshape(num_V, r)
                W_mat = W_flat.reshape(num_W, r)
                
                # Round to nearest integer or half-integer
                def round_to_half(x):
                    return round(x * 2) / 2.0
                
                U_list = [[round_to_half(U_mat[i, c]) for c in range(r)] for i in range(num_U)]
                V_list = [[round_to_half(V_mat[i, c]) for c in range(r)] for i in range(num_V)]
                W_list = [[round_to_half(W_mat[i, c]) for c in range(r)] for i in range(num_W)]
                
                # Validate
                valid, msg = _validate(U_list, V_list, W_list, n, m, p)
                if valid:
                    return {'U': U_list, 'V': V_list, 'W': W_list}
                
                # If rounding didn't work, try without rounding
                U_list2 = [[float(U_mat[i, c]) for c in range(r)] for i in range(num_U)]
                V_list2 = [[float(V_mat[i, c]) for c in range(r)] for i in range(num_V)]
                W_list2 = [[float(W_mat[i, c]) for c in range(r)] for i in range(num_W)]
                
                valid2, msg2 = _validate(U_list2, V_list2, W_list2, n, m, p)
                if valid2:
                    return {'U': U_list2, 'V': V_list2, 'W': W_list2}
        
        if time.time() > deadline - 15:
            break
    
    # If we didn't find a solution, fall back to schoolbook
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
