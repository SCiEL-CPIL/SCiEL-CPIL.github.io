"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def _schoolbook():
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


def _verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly represents matrix multiplication."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(rank):
                    s += Fraction(U[i * m + j][c]) * Fraction(V[j * p + k][c]) * Fraction(W[k * n + i][c])
                if s != 1:
                    return False
    return True


def _try_numerical_rank_23_3x3x3(deadline):
    """
    Attempt to find a rank-23 decomposition for 3x3x3 matrix multiplication
    using numerical optimization.
    Returns (U, V, W) as lists of lists of Fractions if successful, else None.
    """
    n = 3
    m = 3
    p = 3
    rank = 23
    
    # Flatten U, V, W into a single vector for optimization
    # U: 9 x 23, V: 9 x 23, W: 9 x 23
    size_U = n * m * rank
    size_V = m * p * rank
    size_W = p * n * rank
    total_size = size_U + size_V + size_W
    
    # Initialize with random values
    np.random.seed(42)
    x0 = np.random.rand(total_size) * 2 - 1
    
    def objective(x):
        U = x[:size_U].reshape(n*m, rank)
        V = x[size_U:size_U+size_V].reshape(m*p, rank)
        W = x[size_U+size_V:].reshape(p*n, rank)
        
        error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    error += (s - 1.0)**2
        return error
    
    # Use Nelder-Mead or L-BFGS-B
    # L-BFGS-B is better for larger problems, but Nelder-Mead might be more robust for small non-convex
    # Given the non-convexity, we might need multiple restarts or a good start.
    # Let's try L-BFGS-B first with a few restarts.
    
    best_x = None
    best_err = float('inf')
    
    for restart in range(5):
        if time.time() > deadline:
            return None
            
        if restart == 0:
            x_init = x0
        else:
            x_init = np.random.rand(total_size) * 2 - 1
            
        res = minimize(objective, x_init, method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-10})
        
        if res.fun < best_err:
            best_err = res.fun
            best_x = res.x
            
        if best_err < 1e-6:
            break
            
    if best_err > 1e-4:
        return None
        
    # Extract and round
    U_flat = best_x[:size_U]
    V_flat = best_x[size_U:size_U+size_V]
    W_flat = best_x[size_U+size_V:]
    
    U = U_flat.reshape(n*m, rank)
    V = V_flat.reshape(m*p, rank)
    W = W_flat.reshape(p*n, rank)
    
    # Round to nearest integer or half-integer
    U_int = np.round(U * 2) / 2
    V_int = np.round(V * 2) / 2
    W_int = np.round(W * 2) / 2
    
    # Convert to list of lists of Fractions
    U_frac = [[Fraction(int(val*2))/2 for val in row] for row in U_int]
    V_frac = [[Fraction(int(val*2))/2 for val in row] for row in V_int]
    W_frac = [[Fraction(int(val*2))/2 for val in row] for row in W_int]
    
    # Verify exactness
    if _verify(U_frac, V_frac, W_frac, n, m, p):
        return U_frac, V_frac, W_frac
        
    return None


def _construct_rank_41(U23, V23, W23):
    """
    Construct a rank-41 decomposition for 3x3x3x5 using a rank-23 decomposition
    for the first 3 columns and schoolbook for the last 2 columns.
    """
    n = 3
    m = 3
    p = 5
    rank_23 = 23
    rank_sb = 9
    total_rank = rank_23 + 2 * rank_sb # 41
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Fill in the rank-23 part for k=0,1,2
    # U is shared
    for i in range(n):
        for j in range(m):
            for c in range(rank_23):
                U[i * m + j][c] = U23[i * m + j][c]
                
    for j in range(m):
        for k in range(3):
            for c in range(rank_23):
                V[j * p + k][c] = V23[j * p + k][c] # Note: V23 has shape 3x3x23? 
                # V23 from _try_numerical... has shape (m*p, rank) = (9, 23)
                # Indexing: V23[j*3+k, c]
                # My V23 variable in the function is (9, 23).
                # So V23[j*3+k, c] is correct.
                
    for k in range(3):
        for i in range(n):
            for c in range(rank_23):
                W[k * n + i][c] = W23[k * n + i][c] # W23 shape (9, 23)
                
    # Fill in the schoolbook part for k=3,4
    # These use columns rank_23 to total_rank-1
    sb_offset = rank_23
    col_idx = sb_offset
    for i in range(n):
        for j in range(m):
            # For k=3
            U[i * m + j][col_idx] = 1
            V[j * p + 3][col_idx] = 1
            W[3 * n + i][col_idx] = 1
            col_idx += 1
            
    col_idx = sb_offset + rank_sb
    for i in range(n):
        for j in range(m):
            # For k=4
            U[i * m + j][col_idx] = 1
            V[j * p + 4][col_idx] = 1
            W[4 * n + i][col_idx] = 1
            col_idx += 1
            
    return {"U": U, "V": V, "W": W}


def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s safety margin
    
    n, m, p = DIMENSIONS
    
    # Try to find a rank-41 decomposition
    result_23 = _try_numerical_rank_23_3x3x3(deadline)
    if result_23 is not None:
        U23, V23, W23 = result_23
        result_41 = _construct_rank_41(U23, V23, W23)
        # Verify the full result
        if _verify(result_41["U"], result_41["V"], result_41["W"], n, m, p):
            return result_41
            
    # Fallback to schoolbook
    return _schoolbook()
# EVOLVE-BLOCK-END
