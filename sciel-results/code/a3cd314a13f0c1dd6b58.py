import numpy as np
import time
from itertools import product
import scipy.optimize as opt
from fractions import Fraction
from math import gcd

DIMENSIONS = [2, 3, 3]
_START_TIME = time.time()
TIME_LIMIT = 120  # seconds

def _time_remaining():
    return TIME_LIMIT - 10 - (time.time() - _START_TIME)

def _to_rational(x, tol=1e-8):
    """Convert a float to a rational (integer or half-integer) if close."""
    # Try integers first
    r = round(x)
    if abs(x - r) < tol:
        return Fraction(r)
    # Try half-integers
    r2 = round(2 * x) / 2
    if abs(x - r2) < tol:
        return Fraction(r2)
    # Try quarters
    r4 = round(4 * x) / 4
    if abs(x - r4) < tol:
        return Fraction(r4)
    # Try eighths
    r8 = round(8 * x) / 8
    if abs(x - r8) < tol:
        return Fraction(r8)
    # If none work, try to find a rational with small denominator
    for denom in range(1, 100):
        num = round(x * denom)
        if abs(x - num / denom) < tol:
            return Fraction(num, denom)
    # Give up, return as float
    return None

def _build_schoolbook(n, m, p):
    """Standard rank n*m*p decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W

def _build_strassen_2x2x2():
    """
    Build the rank-7 Strassen decomposition for 2x2x2 matrix multiplication.
    Returns U (4x7), V (4x7), W (4x7) as lists.
    """
    rank = 7
    U = [[0] * rank for _ in range(4)]
    V = [[0] * rank for _ in range(4)]
    W = [[0] * rank for _ in range(4)]
    
    # U: A indices: 0=A00, 1=A01, 2=A10, 3=A11
    # V: B indices: 0=B00, 1=B01, 2=B10, 3=B11
    # W: C indices: 0=C00, 1=C01, 2=C10, 3=C11
    
    # M1: (A00+A11)(B00+B11)
    U[0][0] += 1
    U[3][0] += 1
    V[0][0] += 1
    V[3][0] += 1
    W[0][0] += 1
    W[3][0] += 1
    
    # M2: (A10+A11)B00
    U[2][1] += 1
    U[3][1] += 1
    V[0][1] += 1
    W[2][1] += 1
    W[3][1] -= 1
    
    # M3: A00(B01-B11)
    U[0][2] += 1
    V[1][2] += 1
    V[3][2] -= 1
    W[1][2] += 1
    W[3][2] += 1
    
    # M4: A11(B10-B00)
    U[3][3] += 1
    V[2][3] += 1
    V[0][3] -= 1
    W[0][3] += 1
    W[2][3] += 1
    
    # M5: (A00+A01)B11
    U[0][4] += 1
    U[1][4] += 1
    V[3][4] += 1
    W[0][4] -= 1
    W[1][4] += 1
    
    # M6: (A10-A00)(B00+B01)
    U[2][5] += 1
    U[0][5] -= 1
    V[0][5] += 1
    V[1][5] += 1
    W[3][5] += 1
    
    # M7: (A01-A11)(B10+B11)
    U[1][6] += 1
    U[3][6] -= 1
    V[2][6] += 1
    V[3][6] += 1
    W[0][6] += 1
    
    return U, V, W

def _build_decomposed_2x3x3():
    """
    Decompose 2x3x3 into:
    - Part 1: 2x2x2 (columns j=0,1; rows k=0,1) -> Strassen rank 7
    - Part 2: 2x2x1 (columns j=0,1; row k=2) -> schoolbook rank 4
    - Part 3: 2x1x3 (column j=2; rows k=0,1,2) -> schoolbook rank 6
    Total rank: 17
    """
    n, m, p = 2, 3, 3
    rank = 17
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Part 1: 2x2x2 block
    U1, V1, W1 = _build_strassen_2x2x2()
    for r in range(7):
        for a_idx in range(4):
            i_local = a_idx // 2
            j_local = a_idx % 2
            a_global = i_local * 3 + j_local
            U[a_global][r] = U1[a_idx][r]
        for b_idx in range(4):
            j_local = b_idx // 2
            k_local = b_idx % 2
            b_global = j_local * 3 + k_local
            V[b_global][r] = V1[b_idx][r]
        for c_idx in range(4):
            k_local = c_idx // 2
            i_local = c_idx % 2
            c_global = k_local * 2 + i_local
            W[c_global][r] = W1[c_idx][r]
    
    # Part 2: 2x2x1 block (j in {0,1}, k=2)
    col_offset = 7
    for i in range(2):
        for j in range(2):
            col = col_offset + i * 2 + j
            a_global = i * 3 + j
            b_global = j * 3 + 2
            c_global = 4 + i
            U[a_global][col] = 1
            V[b_global][col] = 1
            W[c_global][col] = 1
    
    # Part 3: 2x1x3 block (j=2, k in {0,1,2})
    col_offset = 11
    for i in range(2):
        for k in range(3):
            col = col_offset + i * 3 + k
            a_global = i * 3 + 2
            b_global = 6 + k
            c_global = k * 2 + i
            U[a_global][col] = 1
            V[b_global][col] = 1
            W[c_global][col] = 1
    
    return U, V, W

def _verify_fast(U, V, W, n, m, p, tol=1e-9):
    """Fast vectorized verification."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.dot(U_arr[a], V_arr[b] * W_arr[c])
                if abs(val - 1) > tol:
                    return False
    return True

def _verify_rational(U, V, W, n, m, p):
    """Verify that U, V, W have integer or half-integer entries and decompose correctly."""
    # Check if all entries are rational (integer or half-integer)
    for name, mat in [('U', U), ('V', V), ('W', W)]:
        for row in mat:
            for val in row:
                if val is None or not isinstance(val, (int, float, Fraction)):
                    return False
                if isinstance(val, float):
                    if not np.isfinite(val):
                        return False
                    r = _to_rational(val)
                    if r is None:
                        return False
                elif isinstance(val, Fraction):
                    # Check if denominator is 1 or 2 (integer or half-integer)
                    if val.denominator not in (1, 2):
                        return False
    # Verify decomposition
    U_arr = np.array([[float(v) if v is not None else 0 for v in row] for row in U], dtype=float)
    V_arr = np.array([[float(v) if v is not None else 0 for v in row] for row in V], dtype=float)
    W_arr = np.array([[float(v) if v is not None else 0 for v in row] for row in W], dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.dot(U_arr[a], V_arr[b] * W_arr[c])
                if abs(val - 1) > 1e-6:
                    return False
    return True

def _round_to_rational_matrix(mat, tol=1e-8):
    """Round a matrix to integer or half-integer values."""
    result = []
    for row in mat:
        new_row = []
        for val in row:
            r = _to_rational(val, tol)
            if r is not None:
                new_row.append(r)
            else:
                new_row.append(None)
        result.append(new_row)
    return result

def _residual_func(params, n, m, p):
    """
    Residual function for scipy.optimize.least_squares.
    params: flattened array of [U, V, W]
    """
    n, m, p = 2, 3, 3
    rank = params.size // (n*m + m*p + p*n)
    
    U_size = n * m * rank
    V_size = m * p * rank
    W_size = p * n * rank
    
    U_flat = params[:U_size]
    V_flat = params[U_size:U_size+V_size]
    W_flat = params[U_size+V_size:U_size+V_size+W_size]
    
    U = U_flat.reshape((n*m, rank))
    V = V_flat.reshape((m*p, rank))
    W = W_flat.reshape((p*n, rank))
    
    residuals = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.dot(U[a], V[b] * W[c])
                residuals.append(val - 1.0)
                
    return np.array(residuals)

def _residual_func_with_regularization(params, n, m, p, reg_weight=0.01):
    """
    Residual function with regularization to encourage sparse/rational solutions.
    """
    n, m, p = 2, 3, 3
    rank = params.size // (n*m + m*p + p*n)
    
    U_size = n * m * rank
    V_size = m * p * rank
    W_size = p * n * rank
    
    U_flat = params[:U_size]
    V_flat = params[U_size:U_size+V_size]
    W_flat = params[U_size+V_size:U_size+V_size+W_size]
    
    U = U_flat.reshape((n*m, rank))
    V = V_flat.reshape((m*p, rank))
    W = W_flat.reshape((p*n, rank))
    
    residuals = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.dot(U[a], V[b] * W[c])
                residuals.append(val - 1.0)
    
    # Add regularization to encourage values close to integers or half-integers
    for mat in [U, V, W]:
        for row in mat:
            for val in row:
                # Distance to nearest integer
                dist_int = abs(val - round(val))
                # Distance to nearest half-integer
                dist_half = abs(val - (round(2 * val) / 2))
                min_dist = min(dist_int, dist_half)
                residuals.append(reg_weight * min_dist)
    
    return np.array(residuals)

def _try_rank_reduction_scipy(n, m, p, target_rank, max_restarts=10):
    """
    Try to find a rank-target_rank decomposition using scipy.optimize.least_squares.
    """
    if _time_remaining() < 20:
        return None
    
    U_size = n * m * target_rank
    V_size = m * p * target_rank
    W_size = p * n * target_rank
    total_size = U_size + V_size + W_size
    
    best_solution = None
    best_cost = float('inf')
    
    # Try multiple initialization strategies
    init_strategies = []
    
    # Strategy 1: Random initialization
    for _ in range(min(3, max_restarts)):
        init_strategies.append(('random', np.random.randn(total_size) * 0.5))
    
    # Strategy 2: Sparse initialization with some entries at 0, 1, -1
    for _ in range(min(3, max_restarts)):
        x0 = np.zeros(total_size)
        # Randomly set some entries to 1 or -1
        num_nonzero = int(total_size * 0.3)
        indices = np.random.choice(total_size, num_nonzero, replace=False)
        x0[indices] = np.random.choice([-1, 1], num_nonzero)
        init_strategies.append(('sparse', x0))
    
    # Strategy 3: Initialize from rank-17 solution with one column zeroed
    U17, V17, W17 = _build_decomposed_2x3x3()
    U17_arr = np.array(U17, dtype=float)
    V17_arr = np.array(V17, dtype=float)
    W17_arr = np.array(W17, dtype=float)
    
    # Try removing each column
    for col_to_remove in range(17):
        if _time_remaining() < 5:
            break
        U_init = np.delete(U17_arr, col_to_remove, axis=1)
        V_init = np.delete(V17_arr, col_to_remove, axis=1)
        W_init = np.delete(W17_arr, col_to_remove, axis=1)
        
        # Pad to target_rank if needed
        if U_init.shape[1] < target_rank:
            pad_cols = target_rank - U_init.shape[1]
            U_init = np.hstack([U_init, np.zeros((U_init.shape[0], pad_cols))])
            V_init = np.hstack([V_init, np.zeros((V_init.shape[0], pad_cols))])
            W_init = np.hstack([W_init, np.zeros((W_init.shape[0], pad_cols))])
        
        x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        init_strategies.append(('from17', x0))
    
    # Strategy 4: Use the rank-17 solution as starting point with noise
    U17_flat = np.array(U17, dtype=float).flatten()
    V17_flat = np.array(V17, dtype=float).flatten()
    W17_flat = np.array(W17, dtype=float).flatten()
    base_init = np.concatenate([U17_flat, V17_flat, W17_flat])
    
    for noise_level in [0.1, 0.05, 0.01]:
        if _time_remaining() < 5:
            break
        x0 = base_init + np.random.randn(total_size) * noise_level
        init_strategies.append(('noisy17', x0))
    
    # Try each initialization
    for strategy_name, x0 in init_strategies:
        if _time_remaining() < 5:
            break
        
        try:
            # Try with and without regularization
            for use_reg in [False, True]:
                if _time_remaining() < 5:
                    break
                
                if use_reg:
                    result = opt.least_squares(
                        _residual_func_with_regularization, 
                        x0, 
                        args=(n, m, p, 0.01),
                        method='trf',
                        max_nfev=2000,
                        ftol=1e-12,
                        xtol=1e-12,
                        gtol=1e-12
                    )
                else:
                    result = opt.least_squares(
                        _residual_func, 
                        x0, 
                        args=(n, m, p),
                        method='trf',
                        max_nfev=2000,
                        ftol=1e-12,
                        xtol=1e-12,
                        gtol=1e-12
                    )
                
                if result.cost < best_cost:
                    best_cost = result.cost
                    best_solution = result.x.copy()
                
                # Check if we have a good solution
                if result.cost < 1e-10:
                    # Reconstruct U, V, W
                    U_flat = result.x[:U_size]
                    V_flat = result.x[U_size:U_size+V_size]
                    W_flat = result.x[U_size+V_size:U_size+V_size+W_size]
                    
                    U = U_flat.reshape((n*m, target_rank))
                    V = V_flat.reshape((m*p, target_rank))
                    W = W_flat.reshape((p*n, target_rank))
                    
                    # Try to round to rational values
                    U_rational = _round_to_rational_matrix(U.tolist())
                    V_rational = _round_to_rational_matrix(V.tolist())
                    W_rational = _round_to_rational_matrix(W.tolist())
                    
                    # Verify rational solution
                    if _verify_rational(U_rational, V_rational, W_rational, n, m, p):
                        return U_rational, V_rational, W_rational
                    
                    # Also try the unrounded solution
                    if _verify_fast(U.tolist(), V.tolist(), W.tolist(), n, m, p):
                        return U.tolist(), V.tolist(), W.tolist()
                        
        except Exception as e:
            continue
            
    # If we have a best solution, try to polish it
    if best_solution is not None and best_cost < 1e-8:
        U_flat = best_solution[:U_size]
        V_flat = best_solution[U_size:U_size+V_size]
        W_flat = best_solution[U_size+V_size:U_size+V_size+W_size]
        
        U = U_flat.reshape((n*m, target_rank))
        V = V_flat.reshape((m*p, target_rank))
        W = W_flat.reshape((p*n, target_rank))
        
        U_rational = _round_to_rational_matrix(U.tolist())
        V_rational = _round_to_rational_matrix(V.tolist())
        W_rational = _round_to_rational_matrix(W.tolist())
        
        if _verify_rational(U_rational, V_rational, W_rational, n, m, p):
            return U_rational, V_rational, W_rational
            
    return None

def construct():
    n, m, p = DIMENSIONS
    
    # First, try the known rank-17 decomposition
    try:
        U, V, W = _build_decomposed_2x3x3()
        if _verify_fast(U, V, W, n, m, p):
            best_U, best_V, best_W = U, V, W
            best_rank = len(U[0])
            
            # Try to reduce rank to 16 using scipy
            if best_rank > 16:
                result = _try_rank_reduction_scipy(n, m, p, 16)
                if result is not None:
                    U_opt, V_opt, W_opt = result
                    # Verify with rational check first, then fast check
                    if _verify_rational(U_opt, V_opt, W_opt, n, m, p) or _verify_fast(U_opt, V_opt, W_opt, n, m, p):
                        best_U, best_V, best_W = U_opt, V_opt, W_opt
                        best_rank = 16
            
            return {"U": best_U, "V": best_V, "W": best_W}
    except Exception:
        pass
    
    # Fallback to schoolbook (rank 18)
    U, V, W = _build_schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}
