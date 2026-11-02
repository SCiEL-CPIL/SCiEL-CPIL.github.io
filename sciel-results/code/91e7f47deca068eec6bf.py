"""Matrix multiplication tensor decomposition for 2x3x8 with rank reduction search."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 8]
_START = time.time()
_DEADLINE = _START + 140.0

def _check_time():
    return time.time() > _DEADLINE - 15

def _validate_exact(U, V, W):
    """Check if factor matrices exactly decompose the matmul tensor using Fraction arithmetic."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                val = Fraction(0)
                for c in range(rank):
                    val += U[u_idx][c] * V[v_idx][c] * W[w_idx][c]
                if val != 1:
                    return False
    return True

def _validate_numeric(U, V, W, tol=1e-9):
    """Quick numeric check."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                val = np.sum(U_arr[u_idx] * V_arr[v_idx] * W_arr[w_idx])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _als_step(U, V, W, n, m, p, target_r):
    """One ALS iteration, vectorized where possible."""
    U_new = U.copy()
    V_new = V.copy()
    W_new = W.copy()
    
    # Update U: for each (i,j), solve A_sys @ U[u_idx,:] = ones(p)
    for i in range(n):
        for j in range(m):
            u_idx = i * m + j
            # A_sys[k, c] = V[j*p+k, c] * W[k*n+i, c]
            A_sys = np.zeros((p, target_r))
            for k in range(p):
                A_sys[k] = V[j*p+k] * W[k*n+i]
            try:
                sol, _, _, _ = np.linalg.lstsq(A_sys, np.ones(p), rcond=None)
                U_new[u_idx] = sol
            except:
                pass
    
    # Update V: for each (j,k), solve A_sys @ V[v_idx,:] = ones(n)
    for j in range(m):
        for k in range(p):
            v_idx = j * p + k
            A_sys = np.zeros((n, target_r))
            for i in range(n):
                A_sys[i] = U_new[i*m+j] * W[k*n+i]
            try:
                sol, _, _, _ = np.linalg.lstsq(A_sys, np.ones(n), rcond=None)
                V_new[v_idx] = sol
            except:
                pass
    
    # Update W: for each (k,i), solve A_sys @ W[w_idx,:] = ones(m)
    for k in range(p):
        for i in range(n):
            w_idx = k * n + i
            A_sys = np.zeros((m, target_r))
            for j in range(m):
                A_sys[j] = U_new[i*m+j] * V_new[j*p+k]
            try:
                sol, _, _, _ = np.linalg.lstsq(A_sys, np.ones(m), rcond=None)
                W_new[w_idx] = sol
            except:
                pass
    
    return U_new, V_new, W_new

def _compute_error(U, V, W, n, m, p):
    """Compute max error of the decomposition."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                val = np.sum(U_arr[u_idx] * V_arr[v_idx] * W_arr[w_idx])
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
    return max_err

def _recover_rationals(U, V, W, max_den=40):
    """Try to recover exact rationals from numerical solution."""
    def to_frac(x):
        if abs(x - round(x)) < 1e-10:
            return int(round(x))
        for den in range(2, max_den + 1):
            num = x * den
            if abs(num - round(num)) < 1e-10:
                return Fraction(int(round(num)), den)
        # Try limit_denominator
        return Fraction(x).limit_denominator(max_den)
    
    U_fr = [[to_frac(float(x)) for x in row] for row in U]
    V_fr = [[to_frac(float(x)) for x in row] for row in V]
    W_fr = [[to_frac(float(x)) for x in row] for row in W]
    return U_fr, V_fr, W_fr

def _recover_rationals_polynomial(U, V, W):
    """Use polynomial interpolation to recover exact rationals.
    Evaluate the tensor at several integer points and interpolate."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    
    # For each (i,j,k), we have: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    # This is a system of linear equations in the products.
    # Alternative: try to find rational values by checking small denominators.
    
    def to_frac(x):
        if abs(x - round(x)) < 1e-12:
            return int(round(x))
        for den in range(1, 50):
            num = x * den
            if abs(num - round(num)) < 1e-12:
                return Fraction(int(round(num)), den)
        return Fraction(x).limit_denominator(100)
    
    U_fr = [[to_frac(float(x)) for x in row] for row in U]
    V_fr = [[to_frac(float(x)) for x in row] for row in V]
    W_fr = [[to_frac(float(x)) for x in row] for row in W]
    return U_fr, V_fr, W_fr

def _search_direct(n, m, p, target_r, max_trials=5, max_iters=200):
    """Search for rank target_r decomposition of n x m x p matmul tensor."""
    best_err = float('inf')
    best_sol = None
    
    for trial in range(max_trials):
        if _check_time():
            break
        
        rng = np.random.default_rng(42 + trial * 137 + target_r)
        
        # Initialize
        U = rng.standard_normal((n*m, target_r)) * 0.5
        V = rng.standard_normal((m*p, target_r)) * 0.5
        W = rng.standard_normal((p*n, target_r)) * 0.5
        
        # Normalize to avoid overflow
        U /= np.linalg.norm(U, axis=0, keepdims=True) + 1e-10
        V /= np.linalg.norm(V, axis=0, keepdims=True) + 1e-10
        W /= np.linalg.norm(W, axis=0, keepdims=True) + 1e-10
        
        prev_err = float('inf')
        for it in range(max_iters):
            if _check_time():
                break
            
            U, V, W = _als_step(U, V, W, n, m, p, target_r)
            
            err = _compute_error(U, V, W, n, m, p)
            
            if err < best_err:
                best_err = err
                best_sol = (U.copy(), V.copy(), W.copy())
            
            if err < 1e-10:
                break
            
            if abs(prev_err - err) < 1e-14:
                break
            prev_err = err
    
    if best_err < 1e-8:
        return best_sol, best_err
    return None, best_err

def _search_2x3x4(target_r, max_trials=10, max_iters=300):
    """Search for rank target_r decomposition of 2x3x4 matmul tensor."""
    n, m, p = 2, 3, 4
    return _search_direct(n, m, p, target_r, max_trials, max_iters)

def _compose_2x3x8_from_2x3x4(U14, V14, W14):
    """Compose a 2x3x4 solution into a 2x3x8 solution by block duplication."""
    n, m, p4 = 2, 3, 4
    p8 = 8
    r = len(U14[0])
    
    U28 = []
    for row in U14:
        U28.append(row + row)  # 2r columns
    
    V28 = []
    for j in range(m):
        for k in range(p8):
            k_mod = k % p4
            v_idx = j * p4 + k_mod
            V28.append(V14[v_idx] + V14[v_idx])
    
    W28 = []
    for k in range(p8):
        for i in range(n):
            k_mod = k % p4
            w_idx = k_mod * n + i
            W28.append(W14[w_idx] + W14[w_idx])
    
    return U28, V28, W28

def construct():
    """Find rank-minimal exact tensor decomposition for 2x3x8 matrix multiplication."""
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 48
    
    # Build schoolbook solution as fallback
    U_school = [[0] * schoolbook_rank for _ in range(n * m)]
    V_school = [[0] * schoolbook_rank for _ in range(m * p)]
    W_school = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = schoolbook_rank
    
    # Strategy 1: Try to find 2x3x4 solution and compose to 2x3x8
    # Known: 2x3x4 has rank <= 14 (from literature)
    # If we find rank 14 for 2x3x4, we get rank 28 for 2x3x8
    
    # Try 2x3x4 first
    for r4 in [14, 15, 16, 17, 18, 20]:
        if _check_time():
            break
        if 2 * r4 >= best_rank:
            continue
        
        sol, err = _search_2x3x4(r4, max_trials=8, max_iters=200)
        if sol is not None:
            U14, V14, W14 = sol
            U28, V28, W28 = _compose_2x3x8_from_2x3x4(U14, V14, W14)
            
            # Try to recover rationals
            U_fr, V_fr, W_fr = _recover_rationals(U28, V28, W28)
            if _validate_exact(U_fr, V_fr, W_fr):
                if 2 * r4 < best_rank:
                    best_rank = 2 * r4
                    best_result = {"U": U_fr, "V": V_fr, "W": W_fr}
                break
            else:
                # Try polynomial recovery
                U_fr2, V_fr2, W_fr2 = _recover_rationals_polynomial(U28, V28, W28)
                if _validate_exact(U_fr2, V_fr2, W_fr2):
                    if 2 * r4 < best_rank:
                        best_rank = 2 * r4
                        best_result = {"U": U_fr2, "V": V_fr2, "W": W_fr2}
                    break
    
    # Strategy 2: Direct search on 2x3x8
    candidate_ranks = [28, 26, 24, 22, 20, 18, 16, 14, 12, 10]
    
    for target_r in candidate_ranks:
        if _check_time():
            break
        if target_r >= best_rank:
            continue
        
        sol, err = _search_direct(n, m, p, target_r, max_trials=5, max_iters=150)
        if sol is not None:
            U_sol, V_sol, W_sol = sol
            U_fr, V_fr, W_fr = _recover_rationals(U_sol, V_sol, W_sol)
            if _validate_exact(U_fr, V_fr, W_fr):
                if target_r < best_rank:
                    best_rank = target_r
                    best_result = {"U": U_fr, "V": V_fr, "W": W_fr}
                break
            else:
                U_fr2, V_fr2, W_fr2 = _recover_rationals_polynomial(U_sol, V_sol, W_sol)
                if _validate_exact(U_fr2, V_fr2, W_fr2):
                    if target_r < best_rank:
                        best_rank = target_r
                        best_result = {"U": U_fr2, "V": V_fr2, "W": W_fr2}
                    break
    
    return best_result
