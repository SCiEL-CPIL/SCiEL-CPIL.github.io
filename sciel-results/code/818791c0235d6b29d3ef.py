import time
import sys
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def _verify_exact(U, V, W, n, m, p):
    """Verify exact decomposition using integer/rational arithmetic."""
    rank = len(U[0])
    U_frac = [[Fraction(x) for x in row] for row in U]
    V_frac = [[Fraction(x) for x in row] for row in V]
    W_frac = [[Fraction(x) for x in row] for row in W]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = Fraction(0)
                for r in range(rank):
                    total += U_frac[i*m+j][r] * V_frac[j*p+k][r] * W_frac[k*n+i][r]
                if total != 1:
                    return False
    return True

def _verify_float(U, V, W, n, m, p, tol=1e-9):
    """Fast float verification."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    U_t = U_arr.reshape(n, m, rank)
    V_t = V_arr.reshape(m, p, rank)
    W_t = W_arr.reshape(p, n, rank)
    
    T = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
    
    if not np.allclose(T, 1.0, atol=tol):
        return False
    return True

def _build_strassen_rank39(n, m, p):
    """Build the rank-39 Strassen-based decomposition."""
    rank = 39
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_term(u_indices, v_indices, w_indices, u_coeffs=None, v_coeffs=None, w_coeffs=None):
        nonlocal col
        if u_coeffs is None:
            u_coeffs = [1] * len(u_indices)
        if v_coeffs is None:
            v_coeffs = [1] * len(v_indices)
        if w_coeffs is None:
            w_coeffs = [1] * len(w_indices)
        for idx, c in zip(u_indices, u_coeffs):
            U[idx][col] += c
        for idx, c in zip(v_indices, v_coeffs):
            V[idx][col] += c
        for idx, c in zip(w_indices, w_coeffs):
            W[idx][col] += c
        col += 1

    def add_strassen_block(k_start):
        a00 = 0
        a01 = 1
        a10 = 3
        a11 = 4
        
        b00 = 0 * 7 + k_start
        b01 = 0 * 7 + k_start + 1
        b10 = 1 * 7 + k_start
        b11 = 1 * 7 + k_start + 1
        
        c00 = k_start * 2 + 0
        c01 = (k_start + 1) * 2 + 0
        c10 = k_start * 2 + 1
        c11 = (k_start + 1) * 2 + 1
        
        add_term([a00, a11], [b00, b11], [c00, c11])
        add_term([a10, a11], [b00], [c10, c11], w_coeffs=[1, -1])
        add_term([a00], [b01, b11], [c01, c11], v_coeffs=[1, -1])
        add_term([a11], [b10, b00], [c00, c10], v_coeffs=[1, -1])
        add_term([a00, a01], [b11], [c00, c01], w_coeffs=[-1, 1])
        add_term([a10, a00], [b00, b01], [c11], u_coeffs=[1, -1])
        add_term([a01, a11], [b10, b11], [c00], u_coeffs=[1, -1])

    add_strassen_block(0)
    add_strassen_block(2)
    add_strassen_block(4)
    
    b06 = 6
    b16 = 7 + 6
    c06 = 6 * 2 + 0
    c16 = 6 * 2 + 1
    
    add_term([0], [b06], [c06])
    add_term([1], [b16], [c06])
    add_term([3], [b06], [c16])
    add_term([4], [b16], [c16])
    
    for k in range(7):
        b2k = 2 * 7 + k
        c0k = k * 2 + 0
        c1k = k * 2 + 1
        
        add_term([2], [b2k], [c0k])
        add_term([5], [b2k], [c1k])
    
    return {"U": U, "V": V, "W": W}

def _try_remove_column(U, V, W, col, n, m, p):
    """Try removing a specific column."""
    U_test = [row[:col] + row[col+1:] for row in U]
    V_test = [row[:col] + row[col+1:] for row in V]
    W_test = [row[:col] + row[col+1:] for row in W]
    
    if _verify_float(U_test, V_test, W_test, n, m, p):
        if _verify_exact(U_test, V_test, W_test, n, m, p):
            return U_test, V_test, W_test
    return None

def _try_scale_and_remove(U, V, W, col_keep, col_remove, scale, n, m, p):
    """Scale col_keep then try removing col_remove."""
    U_test = [row[:] for row in U]
    V_test = [row[:] for row in V]
    W_test = [row[:] for row in W]
    
    for i in range(len(U_test)):
        U_test[i][col_keep] *= scale
    for i in range(len(V_test)):
        V_test[i][col_keep] *= scale
    for i in range(len(W_test)):
        W_test[i][col_keep] *= scale
    
    # Now try removing col_remove
    if col_remove < col_keep:
        U_rem = [row[:col_remove] + row[col_remove+1:] for row in U_test]
        V_rem = [row[:col_remove] + row[col_remove+1:] for row in V_test]
        W_rem = [row[:col_remove] + row[col_remove+1:] for row in W_test]
    else:
        U_rem = [row[:col_remove] + row[col_remove+1:] for row in U_test]
        V_rem = [row[:col_remove] + row[col_remove+1:] for row in V_test]
        W_rem = [row[:col_remove] + row[col_remove+1:] for row in W_test]
    
    if _verify_float(U_rem, V_rem, W_rem, n, m, p):
        if _verify_exact(U_rem, V_rem, W_rem, n, m, p):
            return U_rem, V_rem, W_rem
    return None

def _try_linear_combo_remove(U, V, W, col_remove, n, m, p, deadline):
    """Try to express col_remove as linear combination of others."""
    current_rank = len(U[0])
    
    # Build the tensor contribution of col_remove
    U_rm = np.array([row[col_remove] for row in U], dtype=np.float64)
    V_rm = np.array([row[col_remove] for row in V], dtype=np.float64)
    W_rm = np.array([row[col_remove] for row in W], dtype=np.float64)
    
    # Try to find coefficients c_j such that sum_j c_j * (U_j * V_j * W_j) = U_rm * V_rm * W_rm
    # This is a nonlinear problem, so use heuristic search
    
    # Simple approach: try each other column as a candidate to absorb
    for col_keep in range(current_rank):
        if col_keep == col_remove:
            continue
        
        U_kp = np.array([row[col_keep] for row in U], dtype=np.float64)
        V_kp = np.array([row[col_keep] for row in V], dtype=np.float64)
        W_kp = np.array([row[col_keep] for row in W], dtype=np.float64)
        
        # Check if col_remove is proportional to col_keep in a useful way
        # Try various scales
        for scale in [-2, -1, -0.5, 0.5, 1, 2]:
            result = _try_scale_and_remove(U, V, W, col_keep, col_remove, scale, n, m, p)
            if result is not None:
                return result
    
    return None

def _random_perturb_search(U, V, W, n, m, p, deadline, max_iters=500):
    """Random perturbation search for lower rank."""
    best_rank = len(U[0])
    best_result = (U[:], V[:], W[:])
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        current_rank = len(U[0])
        
        # Try removing a random column
        col = np.random.randint(current_rank)
        result = _try_remove_column(U, V, W, col, n, m, p)
        if result is not None:
            U, V, W = result
            best_rank = len(U[0])
            best_result = (U[:], V[:], W[:])
            continue
        
        # Try scaling a random column then removing another
        col_keep = np.random.randint(current_rank)
        col_remove = np.random.randint(current_rank)
        if col_keep == col_remove:
            continue
        
        scale = np.random.choice([-2, -1, -0.5, 0.5, 1, 2])
        result = _try_scale_and_remove(U, V, W, col_keep, col_remove, scale, n, m, p)
        if result is not None:
            U, V, W = result
            best_rank = len(U[0])
            best_result = (U[:], V[:], W[:])
    
    return best_result

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    n, m, p = DIMENSIONS
    
    result = _build_strassen_rank39(n, m, p)
    
    if not _verify_float(result["U"], result["V"], result["W"], n, m, p):
        # Fallback: schoolbook
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
        return {"U": U, "V": V, "W": W}
    
    U, V, W = result["U"], result["V"], result["W"]
    
    # Phase 1: Systematic column removal
    progress = True
    while progress and time.time() < deadline - 30:
        progress = False
        current_rank = len(U[0])
        for col in range(current_rank):
            result = _try_remove_column(U, V, W, col, n, m, p)
            if result is not None:
                U, V, W = result
                progress = True
                break
    
    # Phase 2: Scale-and-remove search
    if time.time() < deadline - 20:
        for _ in range(100):
            if time.time() > deadline - 20:
                break
            current_rank = len(U[0])
            found = False
            for col_remove in range(current_rank):
                if time.time() > deadline - 20:
                    break
                result = _try_linear_combo_remove(U, V, W, col_remove, n, m, p, deadline)
                if result is not None:
                    U, V, W = result
                    found = True
                    break
            if not found:
                break
    
    # Phase 3: Random perturbation search with remaining time
    if time.time() < deadline - 5:
        U, V, W = _random_perturb_search(U, V, W, n, m, p, deadline - 5)
    
    # Final verification
    if _verify_exact(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    # Fallback to original if something went wrong
    result = _build_strassen_rank39(n, m, p)
    return result
# EVOLVE-BLOCK-END

if __name__ == "__main__":
    res = construct()
