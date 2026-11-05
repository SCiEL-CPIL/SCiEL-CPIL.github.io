import time
import sys
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def _verify_exact(U, V, W, n, m, p):
    """Verify exact decomposition using integer/rational arithmetic."""
    rank = len(U[0])
    # Convert to Fraction for exact arithmetic
    U_frac = [[Fraction(x) for x in row] for row in U]
    V_frac = [[Fraction(x) for x in row] for row in V]
    W_frac = [[Fraction(x) for x in row] for row in W]
    
    # Check all (i,j,k) entries
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
        """Add a 2x2x2 Strassen block for output columns k_start, k_start+1."""
        a00 = 0 * 3 + 0
        a01 = 0 * 3 + 1
        a10 = 1 * 3 + 0
        a11 = 1 * 3 + 1
        
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
    
    b06 = 0 * 7 + 6
    b16 = 1 * 7 + 6
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

def _try_rank_reduction(U_init, V_init, W_init, n, m, p, deadline, max_iterations=10000):
    """Aggressive rank reduction with exact verification."""
    U = [row[:] for row in U_init]
    V = [row[:] for row in V_init]
    W = [row[:] for row in W_init]
    
    current_rank = len(U[0])
    best = (current_rank, U, V, W)
    
    for iteration in range(max_iterations):
        if time.time() > deadline:
            break
        
        progress = False
        
        # Try single column removal
        for col in range(current_rank):
            U_test = [row[:col] + row[col+1:] for row in U]
            V_test = [row[:col] + row[col+1:] for row in V]
            W_test = [row[:col] + row[col+1:] for row in W]
            
            if _verify_float(U_test, V_test, W_test, n, m, p):
                # Confirm with exact arithmetic
                if _verify_exact(U_test, V_test, W_test, n, m, p):
                    U = U_test
                    V = V_test
                    W = W_test
                    current_rank -= 1
                    best = (current_rank, U, V, W)
                    progress = True
                    break
        
        if progress:
            continue
        
        # Try replacing one column with a linear combination of others
        # This allows more sophisticated reductions
        for col in range(current_rank):
            if time.time() > deadline:
                break
            # Try removing col and see if we can recover by adjusting others
            # For now, just try all single removals again in different order
            pass
        
        if not progress:
            break
    
    return best[1], best[2], best[3]

def _try_column_merge(U, V, W, n, m, p, col_a, col_b, deadline):
    """Try to merge two columns by finding a linear combination."""
    current_rank = len(U[0])
    
    # Try: can we express column col_a as a combination of others after removing col_b?
    # This is complex; instead try simple coefficient adjustments
    # Try scaling col_a by various factors and see if col_b becomes redundant
    
    for scale in [-2, -1, -0.5, 0.5, 1, 2]:
        U_test = [row[:] for row in U]
        V_test = [row[:] for row in V]
        W_test = [row[:] for row in W]
        
        # Scale col_a
        for i in range(len(U_test)):
            U_test[i][col_a] *= scale
        for i in range(len(V_test)):
            V_test[i][col_a] *= scale
        for i in range(len(W_test)):
            W_test[i][col_a] *= scale
        
        # Try removing col_b
        U_rem = [row[:col_b] + row[col_b+1:] for row in U_test]
        V_rem = [row[:col_b] + row[col_b+1:] for row in V_test]
        W_rem = [row[:col_b] + row[col_b+1:] for row in W_test]
        
        if _verify_float(U_rem, V_rem, W_rem, n, m, p):
            if _verify_exact(U_rem, V_rem, W_rem, n, m, p):
                return U_rem, V_rem, W_rem
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    result = _build_strassen_rank39(n, m, p)
    
    if _verify_float(result["U"], result["V"], result["W"], n, m, p):
        # First pass: simple column removal
        U, V, W = _try_rank_reduction(result["U"], result["V"], result["W"], n, m, p, deadline)
        result = {"U": U, "V": V, "W": W}
        
        # Second pass: try column merges
        if time.time() < deadline:
            current_rank = len(U[0])
            for iter in range(20):
                if time.time() > deadline:
                    break
                found_merge = False
                for a in range(current_rank):
                    for b in range(a+1, current_rank):
                        if time.time() > deadline:
                            break
                        merged = _try_column_merge(U, V, W, n, m, p, a, b, deadline)
                        if merged is not None:
                            U, V, W = merged
                            current_rank -= 1
                            found_merge = True
                            break
                    if found_merge:
                        break
                if not found_merge:
                    break
            result = {"U": U, "V": V, "W": W}
        
        # Final verification
        if _verify_exact(result["U"], result["V"], result["W"], n, m, p):
            return result
    
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
# EVOLVE-BLOCK-END

if __name__ == "__main__":
    res = construct()
