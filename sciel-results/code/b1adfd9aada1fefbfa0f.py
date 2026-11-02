"""Combined approach: structured search for rank-minimal 2x3x3 matmul decomposition."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START

def _schoolbook_decomp():
    """Return the rank-18 schoolbook decomposition as (U, V, W) lists."""
    n, m, p = DIMENSIONS
    rank = n * m * p  # 18
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


def _verify_exact(U, V, W):
    """Verify exact decomposition using Fraction arithmetic."""
    n, m, p = DIMENSIONS
    R = len(U[0])
    for i in range(n):
        for j in range(m):
            for l in range(p):
                s = Fraction(0, 1)
                for c in range(R):
                    s += Fraction(U[i * m + j][c]) * Fraction(V[j * p + l][c]) * Fraction(W[l * n + i][c])
                if s != 1:
                    return False
    return True


def _verify_float(U, V, W, tol=1e-9):
    """Fast float verification."""
    n, m, p = DIMENSIONS
    R = len(U[0])
    for i in range(n):
        for j in range(m):
            for l in range(p):
                s = 0.0
                for c in range(R):
                    s += U[i * m + j][c] * V[j * p + l][c] * W[l * n + i][c]
                if abs(s - 1.0) > tol:
                    return False
    return True


def _snap_to_halfint(x):
    """Snap a float to the nearest half-integer."""
    r = round(x * 2) / 2.0
    if abs(x - r) < 1e-6:
        return r
    # Try simple fractions
    for den in [1, 2, 3, 4, 5, 6]:
        f = round(x * den) / den
        if abs(x - f) < 1e-6:
            return f
    return x


def _snap_matrix(M, tol=1e-6):
    """Snap all entries of a list-of-lists to nearest half-integers."""
    return [[_snap_to_halfint(x) for x in row] for row in M]


def _try_merge_rank18_to_rank17(deadline):
    """
    Attempt to reduce the rank-18 schoolbook to rank 17 by finding a linear
    dependency among the 18 rank-1 terms.
    
    Strategy: The 18 terms are T_{i,j,k} = e_i ⊗ e_j ⊗ e_k (in flattened form).
    We look for coefficients alpha_c such that sum_c alpha_c * T_c = 0.
    If such a non-trivial relation exists, we can eliminate one term.
    
    For 2x3x3, the tensor space has dimension 18 (all entries are 1), so
    the 18 basis tensors are linearly independent in the full tensor space.
    However, the matmul tensor has a specific structure that may allow
    a different decomposition.
    
    Alternative: Use the fact that we only need to represent the specific
    tensor T[i,j,l]=1, not the full basis. We can search for a rank-17
    decomposition directly using coordinate descent initialized near schoolbook.
    """
    n, m, p = DIMENSIONS
    rng = random.Random(42)
    
    # Start with rank-18 schoolbook
    U18, V18, W18 = _schoolbook_decomp()
    
    # Try to find a rank-17 decomposition by perturbing and optimizing
    # Use coordinate descent with rank 17
    
    R = 17
    best_err = float('inf')
    best_result = None
    
    max_outer = 50
    max_inner = 30
    
    for outer in range(max_outer):
        if time.time() > deadline - 5:
            break
        
        # Initialize near schoolbook: take first 17 terms and perturb
        U = np.array(U18[:n*m][:], dtype=float)  # wrong shape, need to reshape
        # Actually U18 is (6, 18), we want (6, 17)
        # Take first 17 columns and add small noise
        U_init = np.array(U18, dtype=float)[:, :R] + rng.gauss(0, 0.1, (n*m, R))
        V_init = np.array(V18, dtype=float)[:, :R] + rng.gauss(0, 0.1, (m*p, R))
        W_init = np.array(W18, dtype=float)[:, :R] + rng.gauss(0, 0.1, (p*n, R))
        
        U_c = U_init.copy()
        V_c = V_init.copy()
        W_c = W_init.copy()
        
        # Coordinate descent
        for inner in range(max_inner):
            if time.time() > deadline - 5:
                break
            
            # Update U: for each (i,j), solve least squares
            for i in range(n):
                for j in range(m):
                    idx = i * m + j
                    # T[i,j,l] = sum_c U[idx,c] * V[j*p+l, c] * W[l*n+i, c] = 1 for all l
                    M = np.zeros((p, R))
                    b = np.ones(p)
                    for l in range(p):
                        for c in range(R):
                            M[l, c] = V_c[j * p + l, c] * W_c[l * n + i, c]
                    try:
                        sol, _, _, _ = np.linalg.lstsq(M, b, rcond=None)
                        U_c[idx, :] = sol
                    except:
                        pass
            
            # Update V: for each (j,l), solve least squares
            for j in range(m):
                for l in range(p):
                    idx = j * p + l
                    M = np.zeros((n, R))
                    b = np.ones(n)
                    for i in range(n):
                        for c in range(R):
                            M[i, c] = U_c[i * m + j, c] * W_c[l * n + i, c]
                    try:
                        sol, _, _, _ = np.linalg.lstsq(M, b, rcond=None)
                        V_c[idx, :] = sol
                    except:
                        pass
            
            # Update W: for each (l,i), solve least squares
            for l in range(p):
                for i in range(n):
                    idx = l * n + i
                    M = np.zeros((m, R))
                    b = np.ones(m)
                    for j in range(m):
                        for c in range(R):
                            M[j, c] = U_c[i * m + j, c] * V_c[j * p + l, c]
                    try:
                        sol, _, _, _ = np.linalg.lstsq(M, b, rcond=None)
                        W_c[idx, :] = sol
                    except:
                        pass
            
            # Compute error
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for l in range(p):
                        s = 0.0
                        for c in range(R):
                            s += U_c[i * m + j, c] * V_c[j * p + l, c] * W_c[l * n + i, c]
                        err += (s - 1.0) ** 2
            
            if err < best_err:
                best_err = err
                best_result = (U_c.copy(), V_c.copy(), W_c.copy())
            
            if best_err < 1e-10:
                break
        
        if best_err < 1e-10:
            break
    
    if best_result is None:
        return None
    
    U_arr, V_arr, W_arr = best_result
    
    # Snap to half-integers
    U_list = _snap_matrix(U_arr.tolist())
    V_list = _snap_matrix(V_arr.tolist())
    W_list = _snap_matrix(W_arr.tolist())
    
    # Verify exactly
    if _verify_exact(U_list, V_list, W_list):
        return U_list, V_list, W_list
    
    # Try with tighter snap
    for tol in [1e-4, 1e-3, 1e-2]:
        U_list2 = _snap_matrix(U_arr.tolist(), tol=tol)
        V_list2 = _snap_matrix(V_arr.tolist(), tol=tol)
        W_list2 = _snap_matrix(W_arr.tolist(), tol=tol)
        if _verify_exact(U_list2, V_list2, W_list2):
            return U_list2, V_list2, W_list2
    
    return None


def _try_general_rank(R, deadline, seed=0):
    """General rank-R search using coordinate descent from random init."""
    n, m, p = DIMENSIONS
    rng = random.Random(seed)
    
    max_outer = 30
    max_inner = 25
    
    best_err = float('inf')
    best_result = None
    
    for outer in range(max_outer):
        if time.time() > deadline - 5:
            break
        
        U_c = rng.gauss(0, 1, (n * m, R))
        V_c = rng.gauss(0, 1, (m * p, R))
        W_c = rng.gauss(0, 1, (p * n, R))
        
        for inner in range(max_inner):
            if time.time() > deadline - 5:
                break
            
            for i in range(n):
                for j in range(m):
                    idx = i * m + j
                    M = np.zeros((p, R))
                    b = np.ones(p)
                    for l in range(p):
                        for c in range(R):
                            M[l, c] = V_c[j * p + l, c] * W_c[l * n + i, c]
                    try:
                        sol, _, _, _ = np.linalg.lstsq(M, b, rcond=None)
                        U_c[idx, :] = sol
                    except:
                        pass
            
            for j in range(m):
                for l in range(p):
                    idx = j * p + l
                    M = np.zeros((n, R))
                    b = np.ones(n)
                    for i in range(n):
                        for c in range(R):
                            M[i, c] = U_c[i * m + j, c] * W_c[l * n + i, c]
                    try:
                        sol, _, _, _ = np.linalg.lstsq(M, b, rcond=None)
                        V_c[idx, :] = sol
                    except:
                        pass
            
            for l in range(p):
                for i in range(n):
                    idx = l * n + i
                    M = np.zeros((m, R))
                    b = np.ones(m)
                    for j in range(m):
                        for c in range(R):
                            M[j, c] = U_c[i * m + j, c] * V_c[j * p + l, c]
                    try:
                        sol, _, _, _ = np.linalg.lstsq(M, b, rcond=None)
                        W_c[idx, :] = sol
                    except:
                        pass
            
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for l in range(p):
                        s = 0.0
                        for c in range(R):
                            s += U_c[i * m + j, c] * V_c[j * p + l, c] * W_c[l * n + i, c]
                        err += (s - 1.0) ** 2
            
            if err < best_err:
                best_err = err
                best_result = (U_c.copy(), V_c.copy(), W_c.copy())
            
            if best_err < 1e-10:
                break
        
        if best_err < 1e-10:
            break
    
    if best_result is None:
        return None
    
    U_arr, V_arr, W_arr = best_result
    U_list = _snap_matrix(U_arr.tolist())
    V_list = _snap_matrix(V_arr.tolist())
    W_list = _snap_matrix(W_arr.tolist())
    
    if _verify_exact(U_list, V_list, W_list):
        return U_list, V_list, W_list
    
    return None


def construct():
    """Construct a rank-minimal exact tensor decomposition for 2x3x3 matmul."""
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    n, m, p = DIMENSIONS
    
    # Try rank 17 first (potential improvement over schoolbook 18)
    result = _try_merge_rank18_to_rank17(deadline)
    if result is not None:
        U, V, W = result
        return {"U": U, "V": V, "W": W}
    
    # Try a few more seeds for rank 17
    for seed in [1, 2, 3, 4, 5]:
        if time.time() > deadline - 10:
            break
        result = _try_general_rank(17, deadline, seed=seed)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook rank 18
    U, V, W = _schoolbook_decomp()
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
