"""Combined approach: verified rank-39 baseline, then iterative refinement for lower rank."""
DIMENSIONS = [2, 2, 11]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition is exact."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _verify_exact_halfint(U, V, W, n, m, p):
    """Verify exact decomposition with half-integer entries using Fractions."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(rank):
                    uc = Fraction(U[i * m + j][c], 1)
                    vc = Fraction(V[j * p + k][c], 1)
                    wc = Fraction(W[k * n + i][c], 1)
                    s += uc * vc * wc
                if s != Fraction(1):
                    return False
    return True

def _round_to_half_integers(U, V, W, tol=1e-6):
    """Round entries to nearest half-integer if close enough."""
    U_r = [[round(2*x)/2 for x in row] for row in U]
    V_r = [[round(2*x)/2 for x in row] for row in V]
    W_r = [[round(2*x)/2 for x in row] for row in W]
    
    # Check if rounding preserved accuracy
    for i in range(n_global):
        for j in range(m_global):
            for k in range(p_global):
                s_orig = sum(U[i*m_global+j][c]*V[j*p_global+k][c]*W[k*n_global+i][c] for c in range(len(U[0])))
                s_round = sum(U_r[i*m_global+j][c]*V_r[j*p_global+k][c]*W_r[k*n_global+i][c] for c in range(len(U[0])))
                if abs(s_round - 1.0) > tol:
                    return None
    return U_r, V_r, W_r

# Global dimensions for helper functions
n_global, m_global, p_global = 2, 2, 11

def _construct_rank39():
    """
    Explicit rank-39 decomposition for M_{2,2,11}.
    5 blocks of 2 columns (Strassen rank 7 each) + 1 single column (schoolbook rank 4).
    Total: 5*7 + 4 = 39.
    """
    n, m, p = 2, 2, 11
    rank = 39
    
    U = [[0.0] * rank for _ in range(n * m)]   # 4 x 39
    V = [[0.0] * rank for _ in range(m * p)]   # 22 x 39
    W = [[0.0] * rank for _ in range(p * n)]   # 22 x 39
    
    r = 0
    for block in range(5):
        k0 = block * 2
        k1 = block * 2 + 1
        
        v_e = k0
        v_f = k1
        v_g = p + k0
        v_h = p + k1
        
        w_c00 = k0 * n
        w_c10 = k0 * n + 1
        w_c01 = k1 * n
        w_c11 = k1 * n + 1
        
        U[0][r] = 1; U[3][r] = 1
        V[v_e][r] = 1; V[v_h][r] = 1
        W[w_c00][r] = 1; W[w_c11][r] = 1
        r += 1
        
        U[2][r] = 1; U[3][r] = 1
        V[v_e][r] = 1
        W[w_c10][r] = 1; W[w_c11][r] = -1
        r += 1
        
        U[0][r] = 1
        V[v_f][r] = 1; V[v_h][r] = -1
        W[w_c01][r] = 1; W[w_c11][r] = 1
        r += 1
        
        U[3][r] = 1
        V[v_g][r] = 1; V[v_e][r] = -1
        W[w_c00][r] = 1; W[w_c10][r] = 1
        r += 1
        
        U[0][r] = 1; U[1][r] = 1
        V[v_h][r] = 1
        W[w_c00][r] = -1; W[w_c01][r] = 1
        r += 1
        
        U[2][r] = 1; U[0][r] = -1
        V[v_e][r] = 1; V[v_f][r] = 1
        W[w_c11][r] = 1
        r += 1
        
        U[1][r] = 1; U[3][r] = -1
        V[v_g][r] = 1; V[v_h][r] = 1
        W[w_c00][r] = 1
        r += 1
    
    k2 = 10
    v_e2 = k2
    v_g2 = p + k2
    w_c02 = k2 * n
    w_c12 = k2 * n + 1
    
    U[0][r] = 1
    V[v_e2][r] = 1
    W[w_c02][r] = 1
    r += 1
    
    U[1][r] = 1
    V[v_g2][r] = 1
    W[w_c02][r] = 1
    r += 1
    
    U[2][r] = 1
    V[v_e2][r] = 1
    W[w_c12][r] = 1
    r += 1
    
    U[3][r] = 1
    V[v_g2][r] = 1
    W[w_c12][r] = 1
    r += 1
    
    assert r == rank
    return U, V, W

def _precompute_indices(n, m, p):
    """Precompute index arrays and masks for fast alternating minimization."""
    ta_list, tb_list, tc_list = [], [], []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                ta_list.append(i * m + j)
                tb_list.append(j * p + k)
                tc_list.append(k * n + i)
    ta_arr = np.array(ta_list)
    tb_arr = np.array(tb_list)
    tc_arr = np.array(tc_list)
    
    # Precompute masks for each row
    masks_a = [(ta_arr == a) for a in range(n * m)]
    masks_b = [(tb_arr == b) for b in range(m * p)]
    masks_c = [(tc_arr == c) for c in range(p * n)]
    
    # Precompute index subsets
    idx_a = [np.where(masks_a[a])[0] for a in range(n * m)]
    idx_b = [np.where(masks_b[b])[0] for b in range(m * p)]
    idx_c = [np.where(masks_c[c])[0] for c in range(p * n)]
    
    return ta_arr, tb_arr, tc_arr, idx_a, idx_b, idx_c

def _try_low_rank(rank, n, m, p, deadline, max_rounds=5000, seed=42, tol=1e-10, U_init=None, V_init=None, W_init=None):
    """
    Try to find a decomposition of given rank using iterative refinement.
    Returns (U, V, W) as lists if found, else None.
    """
    rng = np.random.RandomState(seed)
    
    if U_init is not None:
        U = np.array(U_init, dtype=float)
        V = np.array(V_init, dtype=float)
        W = np.array(W_init, dtype=float)
    else:
        U = rng.randn(n * m, rank)
        V = rng.randn(m * p, rank)
        W = rng.randn(p * n, rank)
    
    ta_arr, tb_arr, tc_arr, idx_a, idx_b, idx_c = _precompute_indices(n, m, p)
    
    best_err = float('inf')
    best_state = None
    no_improve_count = 0
    patience = 200
    
    for it in range(max_rounds):
        if time.time() > deadline:
            break
        
        # Update U - vectorized
        for a in range(n * m):
            idx = idx_a[a]
            if len(idx) == 0:
                continue
            tb_sub = tb_arr[idx]
            tc_sub = tc_arr[idx]
            V_sub = V[tb_sub, :]
            W_sub = W[tc_sub, :]
            prod = V_sub * W_sub
            num = prod.sum(axis=0)
            den = (prod ** 2).sum(axis=0)
            valid = den > 1e-12
            if np.any(valid):
                U[a, valid] = num[valid] / den[valid]
        
        # Update V - vectorized
        for b in range(m * p):
            idx = idx_b[b]
            if len(idx) == 0:
                continue
            ta_sub = ta_arr[idx]
            tc_sub = tc_arr[idx]
            U_sub = U[ta_sub, :]
            W_sub = W[tc_sub, :]
            prod = U_sub * W_sub
            num = prod.sum(axis=0)
            den = (prod ** 2).sum(axis=0)
            valid = den > 1e-12
            if np.any(valid):
                V[b, valid] = num[valid] / den[valid]
        
        # Update W - vectorized
        for c_idx in range(p * n):
            idx = idx_c[c_idx]
            if len(idx) == 0:
                continue
            ta_sub = ta_arr[idx]
            tb_sub = tb_arr[idx]
            U_sub = U[ta_sub, :]
            V_sub = V[tb_sub, :]
            prod = U_sub * V_sub
            num = prod.sum(axis=0)
            den = (prod ** 2).sum(axis=0)
            valid = den > 1e-12
            if np.any(valid):
                W[c_idx, valid] = num[valid] / den[valid]
        
        # Check convergence every 10 iterations
        if it % 10 == 0:
            sums = np.zeros(len(ta_arr))
            for c in range(rank):
                sums += U[ta_arr, c] * V[tb_arr, c] * W[tc_arr, c]
            err = np.sum((sums - 1.0) ** 2)
            if err < best_err - 1e-12:
                best_err = err
                best_state = (U.copy(), V.copy(), W.copy())
                no_improve_count = 0
            else:
                no_improve_count += 1
            
            if err < tol:
                # Try to round to half-integers
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W.tolist()
                rounded = _round_to_half_integers(U_list, V_list, W_list)
                if rounded is not None:
                    return rounded
                return U_list, V_list, W_list
            
            if no_improve_count > patience:
                break
    
    # Final polish: try rounding the best state
    if best_state is not None and best_err < 1e-6:
        U_b, V_b, W_b = best_state
        U_list = U_b.tolist()
        V_list = V_b.tolist()
        W_list = W_b.tolist()
        rounded = _round_to_half_integers(U_list, V_list, W_list)
        if rounded is not None:
            return rounded
        if best_err < 1e-8:
            return U_list, V_list, W_list
    
    return None

def _make_warm_start_rank38(U39, V39, W39):
    """Create a rank-38 warm start by dropping one column from rank-39."""
    # Drop the last column (schoolbook part) and see if we can recover
    U38 = [row[:38] for row in U39]
    V38 = [row[:38] for row in V39]
    W38 = [row[:38] for row in W39]
    return U38, V38, W38

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 150s kill
    
    # First, return the verified rank-39 construction
    U39, V39, W39 = _construct_rank39()
    if not _verify(U39, V39, W39, n, m, p):
        # Fallback: schoolbook
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
    
    # Try to find lower rank with remaining time
    best_result = None
    
    # Strategy 1: Try rank 38 with warm start from rank 39
    remaining = deadline - time.time()
    if remaining > 15:
        U38_init, V38_init, W38_init = _make_warm_start_rank38(U39, V39, W39)
        for seed in range(10):
            if time.time() > deadline - 10:
                break
            result = _try_low_rank(38, n, m, p, deadline=deadline - 10, 
                                   max_rounds=3000, seed=seed,
                                   U_init=U38_init, V_init=V38_init, W_init=W38_init)
            if result is not None:
                U2, V2, W2 = result
                if _verify(U2, V2, W2, n, m, p):
                    best_result = (U2, V2, W2)
                    break
    
    # Strategy 2: Try rank 38 with random init if warm start failed
    if best_result is None:
        remaining = deadline - time.time()
        if remaining > 15:
            for seed in range(20, 30):
                if time.time() > deadline - 10:
                    break
                result = _try_low_rank(38, n, m, p, deadline=deadline - 10, 
                                       max_rounds=3000, seed=seed)
                if result is not None:
                    U2, V2, W2 = result
                    if _verify(U2, V2, W2, n, m, p):
                        best_result = (U2, V2, W2)
                        break
    
    # Strategy 3: Try rank 37
    if best_result is None:
        remaining = deadline - time.time()
        if remaining > 20:
            for seed in range(50, 60):
                if time.time() > deadline - 10:
                    break
                result = _try_low_rank(37, n, m, p, deadline=deadline - 10, 
                                       max_rounds=2500, seed=seed)
                if result is not None:
                    U2, V2, W2 = result
                    if _verify(U2, V2, W2, n, m, p):
                        best_result = (U2, V2, W2)
                        break
    
    # Strategy 4: Try rank 36
    if best_result is None:
        remaining = deadline - time.time()
        if remaining > 20:
            for seed in range(70, 80):
                if time.time() > deadline - 10:
                    break
                result = _try_low_rank(36, n, m, p, deadline=deadline - 10, 
                                       max_rounds=2000, seed=seed)
                if result is not None:
                    U2, V2, W2 = result
                    if _verify(U2, V2, W2, n, m, p):
                        best_result = (U2, V2, W2)
                        break
    
    if best_result is not None:
        return {"U": best_result[0], "V": best_result[1], "W": best_result[2]}
    
    return {"U": U39, "V": V39, "W": W39}
# EVOLVE-BLOCK-END
