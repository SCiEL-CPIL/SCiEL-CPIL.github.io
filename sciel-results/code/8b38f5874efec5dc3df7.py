"""Matrix multiplication 4,4,8 rank-minimal search with Strassen + linear system refinement."""
import random
import time
import numpy as np
from itertools import product

DIMENSIONS = [4, 4, 8]

def _strassen_2x2_factors():
    """
    Returns U, V, W for 2x2 matrix multiplication using Strassen's algorithm.
    U: 4x7, V: 4x7, W: 4x7
    Indices: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A11+A22)(B11+B22) -> contributes to C11, C22
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (A21+A22)B11 -> contributes to C21
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1
    
    # M3 = A11(B12-B22) -> contributes to C12
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1
    
    # M4 = A22(B21-B11) -> contributes to C21
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[2, 3] = 1
    
    # M5 = (A11+A12)B22 -> contributes to C12
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[1, 4] = 1
    
    # M6 = (A21-A11)(B11+B12) -> contributes to C22
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7 = (A12-A22)(B21+B22) -> contributes to C11
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _strassen_4x4_factors():
    """
    Returns U, V, W for 4x4 matrix multiplication using recursive Strassen.
    U: 16x49, V: 16x49, W: 16x49
    """
    U2, V2, W2 = _strassen_2x2_factors()
    
    U = np.zeros((16, 49))
    V = np.zeros((16, 49))
    W = np.zeros((16, 49))
    
    high_level_factors = [
        {'U': {(0,0): 1, (1,1): 1}, 'V': {(0,0): 1, (1,1): 1}, 'W': {(0,0): 1, (1,1): 1}},
        {'U': {(1,0): 1, (1,1): 1}, 'V': {(0,0): 1}, 'W': {(1,0): 1}},
        {'U': {(0,0): 1}, 'V': {(0,1): 1, (1,1): -1}, 'W': {(0,1): 1}},
        {'U': {(1,1): 1}, 'V': {(1,0): 1, (0,0): -1}, 'W': {(0,0): 1, (1,0): 1}},
        {'U': {(0,0): 1, (0,1): 1}, 'V': {(1,1): 1}, 'W': {(0,0): -1, (0,1): 1}},
        {'U': {(1,0): 1, (0,0): -1}, 'V': {(0,0): 1, (0,1): 1}, 'W': {(1,1): 1}},
        {'U': {(0,1): 1, (1,1): -1}, 'V': {(1,0): 1, (1,1): 1}, 'W': {(0,0): 1}},
    ]
    
    for h, hf in enumerate(high_level_factors):
        for l in range(7):
            factor_idx = h * 7 + l
            
            for (bi, bj), coeff_u in hf['U'].items():
                for li in range(2):
                    for lj in range(2):
                        a_row = bi * 2 + li
                        a_col = bj * 2 + lj
                        U_idx = a_row * 4 + a_col
                        U[U_idx, factor_idx] += coeff_u * U2[li * 2 + lj, l]
            
            for (bi, bj), coeff_v in hf['V'].items():
                for li in range(2):
                    for lj in range(2):
                        b_row = bi * 2 + li
                        b_col = bj * 2 + lj
                        V_idx = b_row * 4 + b_col
                        V[V_idx, factor_idx] += coeff_v * V2[li * 2 + lj, l]
            
            for (bi, bj), coeff_w in hf['W'].items():
                for li in range(2):
                    for lj in range(2):
                        c_row = bi * 2 + li
                        c_col = bj * 2 + lj
                        W_idx = c_row * 4 + c_col
                        W[W_idx, factor_idx] += coeff_w * W2[li * 2 + lj, l]
    
    return U, V, W

def _adapt_to_4x8(U4, V4, W4):
    """
    Adapt 4x4 Strassen factors to 4x8 multiplication.
    """
    rank4 = U4.shape[1]
    
    U = U4.copy()
    
    V = np.zeros((32, rank4 * 2))
    V[:16, :rank4] = V4
    V[16:, rank4:] = V4
    
    W = np.zeros((32, rank4 * 2))
    W[:16, :rank4] = W4
    W[16:, rank4:] = W4
    
    return U.tolist(), V.tolist(), W.tolist()

def _validate_fast(U, V, W, n, m, p):
    """Fast vectorized validation using numpy."""
    rank = len(U[0]) if U else 0
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    i_idx = []
    j_idx = []
    k_idx = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                i_idx.append(i)
                j_idx.append(j)
                k_idx.append(k)
    
    i_arr = np.array(i_idx)
    j_arr = np.array(j_idx)
    k_arr = np.array(k_idx)
    
    a_arr = i_arr * m + j_arr
    b_arr = j_arr * p + k_arr
    c_arr = k_arr * n + i_arr
    
    U_sel = U_arr[a_arr, :]
    V_sel = V_arr[b_arr, :]
    W_sel = W_arr[c_arr, :]
    
    products = U_sel * V_sel * W_sel
    sums = products.sum(axis=1)
    
    if np.max(np.abs(sums - 1.0)) > 1e-9:
        return False
    
    c_alt_arr = ((k_arr + 1) % p) * n + i_arr
    mask = c_alt_arr != c_arr
    if np.any(mask):
        W_alt_sel = W_arr[c_alt_arr[mask], :]
        products_alt = U_sel[mask] * V_sel[mask] * W_alt_sel
        sums_alt = products_alt.sum(axis=1)
        if np.max(np.abs(sums_alt)) > 1e-9:
            return False
    
    return True

def _solve_for_W_given_UV(U, V, n, m, p, deadline):
    """
    Given U (nm x r) and V (mp x r), solve the linear system for W (pn x r).
    The equations are: for each (i,j,k), sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    And for wrong k': sum_r U[i*m+j, r] * V[j*p+k', r] * W[k'*n+i, r] = 0
    
    This is linear in W. We build the system and solve it.
    """
    start = time.time()
    nm = n * m
    mp = m * p
    pn = p * n
    rank = U.shape[1]
    
    # Build the linear system: A * w = b where w is the flattened W (pn*r unknowns)
    # For each (i,j,k), we have one equation involving W[k*n+i, :]
    # The equation is: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = target
    
    # Number of equations: n * m * p (for the 1's) + n * m * p * (p-1) (for the 0's)
    # But many are redundant. Let's use only the n*m*p equations for the 1's first,
    # then verify the zeros.
    
    # Actually, let's build a system where each equation corresponds to one (i,j,k) triple.
    # For a given (i,j,k), the equation involves W[k*n+i, :] with coefficients U[i*m+j, r]*V[j*p+k, r].
    # So each equation only involves r unknowns (one row of W).
    
    # This means the system is block-diagonal! Each block corresponds to a fixed (k,i) pair,
    # and the equations for that block are over different j values.
    
    # For fixed (k,i): equations are for j=0..m-1
    # A_block[j, r] = U[i*m+j, r] * V[j*p+k, r]
    # b_block[j] = 1
    
    # So we have p*n blocks, each of size m x r.
    # If m <= r, we can solve each block independently.
    
    W = np.zeros((pn, rank))
    
    for k in range(p):
        for i in range(n):
            if time.time() > deadline - 5:
                return None
            
            # Build the m x r system
            A_block = np.zeros((m, rank))
            b_block = np.ones(m)
            
            for j in range(m):
                a_idx = i * m + j
                b_idx = j * p + k
                A_block[j, :] = U[a_idx, :] * V[b_idx, :]
            
            # Solve A_block * w_row = b_block
            # w_row is W[k*n+i, :]
            try:
                # Use least squares if overdetermined, or direct solve
                if m == rank:
                    w_row = np.linalg.solve(A_block, b_block)
                elif m < rank:
                    # Underdetermined: use minimum norm solution
                    w_row = np.linalg.lstsq(A_block, b_block, rcond=None)[0]
                else:
                    # Overdetermined: least squares
                    w_row = np.linalg.lstsq(A_block, b_block, rcond=None)[0]
            except np.linalg.LinAlgError:
                return None
            
            W[k * n + i, :] = w_row
    
    # Check if W entries are close to integers or half-integers
    W_rounded = np.round(W * 2) / 2
    if np.max(np.abs(W - W_rounded)) > 1e-6:
        return None
    
    # Verify the full decomposition
    W_list = W_rounded.tolist()
    U_list = U.tolist()
    V_list = V.tolist()
    if not _validate_fast(U_list, V_list, W_list, n, m, p):
        return None
    
    return W_rounded

def _try_random_rank_with_linear_solve(r, n, m, p, deadline, max_attempts=100):
    """
    Try to find a rank-r decomposition by randomly initializing U and V,
    then solving for W via the linear system.
    """
    nm = n * m
    mp = m * p
    
    for attempt in range(max_attempts):
        if time.time() > deadline - 10:
            return None
        
        # Random initialization with entries from {0, 1, -1}
        U = np.array([[random.choice([0, 1, -1]) for _ in range(r)] for _ in range(nm)], dtype=np.float64)
        V = np.array([[random.choice([0, 1, -1]) for _ in range(r)] for _ in range(mp)], dtype=np.float64)
        
        W = _solve_for_W_given_UV(U, V, n, m, p, deadline)
        if W is not None:
            return U.tolist(), V.tolist(), W.tolist()
        
        # Try a few local modifications
        for _ in range(3):
            if time.time() > deadline - 10:
                return None
            mat = random.choice([U, V])
            row = random.randrange(len(mat))
            col = random.randrange(r)
            mat[row, col] = -mat[row, col] if mat[row, col] != 0 else 1
            
            W = _solve_for_W_given_UV(U, V, n, m, p, deadline)
            if W is not None:
                return U.tolist(), V.tolist(), W.tolist()
    
    return None

def _construct_strassen_4x8():
    """Construct Strassen-based decomposition for 4x4 * 4x8."""
    U4, V4, W4 = _strassen_4x4_factors()
    U, V, W = _adapt_to_4x8(U4, V4, W4)
    return {"U": U, "V": V, "W": W}

def _try_reduce_rank_from_strassen(n, m, p, deadline):
    """
    Start from the Strassen rank-98 solution and try to reduce rank
    by identifying and removing redundant columns.
    """
    U_str, V_str, W_str = _construct_strassen_4x8()
    U_arr = np.array(U_str, dtype=np.float64)
    V_arr = np.array(V_str, dtype=np.float64)
    W_arr = np.array(W_str, dtype=np.float64)
    
    rank = U_arr.shape[1]
    keep = list(range(rank))
    
    changed = True
    while changed and time.time() < deadline - 20:
        changed = False
        for r_idx in range(len(keep) - 1, -1, -1):
            if time.time() > deadline - 20:
                break
            test_keep = keep[:r_idx] + keep[r_idx+1:]
            if len(test_keep) == 0:
                continue
            
            U_test = U_arr[:, test_keep]
            V_test = V_arr[:, test_keep]
            W_test = W_arr[:, test_keep]
            
            U_list = U_test.tolist()
            V_list = V_test.tolist()
            W_list = W_test.tolist()
            
            if _validate_fast(U_list, V_list, W_list, n, m, p):
                keep = test_keep
                changed = True
                break
    
    if len(keep) < rank:
        U_final = U_arr[:, keep]
        V_final = V_arr[:, keep]
        W_final = W_arr[:, keep]
        return U_final.tolist(), V_final.tolist(), W_final.tolist()
    
    return None

def _try_fixed_U_solve_VW(U, n, m, p, deadline):
    """
    Given U, try to find V and W by random V initialization and solving for W,
    or by a more systematic approach.
    """
    mp = m * p
    pn = p * n
    rank = U.shape[1]
    
    for attempt in range(50):
        if time.time() > deadline - 10:
            return None
        
        V = np.array([[random.choice([0, 1, -1]) for _ in range(rank)] for _ in range(mp)], dtype=np.float64)
        
        W = _solve_for_W_given_UV(U, V, n, m, p, deadline)
        if W is not None:
            return U.tolist(), V.tolist(), W.tolist()
        
        # Local modification
        row = random.randrange(len(V))
        col = random.randrange(rank)
        V[row, col] = -V[row, col] if V[row, col] != 0 else 1
        
        W = _solve_for_W_given_UV(U, V, n, m, p, deadline)
        if W is not None:
            return U.tolist(), V.tolist(), W.tolist()
    
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 135  # safety margin
    
    best_result = None
    best_rank = float('inf')
    
    # Step 1: Strassen-based decomposition (rank 98 for 4x8)
    try:
        result = _construct_strassen_4x8()
        U, V, W = result['U'], result['V'], result['W']
        if _validate_fast(U, V, W, n, m, p):
            best_result = result
            best_rank = len(U[0])
    except Exception:
        pass
    
    # Step 2: Try to reduce rank from Strassen
    try:
        reduced = _try_reduce_rank_from_strassen(n, m, p, deadline)
        if reduced is not None:
            U_red, V_red, W_red = reduced
            if _validate_fast(U_red, V_red, W_red, n, m, p):
                red_rank = len(U_red[0])
                if red_rank < best_rank:
                    best_rank = red_rank
                    best_result = {"U": U_red, "V": V_red, "W": W_red}
    except Exception:
        pass
    
    # Step 3: Try linear system approach at various ranks
    # Start from a reasonable rank and go down
    if best_result is not None:
        search_start = max(1, best_rank - 30)
    else:
        search_start = 20
    
    # Try ranks from low to high, but focus on the range near our best
    for r in range(search_start, min(n * m * p + 1, 130)):
        if time.time() > deadline - 25:
            break
        if r >= best_rank:
            break
        
        result = _try_random_rank_with_linear_solve(r, n, m, p, deadline, max_attempts=20)
        if result is not None:
            U, V, W = result
            if _validate_fast(U, V, W, n, m, p):
                best_rank = r
                best_result = {"U": U, "V": V, "W": W}
                # Continue searching for even lower ranks
                search_start = r
    
    # Step 4: Try a few more targeted searches near the best rank found
    if best_result is not None and time.time() < deadline - 30:
        for r in range(max(1, best_rank - 10), best_rank):
            if time.time() > deadline - 15:
                break
            result = _try_random_rank_with_linear_solve(r, n, m, p, deadline, max_attempts=50)
            if result is not None:
                U, V, W = result
                if _validate_fast(U, V, W, n, m, p):
                    best_rank = r
                    best_result = {"U": U, "V": V, "W": W}
    
    if best_result is not None:
        return best_result
    
    # Fallback to schoolbook
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
