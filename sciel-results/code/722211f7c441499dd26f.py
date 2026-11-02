"""Matrix multiplication 4,4,8 rank-minimal search with Strassen decomposition."""
import random
import time
import numpy as np

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
    U[0, 0] = 1; U[3, 0] = 1  # A11, A22
    V[0, 0] = 1; V[3, 0] = 1  # B11, B22
    W[0, 0] = 1; W[3, 0] = 1  # C11, C22
    
    # M2 = (A21+A22)B11 -> contributes to C21
    U[2, 1] = 1; U[3, 1] = 1  # A21, A22
    V[0, 1] = 1               # B11
    W[2, 1] = 1               # C21
    
    # M3 = A11(B12-B22) -> contributes to C12
    U[0, 2] = 1               # A11
    V[1, 2] = 1; V[3, 2] = -1 # B12, -B22
    W[1, 2] = 1               # C12
    
    # M4 = A22(B21-B11) -> contributes to C21
    U[3, 3] = 1               # A22
    V[2, 3] = 1; V[0, 3] = -1 # B21, -B11
    W[2, 3] = 1               # C21
    
    # M5 = (A11+A12)B22 -> contributes to C12
    U[0, 4] = 1; U[1, 4] = 1  # A11, A12
    V[3, 4] = 1               # B22
    W[1, 4] = 1               # C12
    
    # M6 = (A21-A11)(B11+B12) -> contributes to C22
    U[2, 5] = 1; U[0, 5] = -1 # A21, -A11
    V[0, 5] = 1; V[1, 5] = 1  # B11, B12
    W[3, 5] = 1               # C22
    
    # M7 = (A12-A22)(B21+B22) -> contributes to C11
    U[1, 6] = 1; U[3, 6] = -1 # A12, -A22
    V[2, 6] = 1; V[3, 6] = 1  # B21, B22
    W[0, 6] = 1               # C11
    
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
    
    # Build index arrays for all (i,j,k) triples
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
    
    # Compute sum_r U[a,r]*V[b,r]*W[c,r] for each (i,j,k)
    # U[a_arr, :] has shape (N, rank)
    # V[b_arr, :] has shape (N, rank)  
    # W[c_arr, :] has shape (N, rank)
    U_sel = U_arr[a_arr, :]  # (N, rank)
    V_sel = V_arr[b_arr, :]  # (N, rank)
    W_sel = W_arr[c_arr, :]  # (N, rank)
    
    # Element-wise product then sum over rank
    products = U_sel * V_sel * W_sel  # (N, rank)
    sums = products.sum(axis=1)  # (N,)
    
    if np.max(np.abs(sums - 1.0)) > 1e-9:
        return False
    
    # Check zero positions: for each (i,j,k), check that c' = (k+1)%p * n + i gives 0
    c_alt_arr = ((k_arr + 1) % p) * n + i_arr
    # Only check where c_alt != c
    mask = c_alt_arr != c_arr
    if np.any(mask):
        W_alt_sel = W_arr[c_alt_arr[mask], :]
        products_alt = U_sel[mask] * V_sel[mask] * W_alt_sel
        sums_alt = products_alt.sum(axis=1)
        if np.max(np.abs(sums_alt)) > 1e-9:
            return False
    
    return True

def _try_random_rank(r, n, m, p, deadline):
    """Try to find a rank-r decomposition via random search with local optimization."""
    nm = n * m
    mp = m * p
    pn = p * n
    start = time.time()
    
    for attempt in range(50):
        if time.time() > deadline - 10:
            return None
        # Random initialization with entries from {0, 1, -1}
        U = [[random.choice([0, 1, -1]) for _ in range(r)] for _ in range(nm)]
        V = [[random.choice([0, 1, -1]) for _ in range(r)] for _ in range(mp)]
        W = [[random.choice([0, 1, -1]) for _ in range(r)] for _ in range(pn)]
        
        for iteration in range(20):
            if time.time() > deadline - 10:
                break
            if _validate_fast(U, V, W, n, m, p):
                return U, V, W
            mat = random.choice([U, V, W])
            row = random.randrange(len(mat))
            col = random.randrange(r)
            mat[row][col] = -mat[row][col] if mat[row][col] != 0 else 1
            if time.time() > deadline - 10:
                break
    
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
    
    # Try to remove each column and check if the decomposition still works
    # This is a greedy approach: try removing columns one at a time
    changed = True
    while changed and time.time() < deadline - 15:
        changed = False
        for r_idx in range(len(keep) - 1, -1, -1):
            if time.time() > deadline - 15:
                break
            # Try removing column keep[r_idx]
            test_keep = keep[:r_idx] + keep[r_idx+1:]
            if len(test_keep) == 0:
                continue
            
            U_test = U_arr[:, test_keep]
            V_test = V_arr[:, test_keep]
            W_test = W_test_arr = W_arr[:, test_keep]
            
            # Quick check: validate
            U_list = U_test.tolist()
            V_list = V_test.tolist()
            W_list = W_test.tolist()
            
            if _validate_fast(U_list, V_list, W_list, n, m, p):
                keep = test_keep
                changed = True
                break  # Restart the loop with the new keep list
    
    if len(keep) < rank:
        U_final = U_arr[:, keep]
        V_final = V_arr[:, keep]
        W_final = W_arr[:, keep]
        return U_final.tolist(), V_final.tolist(), W_final.tolist()
    
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130  # safety margin
    
    # First, try the Strassen-based decomposition (rank 98 for 4x8)
    best_result = None
    best_rank = float('inf')
    
    try:
        result = _construct_strassen_4x8()
        U, V, W = result['U'], result['V'], result['W']
        if _validate_fast(U, V, W, n, m, p):
            best_result = result
            best_rank = len(U[0])
    except Exception:
        pass
    
    # Try to reduce rank from the Strassen solution
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
    
    # Try random search at various ranks
    if best_result is not None:
        # Try to find something better than current best
        search_start = max(1, best_rank - 20)
    else:
        search_start = 1
    
    for r in range(search_start, min(n * m * p + 1, 200)):
        if time.time() > deadline - 15:
            break
        if r >= best_rank:
            break
        result = _try_random_rank(r, n, m, p, deadline)
        if result is not None:
            U, V, W = result
            if _validate_fast(U, V, W, n, m, p):
                best_rank = r
                best_result = {"U": U, "V": V, "W": W}
                break  # Found a valid lower rank, stop
    
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
