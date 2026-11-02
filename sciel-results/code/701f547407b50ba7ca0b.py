"""Matrix multiplication 4,4,8 rank-minimal search with Strassen decomposition and rank reduction."""
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

def _validate(U, V, W, n, m, p):
    """Check if the decomposition is exact for the 4x4 * 4x8 tensor."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                s = 0.0
                for r in range(rank):
                    s += U[a][r] * V[b][r] * W[c][r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _validate_fast(U, V, W, n, m, p):
    """Fast vectorized validation using numpy."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                s = np.dot(np.dot(U_arr[a], V_arr[b]), W_arr[c])
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _construct_strassen_4x8():
    """Construct Strassen-based decomposition for 4x4 * 4x8."""
    U4, V4, W4 = _strassen_4x4_factors()
    U, V, W = _adapt_to_4x8(U4, V4, W4)
    return {"U": U, "V": V, "W": W}

def _try_reduce_rank(U, V, W, target_rank, n, m, p, deadline, max_iterations=5000):
    """
    Try to reduce rank by attempting to set some columns to zero 
    while maintaining exactness. Uses local search.
    """
    U = [row[:] for row in U]
    V = [row[:] for row in V]
    W = [row[:] for row in W]
    current_rank = len(U[0])
    
    if current_rank <= target_rank:
        return U, V, W
    
    # Try to find columns that can be eliminated
    # Strategy: try setting random subsets of columns to zero and check validity
    # More efficient: try to find linear dependencies among columns
    
    # First, try simple column elimination
    columns_to_try = list(range(current_rank))
    random.shuffle(columns_to_try)
    
    eliminated = set()
    for col in columns_to_try:
        if time.time() > deadline - 20:
            break
        if len(eliminated) >= current_rank - target_rank:
            break
        
        # Save current state
        U_save = [row[:] for row in U]
        V_save = [row[:] for row in V]
        W_save = [row[:] for row in W]
        
        # Try eliminating this column
        for r in range(len(U)):
            U[r][col] = 0.0
        for r in range(len(V)):
            V[r][col] = 0.0
        for r in range(len(W)):
            W[r][col] = 0.0
        
        if _validate_fast(U, V, W, n, m, p):
            eliminated.add(col)
        else:
            # Restore
            U = U_save
            V = V_save
            W = W_save
    
    if len(eliminated) > 0:
        # Remove eliminated columns
        keep_cols = [c for c in range(current_rank) if c not in eliminated]
        U = [[row[c] for c in keep_cols] for row in U]
        V = [[row[c] for c in keep_cols] for row in V]
        W = [[row[c] for c in keep_cols] for row in W]
    
    return U, V, W

def _try_random_rank(r, n, m, p, deadline, max_attempts=20):
    """Try to find a rank-r decomposition via random search with local optimization."""
    nm = n * m
    mp = m * p
    pn = p * n
    
    for attempt in range(max_attempts):
        if time.time() > deadline - 20:
            return None
        
        # Initialize with entries from {-1, 0, 1}
        U = [[random.choice([-1, 0, 1]) for _ in range(r)] for _ in range(nm)]
        V = [[random.choice([-1, 0, 1]) for _ in range(r)] for _ in range(mp)]
        W = [[random.choice([-1, 0, 1]) for _ in range(r)] for _ in range(pn)]
        
        # Local search: try flipping entries
        for iteration in range(100):
            if time.time() > deadline - 20:
                break
            if _validate_fast(U, V, W, n, m, p):
                return U, V, W
            
            # Pick a random entry to modify
            mat_name = random.choice(['U', 'V', 'W'])
            if mat_name == 'U':
                mat = U
            elif mat_name == 'V':
                mat = V
            else:
                mat = W
            
            row = random.randrange(len(mat))
            col = random.randrange(r)
            
            # Try changing to a different value
            old_val = mat[row][col]
            new_val = random.choice([-1, 0, 1])
            if new_val == old_val:
                new_val = (old_val + 1) % 3 - 1  # Cycle through -1, 0, 1
            
            mat[row][col] = new_val
    
    return None

def _find_redundant_columns(U, V, W, n, m, p, deadline):
    """
    Find columns that are linearly redundant using tensor network structure.
    A column r is redundant if there exist coefficients alpha_r, beta_r, gamma_r
    such that the contribution of column r can be absorbed by other columns.
    
    More practically: try pairs of columns and see if one can be eliminated
    by adjusting the other.
    """
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    current_rank = U_arr.shape[1]
    
    eliminated = set()
    columns = list(range(current_rank))
    random.shuffle(columns)
    
    for col in columns:
        if time.time() > deadline - 20:
            break
        
        # Try to eliminate column col by checking if its contribution is zero
        # or can be absorbed
        U_save = U_arr.copy()
        V_save = V_arr.copy()
        W_save = W_arr.copy()
        
        U_arr[:, col] = 0.0
        V_arr[:, col] = 0.0
        W_arr[:, col] = 0.0
        
        if _validate_fast(U_arr.tolist(), V_arr.tolist(), W_arr.tolist(), n, m, p):
            eliminated.add(col)
        else:
            U_arr = U_save
            V_arr = V_save
            W_arr = W_save
    
    if len(eliminated) > 0:
        keep_cols = [c for c in range(current_rank) if c not in eliminated]
        U = [[row[c] for c in keep_cols] for row in U]
        V = [[row[c] for c in keep_cols] for row in V]
        W = [[row[c] for c in keep_cols] for row in W]
    
    return U, V, W, len(eliminated)

def _local_search_reduce(U, V, W, n, m, p, deadline, max_iters=10000):
    """
    Local search: perturb entries to enable rank reduction.
    Strategy: randomly modify entries, then try to eliminate columns.
    """
    U = [row[:] for row in U]
    V = [row[:] for row in V]
    W = [row[:] for row in W]
    current_rank = len(U[0])
    best_U, best_V, best_W = U, V, W
    best_rank = current_rank
    
    for iteration in range(max_iters):
        if time.time() > deadline - 20:
            break
        
        # Try to eliminate a random column
        col = random.randrange(current_rank)
        
        U_save = [row[:] for row in U]
        V_save = [row[:] for row in V]
        W_save = [row[:] for row in W]
        
        # Zero out column
        for r in range(len(U)):
            U[r][col] = 0.0
        for r in range(len(V)):
            V[r][col] = 0.0
        for r in range(len(W)):
            W[r][col] = 0.0
        
        if _validate_fast(U, V, W, n, m, p):
            # Successfully eliminated
            keep_cols = [c for c in range(current_rank) if c != col]
            U = [[row[c] for c in keep_cols] for row in U]
            V = [[row[c] for c in keep_cols] for row in V]
            W = [[row[c] for c in keep_cols] for row in W]
            current_rank -= 1
            best_U, best_V, best_W = U, V, W
            best_rank = current_rank
        else:
            # Restore and try a perturbation
            U = U_save
            V = V_save
            W = W_save
            
            # Perturb a random entry
            mat_name = random.choice(['U', 'V', 'W'])
            if mat_name == 'U':
                mat = U
            elif mat_name == 'V':
                mat = V
            else:
                mat = W
            
            row = random.randrange(len(mat))
            col_p = random.randrange(len(mat[0]))
            old_val = mat[row][col_p]
            
            # Try small perturbations
            for delta in [-0.5, 0.5, -1.0, 1.0]:
                new_val = old_val + delta
                mat[row][col_p] = new_val
                
                if _validate_fast(U, V, W, n, m, p):
                    # Check if we can now eliminate a column
                    for test_col in range(len(U[0])):
                        U_test = [row[:] for row in U]
                        V_test = [row[:] for row in V]
                        W_test = [row[:] for row in W]
                        
                        for r in range(len(U_test)):
                            U_test[r][test_col] = 0.0
                        for r in range(len(V_test)):
                            V_test[r][test_col] = 0.0
                        for r in range(len(W_test)):
                            W_test[r][test_col] = 0.0
                        
                        if _validate_fast(U_test, V_test, W_test, n, m, p):
                            # Keep the perturbation and elimination
                            keep_cols = [c for c in range(len(U[0])) if c != test_col]
                            U = [[row[c] for c in keep_cols] for row in U]
                            V = [[row[c] for c in keep_cols] for row in V]
                            W = [[row[c] for c in keep_cols] for row in W]
                            current_rank -= 1
                            best_U, best_V, best_W = U, V, W
                            best_rank = current_rank
                            break
                        else:
                            # Restore
                            U = U_save
                            V = V_save
                            W = W_save
                            mat[row][col_p] = old_val
                    else:
                        # No elimination possible, restore perturbation
                        mat[row][col_p] = old_val
                else:
                    mat[row][col_p] = old_val
    
    return best_U, best_V, best_W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin (10s buffer before 120s limit)
    
    # First, construct the Strassen-based decomposition (rank 98)
    try:
        result = _construct_strassen_4x8()
        U, V, W = result['U'], result['V'], result['W']
        
        # Validate
        if _validate_fast(U, V, W, n, m, p):
            best_U, best_V, best_W = U, V, W
            best_rank = len(U[0])
            
            # Try greedy rank reduction
            if time.time() < deadline - 40:
                U_red, V_red, W_red = _try_greedy_rank_reduction(
                    best_U, best_V, best_W, n, m, p, deadline
                )
                if len(U_red[0]) < best_rank and _validate_fast(U_red, V_red, W_red, n, m, p):
                    best_U, best_V, best_W = U_red, V_red, W_red
                    best_rank = len(U_red[0])
            
            # Try local search for further reduction
            if time.time() < deadline - 30:
                U_ls, V_ls, W_ls = _local_search_reduce(
                    best_U, best_V, best_W, n, m, p, deadline
                )
                if len(U_ls[0]) < best_rank and _validate_fast(U_ls, V_ls, W_ls, n, m, p):
                    best_U, best_V, best_W = U_ls, V_ls, W_ls
                    best_rank = len(U_ls[0])
            
            # Try random search for even lower rank
            if time.time() < deadline - 25:
                for r in range(max(1, best_rank - 15), best_rank):
                    if time.time() > deadline - 20:
                        break
                    result_rand = _try_random_rank(r, n, m, p, deadline)
                    if result_rand is not None:
                        U_r, V_r, W_r = result_rand
                        if _validate_fast(U_r, V_r, W_r, n, m, p):
                            best_U, best_V, best_W = U_r, V_r, W_r
                            best_rank = r
                            break
            
            return {"U": best_U, "V": best_V, "W": best_W}
        else:
            pass
    except Exception:
        pass
    
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
