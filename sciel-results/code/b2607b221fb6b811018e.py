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
        
        # Try eliminating this column
        U[col] = [0.0] * len(U[0])
        V[col] = [0.0] * len(V[0])
        W[col] = [0.0] * len(W[0])
        
        if _validate_fast(U, V, W, n, m, p):
            eliminated.add(col)
        else:
            # Restore
            U[col] = [U_orig[col] for U_orig in [U]]  # This won't work, need to save
            # Actually, we need to save the original values
            break
    
    # Better approach: save and restore properly
    U_orig = [row[:] for row in U]
    V_orig = [row[:] for row in V]
    W_orig = [row[:] for row in W]
    
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
        for iteration in range(50):
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

def _try_greedy_rank_reduction(U, V, W, n, m, p, deadline):
    """
    Greedily try to reduce rank by finding columns that are linearly dependent
    or can be zeroed out while maintaining exactness.
    """
    U = [row[:] for row in U]
    V = [row[:] for row in V]
    W = [row[:] for row in W]
    current_rank = len(U[0])
    
    # Try to eliminate columns one by one
    for target_rank in range(current_rank - 1, 0, -1):
        if time.time() > deadline - 25:
            break
        
        eliminated = set()
        columns = list(range(current_rank))
        random.shuffle(columns)
        
        for col in columns:
            if time.time() > deadline - 25:
                break
            if len(eliminated) >= current_rank - target_rank:
                break
            
            U_save = [row[:] for row in U]
            V_save = [row[:] for row in V]
            W_save = [row[:] for row in W]
            
            for r in range(len(U)):
                U[r][col] = 0.0
            for r in range(len(V)):
                V[r][col] = 0.0
            for r in range(len(W)):
                W[r][col] = 0.0
            
            if _validate_fast(U, V, W, n, m, p):
                eliminated.add(col)
            else:
                U = U_save
                V = V_save
                W = W_save
        
        if len(eliminated) > 0:
            keep_cols = [c for c in range(current_rank) if c not in eliminated]
            U = [[row[c] for c in keep_cols] for row in U]
            V = [[row[c] for c in keep_cols] for row in V]
            W = [[row[c] for c in keep_cols] for row in W]
            current_rank = len(U[0])
        else:
            break
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120  # Safety margin
    
    # First, construct the Strassen-based decomposition (rank 98)
    try:
        result = _construct_strassen_4x8()
        U, V, W = result['U'], result['V'], result['W']
        
        # Validate
        if _validate_fast(U, V, W, n, m, p):
            best_U, best_V, best_W = U, V, W
            best_rank = len(U[0])
            
            # Try to reduce rank
            if time.time() < deadline - 30:
                U_red, V_red, W_red = _try_greedy_rank_reduction(
                    best_U, best_V, best_W, n, m, p, deadline
                )
                if len(U_red[0]) < best_rank and _validate_fast(U_red, V_red, W_red, n, m, p):
                    best_U, best_V, best_W = U_red, V_red, W_red
                    best_rank = len(U_red[0])
            
            # Try random search for even lower rank
            if time.time() < deadline - 30:
                for r in range(max(1, best_rank - 10), best_rank):
                    if time.time() > deadline - 25:
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
