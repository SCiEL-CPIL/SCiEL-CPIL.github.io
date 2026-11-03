"""Matrix multiplication 2x4x5 tensor decomposition - optimized Strassen + schoolbook."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W, n, m, p, rank):
    """Verify that the decomposition is exact using the tensor check."""
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    # Check matching triples: sum_c U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c] = 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U_np[i*m+j] * V_np[j*p+k] * W_np[k*n+i])
                if abs(s - 1) > 1e-8:
                    return False
    # Check non-matching triples sum to 0
    for a in range(n*m):
        for b in range(m*p):
            for c in range(p*n):
                i_a = a // m
                j_a = a % m
                j_b = b // p
                k_b = b % p
                k_c = c // n
                i_c = c % n
                if j_a == j_b and k_b == k_c and i_a == i_c:
                    continue
                s = np.sum(U_np[a] * V_np[b] * W_np[c])
                if abs(s) > 1e-8:
                    return False
    return True

def _strassen_2x2x2(offset_u, offset_v, offset_w, U, V, W, m, p, n, col_start):
    """
    Add a Strassen 2x2x2 block at given offsets.
    Returns new column offset.
    """
    col = col_start
    
    # M1 = (A00+A11)*(B00+B11)
    U[offset_u + 0][col] = 1
    U[offset_u + 3][col] = 1
    V[offset_v + 0][col] = 1
    V[offset_v + 3][col] = 1
    W[offset_w + 0][col] = 1
    W[offset_w + 3][col] = 1
    col += 1
    
    # M2 = (A10+A11)*B00
    U[offset_u + 2][col] = 1
    U[offset_u + 3][col] = 1
    V[offset_v + 0][col] = 1
    W[offset_w + 2][col] = 1
    W[offset_w + 3][col] = -1
    col += 1
    
    # M3 = A00*(B01-B11)
    U[offset_u + 0][col] = 1
    V[offset_v + 1][col] = 1
    V[offset_v + 3][col] = -1
    W[offset_w + 0][col] = 1
    W[offset_w + 3][col] = 1
    col += 1
    
    # M4 = A11*(B10-B00)
    U[offset_u + 3][col] = 1
    V[offset_v + 2][col] = 1
    V[offset_v + 0][col] = -1
    W[offset_w + 0][col] = 1
    W[offset_w + 2][col] = 1
    col += 1
    
    # M5 = (A00+A01)*B11
    U[offset_u + 0][col] = 1
    U[offset_u + 1][col] = 1
    V[offset_v + 3][col] = 1
    W[offset_w + 0][col] = -1
    W[offset_w + 1][col] = 1
    col += 1
    
    # M6 = (A10-A00)*(B00+B01)
    U[offset_u + 2][col] = 1
    U[offset_u + 0][col] = -1
    V[offset_v + 0][col] = 1
    V[offset_v + 1][col] = 1
    W[offset_w + 3][col] = 1
    col += 1
    
    # M7 = (A01-A11)*(B10+B11)
    U[offset_u + 1][col] = 1
    U[offset_u + 3][col] = -1
    V[offset_v + 2][col] = 1
    V[offset_v + 3][col] = 1
    W[offset_w + 0][col] = 1
    col += 1
    
    return col

def _schoolbook_2x2x1(offset_u, offset_v, offset_w, U, V, W, m, p, n, col_start):
    """
    Add a schoolbook 2x2x1 block (one column of B).
    U offset: 4 rows (i=0,1; j=0,1)
    V offset: 2 rows (j=0,1; k=0)
    W offset: 2 rows (k=0; i=0,1)
    Returns new column offset.
    """
    col = col_start
    
    # A[0,0]*B[0,0] -> C[0,0]
    U[offset_u + 0][col] = 1
    V[offset_v + 0][col] = 1
    W[offset_w + 0][col] = 1
    col += 1
    
    # A[0,1]*B[1,0] -> C[0,0]
    U[offset_u + 1][col] = 1
    V[offset_v + 1][col] = 1
    W[offset_w + 0][col] = 1
    col += 1
    
    # A[1,0]*B[0,0] -> C[1,0]
    U[offset_u + 2][col] = 1
    V[offset_v + 0][col] = 1
    W[offset_w + 1][col] = 1
    col += 1
    
    # A[1,1]*B[1,0] -> C[1,0]
    U[offset_u + 3][col] = 1
    V[offset_v + 1][col] = 1
    W[offset_w + 1][col] = 1
    col += 1
    
    return col

def _strassen_2x2x3(offset_u, offset_v, offset_w, U, V, W, m, p, n, col_start):
    """
    Add a Strassen 2x2x3 block: Strassen for cols 0-1 (rank 7) + schoolbook for col 2 (rank 4) = rank 11.
    U offset: 4 rows (i=0,1; j=0,1)
    V offset: 6 rows (j=0,1; k=0,1,2)
    W offset: 6 rows (k=0,1,2; i=0,1)
    Returns new column offset.
    """
    col = col_start
    
    # First 2 columns: Strassen 2x2x2
    col = _strassen_2x2x2(offset_u, offset_v, offset_w, U, V, W, m, p, n, col)
    
    # 3rd column: schoolbook 2x2x1
    # V offset for col 2: offset_v + 4 (j=0,1; k=2)
    # W offset for col 2: offset_w + 4 (k=2; i=0,1)
    col = _schoolbook_2x2x1(offset_u, offset_v + 4, offset_w + 4, U, V, W, m, p, n, col)
    
    return col

def _construct_hybrid():
    """
    Construct a hybrid Strassen+schoolbook decomposition for 2x4x5.
    
    Strategy:
    - Split A into two 2x2 blocks: A1 (cols 0-1), A2 (cols 2-3)
    - Split B into two 2x5 blocks: B1 (rows 0-1), B2 (rows 2-3)
    - C = A1*B1 + A2*B2
    
    For each 2x2 * 2x5 product, split the 5 columns as 2+3:
    - Use Strassen (rank 7) for the first 2 columns
    - Use Strassen+schoolbook (rank 11) for the next 3 columns
    
    Total per 2x2*2x5: 7 + 11 = 18
    Total for both: 36
    """
    n, m, p = 2, 4, 5
    total_rank = 36
    U = [[0.0] * total_rank for _ in range(n * m)]  # 8 rows
    V = [[0.0] * total_rank for _ in range(m * p)]  # 20 rows
    W = [[0.0] * total_rank for _ in range(p * n)]  # 10 rows
    
    col_offset = 0
    
    # Block 1: A1 (U rows 0-3) * B1 (V rows 0-9)
    # Sub-block 1a: A1 * B1_cols0-1 (V rows 0-3, W rows 0-3)
    col_offset = _strassen_2x2x2(0, 0, 0, U, V, W, m, p, n, col_offset)
    
    # Sub-block 1b: A1 * B1_cols2-4 (V rows 4-9, W rows 4-9)
    col_offset = _strassen_2x2x3(0, 4, 4, U, V, W, m, p, n, col_offset)
    
    # Block 2: A2 (U rows 4-7) * B2 (V rows 10-19)
    # Sub-block 2a: A2 * B2_cols0-1 (V rows 10-13, W rows 0-3)
    col_offset = _strassen_2x2x2(4, 10, 0, U, V, W, m, p, n, col_offset)
    
    # Sub-block 2b: A2 * B2_cols2-4 (V rows 14-19, W rows 4-9)
    col_offset = _strassen_2x2x3(4, 14, 4, U, V, W, m, p, n, col_offset)
    
    assert col_offset == total_rank, f"Expected {total_rank} columns, got {col_offset}"
    
    return {"U": U, "V": V, "W": W}

def _try_structured_search(n, m, p, target_rank, time_budget, seed=42):
    """
    Try to find a rank-`target_rank` decomposition via structured random search.
    Uses the insight that entries should be 0, ±1, or ±0.5 with sparse patterns.
    """
    rng = np.random.RandomState(seed)
    start = time.time()
    
    for attempt in range(500):
        if time.time() - start > time_budget:
            break
        
        U = np.zeros((n*m, target_rank))
        V = np.zeros((m*p, target_rank))
        W = np.zeros((p*n, target_rank))
        
        for c in range(target_rank):
            # Each factor vector has 2-4 non-zero entries
            nz_u = rng.choice(n*m, size=rng.randint(2, 4), replace=False)
            nz_v = rng.choice(m*p, size=rng.randint(2, 4), replace=False)
            nz_w = rng.choice(p*n, size=rng.randint(2, 4), replace=False)
            
            for idx in nz_u:
                U[idx, c] = rng.choice([-1, 1]) * rng.choice([0.5, 1.0])
            for idx in nz_v:
                V[idx, c] = rng.choice([-1, 1]) * rng.choice([0.5, 1.0])
            for idx in nz_w:
                W[idx, c] = rng.choice([-1, 1]) * rng.choice([0.5, 1.0])
        
        # Quick check: verify matching triples
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                    if abs(s - 1) > 1e-6:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if not valid:
            continue
        
        # Full verification
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
        if _verify(U_list, V_list, W_list, n, m, p, target_rank):
            return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    # First, get the constructive hybrid decomposition (rank 36)
    result = _construct_hybrid()
    
    U = result["U"]
    V = result["V"]
    W = result["W"]
    rank = len(U[0])
    
    if not _verify(U, V, W, n, m, p, rank):
        # Fallback to schoolbook if verification fails
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
    
    best_rank = rank
    best_result = result
    
    # Try to find a lower rank decomposition via structured search
    for target_rank in range(best_rank - 1, 24, -1):
        if time.time() > deadline - 20:
            break
        
        time_left = deadline - 20 - time.time()
        if time_left < 3:
            break
        
        search_time = min(time_left, 15)
        
        found = _try_structured_search(n, m, p, target_rank, search_time, seed=target_rank)
        
        if found is not None:
            best_rank = target_rank
            best_result = found
            continue
        else:
            break
    
    return best_result
# EVOLVE-BLOCK-END
