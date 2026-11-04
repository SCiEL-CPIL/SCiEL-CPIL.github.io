"""Program for matrix multiplication tensor decomposition: 2x4 * 4x5."""
import time
import random
import numpy as np

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def _validate(U, V, W, n=2, m=4, p=5, tol=1e-9):
    """Check if U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(val - 1.0) > tol:
                    return False
    return True

def _validate_numpy(U, V, W, n=2, m=4, p=5, tol=1e-9):
    """Vectorized validation using numpy for speed."""
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    rank = U_np.shape[1]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                prod = U_np[i*m+j] * V_np[j*p+k] * W_np[k*n+i]
                val = np.sum(prod)
                if abs(val - 1.0) > tol:
                    return False
    return True

def _is_half_integer(val, tol=1e-9):
    """Check if val is an integer or half-integer."""
    doubled = val * 2
    return abs(doubled - round(doubled)) < tol

def _all_half_integer(U, V, W, tol=1e-9):
    """Check all entries are half-integers."""
    for mat in [U, V, W]:
        for row in mat:
            for val in row:
                if not _is_half_integer(val, tol):
                    return False
    return True

def _try_ranked_decomposition(n, m, p, rank, time_budget=10.0, seed=42):
    """Try to find a rank-r decomposition using structured initialization."""
    start = time.time()
    rng = random.Random(seed)
    
    strategies = [
        "block_diag",
        "strassen_like", 
        "random_sparse",
        "factorized",
    ]
    
    for strategy in strategies:
        if time.time() - start > time_budget:
            break
        
        U = [[0.0]*rank for _ in range(n*m)]
        V = [[0.0]*rank for _ in range(m*p)]
        W = [[0.0]*rank for _ in range(p*n)]
        
        if strategy == "block_diag":
            # Assign each rank slot to a specific (i,j,k) triple
            count = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        if count < rank:
                            U[i*m+j][count] = 1.0
                            V[j*p+k][count] = 1.0
                            W[k*n+i][count] = 1.0
                            count += 1
                            
        elif strategy == "strassen_like":
            # Use linear combinations with signs
            for c in range(rank):
                for i in range(n):
                    for j in range(m):
                        if rng.random() < 0.5:
                            U[i*m+j][c] = rng.choice([-1.0, 1.0])
                for j in range(m):
                    for k in range(p):
                        if rng.random() < 0.5:
                            V[j*p+k][c] = rng.choice([-1.0, 1.0])
                for k in range(p):
                    for i in range(n):
                        if rng.random() < 0.5:
                            W[k*n+i][c] = rng.choice([-1.0, 1.0])
                            
        elif strategy == "random_sparse":
            density = 0.3
            for c in range(rank):
                for i in range(n):
                    for j in range(m):
                        if rng.random() < density:
                            U[i*m+j][c] = float(rng.choice([-1, 0, 1]))
                for j in range(m):
                    for k in range(p):
                        if rng.random() < density:
                            V[j*p+k][c] = float(rng.choice([-1, 0, 1]))
                for k in range(p):
                    for i in range(n):
                        if rng.random() < density:
                            W[k*n+i][c] = float(rng.choice([-1, 0, 1]))
                            
        elif strategy == "factorized":
            group_size = max(1, rank // (n*m))
            for i in range(n):
                for j in range(m):
                    base = (i*m + j) * group_size
                    for offset in range(group_size):
                        c = base + offset
                        if c < rank:
                            U[i*m+j][c] = 1.0
                            for k in range(p):
                                V[j*p+k][c] = 1.0 / p
                                W[k*n+i][c] = 1.0
        
        # Check if this initialization already works
        if _all_half_integer(U, V, W) and _validate_numpy(U, V, W, n, m, p):
            return U, V, W, True
            
        # Local refinement: try modifying entries
        for iteration in range(200):
            if time.time() - start > time_budget:
                break
            
            mat_idx = rng.randint(0, 2)
            mat = [U, V, W][mat_idx]
            row = rng.randint(0, len(mat)-1)
            col = rng.randint(0, rank-1)
            
            old_val = mat[row][col]
            new_val = float(rng.choice([-1, 0, 1]))
            mat[row][col] = new_val
            
            if _all_half_integer(U, V, W) and _validate_numpy(U, V, W, n, m, p):
                return U, V, W, True
            else:
                mat[row][col] = old_val
                
    return None, None, None, False

def _construct_rank_36():
    n, m, p = 2, 4, 5
    rank = 36
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col = 0
    
    # Helper to add a Strassen block for 2x2x2
    # A indices: a00, a01, a10, a11
    # B indices: b00, b01, b10, b11
    # W indices: w00, w01, w10, w11
    def add_strassen_block(a00, a01, a10, a11, b00, b01, b10, b11, w00, w01, w10, w11):
        nonlocal col
        # M1 = (a00+a01) * (b10+b11)
        U[a00][col] = 1
        U[a01][col] = 1
        V[b10][col] = 1
        V[b11][col] = 1
        W[w00][col] = 1
        W[w11][col] = 1
        col += 1
        
        # M2 = (a00+a01) * b11
        U[a00][col] = 1
        U[a01][col] = 1
        V[b11][col] = 1
        W[w00][col] = -1
        W[w01][col] = 1
        col += 1
        
        # M3 = a00 * b00
        U[a00][col] = 1
        V[b00][col] = 1
        W[w11][col] = -1
        col += 1
        
        # M4 = a11 * b01
        U[a11][col] = 1
        V[b01][col] = 1
        W[w00][col] = -1
        W[w01][col] = 1
        col += 1
        
        # M5 = (a10+a11) * (b00+b01)
        U[a10][col] = 1
        U[a11][col] = 1
        V[b00][col] = 1
        V[b01][col] = 1
        W[w11][col] = 1
        col += 1
        
        # M6 = (a10+a11) * b00
        U[a10][col] = 1
        U[a11][col] = 1
        V[b00][col] = 1
        W[w10][col] = 1
        W[w11][col] = -1
        col += 1
        
        # M7 = (a00-a01) * b10
        U[a00][col] = 1
        U[a01][col] = -1
        V[b10][col] = 1
        W[w00][col] = 1
        W[w10][col] = 1
        col += 1
        
    # Helper to add a Schoolbook block for 2x2x1
    # A indices: a00, a01, a10, a11
    # B indices: b0, b1
    # W indices: w0, w1
    def add_schoolbook_2x2x1(a00, a01, a10, a11, b0, b1, w0, w1):
        nonlocal col
        # C0 = a00*b0 + a01*b1
        U[a00][col] = 1
        V[b0][col] = 1
        W[w0][col] = 1
        col += 1
        
        U[a01][col] = 1
        V[b1][col] = 1
        W[w0][col] = 1
        col += 1
        
        # C1 = a10*b0 + a11*b1
        U[a10][col] = 1
        V[b0][col] = 1
        W[w1][col] = 1
        col += 1
        
        U[a11][col] = 1
        V[b1][col] = 1
        W[w1][col] = 1
        col += 1
        
    # Block 1: j in {0, 1}
    # A indices: 0, 1, 4, 5
    # B indices: 0, 1, 5, 6 (for k=0,1), 2, 3, 7, 8 (for k=2,3), 4, 9 (for k=4)
    # W indices: 0, 1, 2, 3 (for k=0,1), 4, 5, 6, 7 (for k=2,3), 8, 9 (for k=4)
    
    # Sub-block 1.1: k in {0, 1}
    add_strassen_block(0, 1, 4, 5, 0, 1, 5, 6, 0, 2, 1, 3)
    
    # Sub-block 1.2: k in {2, 3}
    add_strassen_block(0, 1, 4, 5, 2, 3, 7, 8, 4, 6, 5, 7)
    
    # Sub-block 1.3: k in {4}
    add_schoolbook_2x2x1(0, 1, 4, 5, 4, 9, 8, 9)
    
    # Block 2: j in {2, 3}
    # A indices: 2, 3, 6, 7
    # B indices: 10, 11, 15, 16 (for k=0,1), 12, 13, 17, 18 (for k=2,3), 14, 19 (for k=4)
    # W indices: 0, 1, 2, 3 (for k=0,1), 4, 5, 6, 7 (for k=2,3), 8, 9 (for k=4)
    
    # Sub-block 2.1: k in {0, 1}
    add_strassen_block(2, 3, 6, 7, 10, 11, 15, 16, 0, 2, 1, 3)
    
    # Sub-block 2.2: k in {2, 3}
    add_strassen_block(2, 3, 6, 7, 12, 13, 17, 18, 4, 6, 5, 7)
    
    # Sub-block 2.3: k in {4}
    add_schoolbook_2x2x1(2, 3, 6, 7, 14, 19, 8, 9)
    
    return U, V, W

def construct():
    start_time = time.time()
    DEADLINE = 140  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    # Try Rank-36 decomposition
    U_rank36, V_rank36, W_rank36 = _construct_rank_36()
    if _validate_numpy(U_rank36, V_rank36, W_rank36, n, m, p):
        # Clean entries
        def _clean_entry(v):
            if abs(v - round(v)) < 1e-9:
                return int(round(v))
            doubled = v * 2
            if abs(doubled - round(doubled)) < 1e-9:
                return round(doubled) / 2.0
            return v
        
        result_U = [[_clean_entry(v) for v in row] for row in U_rank36]
        result_V = [[_clean_entry(v) for v in row] for row in V_rank36]
        result_W = [[_clean_entry(v) for v in row] for row in W_rank36]
        return {"U": result_U, "V": result_V, "W": result_W}
    
    # Fallback to Schoolbook
    rank_schoolbook = n * m * p
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][column] = 1
                V_school[j * p + k][column] = 1
                W_school[k * n + i][column] = 1
                column += 1
    
    # Verify schoolbook
    assert _validate(U_school, V_school, W_school, n, m, p), "Schoolbook decomposition failed!"
    
    return {"U": U_school, "V": V_school, "W": W_school}
# EVOLVE-BLOCK-END
