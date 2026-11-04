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

def construct():
    start_time = time.time()
    DEADLINE = 140  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    # Schoolbook: rank = n*m*p = 40
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
    
    # Try to find lower rank decompositions
    # Lower bound: max(n*m, m*p, p*n) = max(8, 20, 10) = 20
    best_rank = rank_schoolbook
    best_U, best_V, best_W = U_school, V_school, W_school
    
    # Try a range of ranks from lower bound up to schoolbook
    lower_bound = max(n*m, m*p, p*n)
    
    for rank in range(lower_bound, rank_schoolbook + 1, 2):
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 15:
            break
        
        time_budget = min(10.0, DEADLINE - elapsed - 5)
        if time_budget < 2:
            break
            
        U, V, W, success = _try_ranked_decomposition(n, m, p, rank, time_budget, seed=rank*7+13)
        if success:
            best_rank = rank
            best_U, best_V, best_W = U, V, W
            break  # Found a valid lower rank, stop
    
    # Return the best decomposition found
    # Ensure entries are proper integers or half-integers
    def _clean_entry(v):
        if abs(v - round(v)) < 1e-9:
            return int(round(v))
        doubled = v * 2
        if abs(doubled - round(doubled)) < 1e-9:
            return round(doubled) / 2.0
        return v
    
    result_U = [[_clean_entry(v) for v in row] for row in best_U]
    result_V = [[_clean_entry(v) for v in row] for row in best_V]
    result_W = [[_clean_entry(v) for v in row] for row in best_W]
    
    return {"U": result_U, "V": result_V, "W": result_W}
# EVOLVE-BLOCK-END
