"""Program for matrix multiplication tensor decomposition: 2x4 * 4x5."""
import time
import random
import numpy as np
from itertools import product

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

def _project_to_half_integer(val, tol=1e-9):
    """Project a value to the nearest half-integer."""
    doubled = val * 2
    return round(doubled) / 2.0

def _compute_residuals(U, V, W, n=2, m=4, p=5):
    """Compute residuals for each (i,j,k) triple."""
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    rank = U_np.shape[1]
    
    residuals = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                prod = U_np[i*m+j] * V_np[j*p+k] * W_np[k*n+i]
                val = np.sum(prod)
                residuals[(i,j,k)] = 1.0 - val
    return residuals

def _try_ranked_decomposition(n, m, p, rank, time_budget=10.0, seed=42):
    """Try to find a rank-r decomposition using structured initialization."""
    start = time.time()
    rng = random.Random(seed)
    
    # Strategy 1: Block diagonal (schoolbook-like but with rank < n*m*p)
    # Strategy 2: Structured linear combinations
    # Strategy 3: Random sparse with refinement
    # Strategy 4: Partition-based
    
    best_result = None
    best_residual_norm = float('inf')
    
    for strategy_idx in range(4):
        if time.time() - start > time_budget:
            break
        
        U = [[0.0]*rank for _ in range(n*m)]
        V = [[0.0]*rank for _ in range(m*p)]
        W = [[0.0]*rank for _ in range(p*n)]
        
        if strategy_idx == 0:
            # Block diagonal: assign rank slots to (i,j,k) triples
            count = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        if count < rank:
                            U[i*m+j][count] = 1.0
                            V[j*p+k][count] = 1.0
                            W[k*n+i][count] = 1.0
                            count += 1
                            
        elif strategy_idx == 1:
            # Structured: use Kronecker-like patterns
            # Try to exploit linearity in i and k
            # For each j, we need to compute a 2x5 product
            # Split into blocks
            j_blocks = 2  # split 4 columns into 2 blocks of 2
            k_blocks = 5  # each output column
            # This is complex, so use random signs with structure
            for c in range(rank):
                # Random sign patterns with some structure
                for i in range(n):
                    for j in range(m):
                        U[i*m+j][c] = float(rng.choice([-1, 0, 1]))
                for j in range(m):
                    for k in range(p):
                        V[j*p+k][c] = float(rng.choice([-1, 0, 1]))
                for k in range(p):
                    for i in range(n):
                        W[k*n+i][c] = float(rng.choice([-1, 0, 1]))
                        
        elif strategy_idx == 2:
            # Random sparse
            density = 0.4
            for c in range(rank):
                for i in range(n):
                    for j in range(m):
                        if rng.random() < density:
                            U[i*m+j][c] = float(rng.choice([-1, 1]))
                for j in range(m):
                    for k in range(p):
                        if rng.random() < density:
                            V[j*p+k][c] = float(rng.choice([-1, 1]))
                for k in range(p):
                    for i in range(n):
                        if rng.random() < density:
                            W[k*n+i][c] = float(rng.choice([-1, 1]))
                            
        elif strategy_idx == 3:
            # Partition-based: split the multiplication
            # Idea: use rank to handle multiple (i,j,k) at once
            # Assign each rank slot to handle a subset
            slots_per_rank = max(1, (n*m*p) // rank)
            c = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        if c < rank:
                            U[i*m+j][c] = 1.0
                            V[j*p+k][c] = 1.0
                            W[k*n+i][c] = 1.0
                        c += 1
                        if c >= rank:
                            break
                    if c >= rank:
                        break
                if c >= rank:
                    break
            # Fill remaining rank slots with random patterns
            while c < rank:
                for i in range(n):
                    for j in range(m):
                        if rng.random() < 0.3:
                            U[i*m+j][c] = float(rng.choice([-1, 1]))
                for j in range(m):
                    for k in range(p):
                        if rng.random() < 0.3:
                            V[j*p+k][c] = float(rng.choice([-1, 1]))
                for k in range(p):
                    for i in range(n):
                        if rng.random() < 0.3:
                            W[k*n+i][c] = float(rng.choice([-1, 1]))
                c += 1
        
        # Check if this initialization works
        if _all_half_integer(U, V, W) and _validate_numpy(U, V, W, n, m, p):
            return U, V, W, True
            
        # Compute current residual norm
        residuals = _compute_residuals(U, V, W, n, m, p)
        residual_norm = sum(abs(r) for r in residuals.values())
        
        # Local refinement: try modifying entries to reduce residuals
        for iteration in range(300):
            if time.time() - start > time_budget:
                break
            
            # Choose which matrix to modify based on which has largest residuals
            mat_idx = rng.randint(0, 2)
            mat = [U, V, W][mat_idx]
            row = rng.randint(0, len(mat)-1)
            col = rng.randint(0, rank-1)
            
            old_val = mat[row][col]
            
            # Try several candidate values
            candidates = [old_val]
            for cv in [-1.0, -0.5, 0.0, 0.5, 1.0]:
                if abs(cv - old_val) > 1e-9:
                    candidates.append(cv)
            # Also try random half-integers
            for _ in range(3):
                candidates.append(float(rng.choice([-2, -1, -0.5, 0, 0.5, 1, 2])) / 2.0)
            
            best_new_val = old_val
            best_new_residual = residual_norm
            
            for new_val in candidates:
                mat[row][col] = new_val
                residuals = _compute_residuals(U, V, W, n, m, p)
                new_residual = sum(abs(r) for r in residuals.values())
                if new_residual < best_new_residual:
                    best_new_residual = new_residual
                    best_new_val = new_val
            
            mat[row][col] = best_new_val
            residual_norm = best_new_residual
            
            # Check if we've solved it
            if _all_half_integer(U, V, W) and _validate_numpy(U, V, W, n, m, p):
                return U, V, W, True
            
            # Track best found
            if residual_norm < best_residual_norm:
                best_residual_norm = residual_norm
                best_result = (
                    [row[:] for row in U],
                    [row[:] for row in V],
                    [row[:] for row in W]
                )
                
            # Early exit if residual is very small
            if residual_norm < 1e-6:
                break
    
    # If we found a near-solution, try to polish it
    if best_result is not None and best_residual_norm < 0.1:
        U, V, W = best_result
        # Final polish: try to snap to exact half-integers
        for _ in range(100):
            if time.time() - start > time_budget:
                break
            if _all_half_integer(U, V, W) and _validate_numpy(U, V, W, n, m, p):
                return U, V, W, True
            # Try small adjustments
            mat_idx = rng.randint(0, 2)
            mat = [U, V, W][mat_idx]
            row = rng.randint(0, len(mat)-1)
            col = rng.randint(0, rank-1)
            old_val = mat[row][col]
            for delta in [-0.5, 0.5]:
                mat[row][col] = old_val + delta
                if _all_half_integer(U, V, W) and _validate_numpy(U, V, W, n, m, p):
                    return U, V, W, True
            mat[row][col] = old_val
    
    return None, None, None, False

def _try_partition_decomposition(n, m, p, time_budget=20.0, seed=42):
    """Try to find a decomposition using partitioning strategies."""
    start = time.time()
    rng = random.Random(seed)
    
    # Strategy: split the 4x5 multiplication into smaller blocks
    # 2x4 * 4x5: split the 4 into 2+2
    # This gives two 2x2 * 2x5 multiplications
    # Each 2x2 * 2x5 has rank at most 2*2*5 = 20, but might be less
    
    # Try to find rank for 2x2 * 2x5 first
    # Then combine
    
    # Actually, let's try a direct approach:
    # Use the fact that we can share computations
    
    # For each pair of rows (i1, i2) in the 2x4 matrix and each column k in the 5,
    # we compute sum_j A[i,j] * B[j,k]
    
    # Try rank = 20 (lower bound)
    for rank in [20, 22, 24, 26, 28, 30, 32, 34, 36, 38]:
        if time.time() - start > time_budget:
            break
        
        U, V, W, success = _try_ranked_decomposition(n, m, p, rank, 
                                                       time_budget=min(5.0, time_budget - (time.time() - start)),
                                                       seed=rank*11+7)
        if success:
            return U, V, W, True
    
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
    best_rank = rank_schoolbook
    best_U, best_V, best_W = U_school, V_school, W_school
    
    # Try partition-based search first
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - 30:
        time_budget = min(25.0, DEADLINE - elapsed - 15)
        U, V, W, success = _try_partition_decomposition(n, m, p, time_budget, seed=12345)
        if success:
            best_rank = len(U[0])
            best_U, best_V, best_W = U, V, W
    
    # Try direct rank search if we haven't found a good solution
    elapsed = time.time() - start_time
    lower_bound = max(n*m, m*p, p*n)
    
    if best_rank > lower_bound and elapsed < DEADLINE - 15:
        for rank in range(lower_bound, best_rank, 2):
            elapsed = time.time() - start_time
            if elapsed > DEADLINE - 10:
                break
            
            time_budget = min(8.0, DEADLINE - elapsed - 5)
            if time_budget < 2:
                break
                
            U, V, W, success = _try_ranked_decomposition(n, m, p, rank, time_budget, seed=rank*7+13)
            if success:
                best_rank = rank
                best_U, best_V, best_W = U, V, W
                break
    
    # Return the best decomposition found
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
