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

def _validate_half_integer(U, V, W, tol=1e-9):
    """Check entries are half-integers and validate decomposition."""
    for mat in [U, V, W]:
        for row in mat:
            for val in row:
                if abs(val * 2 - round(val * 2)) > tol:
                    return False
    return _validate(U, V, W)

def _try_ranked_decomposition(n, m, p, rank, time_budget=10.0, seed=42):
    """Try to find a rank-r decomposition using structured initialization."""
    start = time.time()
    rng = random.Random(seed)
    
    # Try different structured initializations
    strategies = [
        "block_diag",      # Block diagonal structure
        "strassen_like",   # Inspired by Strassen's algorithm
        "random_sparse",   # Random sparse initialization
        "factorized",      # Factorized structure
    ]
    
    for strategy in strategies:
        if time.time() - start > time_budget:
            break
            
        if strategy == "block_diag":
            # Initialize with block structure
            U = [[0.0]*rank for _ in range(n*m)]
            V = [[0.0]*rank for _ in range(m*p)]
            W = [[0.0]*rank for _ in range(p*n)]
            
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
            # For 2x2x2, Strassen uses 7 multiplications
            # Try to extend this idea
            U = [[0.0]*rank for _ in range(n*m)]
            V = [[0.0]*rank for _ in range(m*p)]
            W = [[0.0]*rank for _ in range(p*n)]
            
            # Use linear combinations
            for c in range(rank):
                # Random linear combinations
                for i in range(n):
                    for j in range(m):
                        if rng.random() < 0.5:
                            U[i*m+j][c] = rng.choice([-1, 1])
                for j in range(m):
                    for k in range(p):
                        if rng.random() < 0.5:
                            V[j*p+k][c] = rng.choice([-1, 1])
                for k in range(p):
                    for i in range(n):
                        if rng.random() < 0.5:
                            W[k*n+i][c] = rng.choice([-1, 1])
                            
        elif strategy == "random_sparse":
            U = [[0.0]*rank for _ in range(n*m)]
            V = [[0.0]*rank for _ in range(m*p)]
            W = [[0.0]*rank for _ in range(p*n)]
            
            density = 0.3
            for c in range(rank):
                for i in range(n):
                    for j in range(m):
                        if rng.random() < density:
                            U[i*m+j][c] = rng.choice([-1, 0, 1])
                for j in range(m):
                    for k in range(p):
                        if rng.random() < density:
                            V[j*p+k][c] = rng.choice([-1, 0, 1])
                for k in range(p):
                    for i in range(n):
                        if rng.random() < density:
                            W[k*n+i][c] = rng.choice([-1, 0, 1])
                            
        elif strategy == "factorized":
            # Try to exploit the factorization structure
            U = [[0.0]*rank for _ in range(n*m)]
            V = [[0.0]*rank for _ in range(m*p)]
            W = [[0.0]*rank for _ in range(p*n)]
            
            # Split rank into groups
            group_size = max(1, rank // (n*m))
            for i in range(n):
                for j in range(m):
                    base = (i*m + j) * group_size
                    for offset in range(group_size):
                        c = base + offset
                        if c < rank:
                            U[i*m+j][c] = 1.0
                            # Distribute V and W
                            for k in range(p):
                                V[j*p+k][c] = 1.0 / p
                                W[k*n+i][c] = 1.0
                            
        # Try to refine using local search
        if _validate_half_integer(U, V, W):
            return U, V, W, True
            
        # Local refinement: try flipping signs and zeroing entries
        improved = False
        for _ in range(100):
            if time.time() - start > time_budget:
                break
            # Pick a random entry to modify
            mat_idx = rng.randint(0, 2)
            mat = [U, V, W][mat_idx]
            row = rng.randint(0, len(mat)-1)
            col = rng.randint(0, rank-1)
            
            old_val = mat[row][col]
            new_val = rng.choice([-1, 0, 1])
            mat[row][col] = new_val
            
            if _validate_half_integer(U, V, W):
                return U, V, W, True
            else:
                mat[row][col] = old_val
                
    return None, None, None, False

def construct():
    start_time = time.time()
    DEADLINE = 145  # safety margin before 150s kill
    
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
    # Try ranks from lower bound up to schoolbook
    best_rank = rank_schoolbook
    best_U, best_V, best_W = U_school, V_school, W_school
    
    # Try a range of ranks
    for rank in range(20, rank_schoolbook + 1, 2):
        if time.time() - start_time > DEADLINE - 10:
            break
            
        time_budget = min(15.0, DEADLINE - (time.time() - start_time) - 5)
        if time_budget < 2:
            break
            
        U, V, W, success = _try_ranked_decomposition(n, m, p, rank, time_budget, seed=rank)
        if success:
            best_rank = rank
            best_U, best_V, best_W = U, V, W
            break  # Found a valid lower rank, stop
    
    # Return the best decomposition found
    # Convert to proper format
    result_U = [[int(round(v)) if abs(v - round(v)) < 1e-9 else v for v in row] for row in best_U]
    result_V = [[int(round(v)) if abs(v - round(v)) < 1e-9 else v for v in row] for row in best_V]
    result_W = [[int(round(v)) if abs(v - round(v)) < 1e-9 else v for v in row] for row in best_W]
    
    return {"U": result_U, "V": result_V, "W": result_W}
# EVOLVE-BLOCK-END
