"""Matrix multiplication 3,4,8 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [3, 4, 8]
import time
_START_TIME = time.time()
_TIME_LIMIT = 115.0  # safety margin before 120s

# EVOLVE-BLOCK-START
import random
import math
import numpy as np
from itertools import product

def _build_schoolbook():
    """Standard schoolbook matrix multiplication tensor decomposition of rank n*m*p."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}

def _validate_decomposition(result, ntest=3, seed=42):
    """Validate that the decomposition computes exact matrix multiplication."""
    n, m, p = DIMENSIONS
    U, V, W = result['U'], result['V'], result['W']
    if not U or not V or not W:
        return False
    if len(U[0]) != len(V[0]) or len(U[0]) != len(W[0]):
        return False
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    random.seed(seed)
    for t in range(ntest):
        A = [[random.uniform(-3, 3) for _ in range(m)] for _ in range(n)]
        B = [[random.uniform(-3, 3) for _ in range(p)] for _ in range(m)]
        Ctrue = [[sum(A[i][jj] * B[jj][kk] for jj in range(m)) for kk in range(p)]
                 for i in range(n)]
        Ccomp = [[0.0 for _ in range(p)] for _ in range(n)]
        for i in range(n):
            for k in range(p):
                acc = 0.0
                for r in range(len(U[0])):
                    uvprod = 0.0
                    for jj in range(m):
                        uvprod += U[i*m+jj][r] * V[jj*p+k][r]
                    acc += uvprod * W[k*n+i][r]
                Ccomp[i][k] = acc
        for i in range(n):
            for k in range(p):
                if abs(Ccomp[i][k] - Ctrue[i][k]) > 1e-9 * max(1, abs(Ctrue[i][k])):
                    return False
    return True

def _validate_exact(result):
    """Exact validation using symbolic-like check with many random points."""
    n, m, p = DIMENSIONS
    U, V, W = result['U'], result['V'], result['W']
    if not U or not V or not W:
        return False
    if len(U[0]) != len(V[0]) or len(U[0]) != len(W[0]):
        return False
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    rng = random.Random(12345)
    for t in range(5):
        # Use small integer values for exactness
        A = [[rng.randint(-2, 2) for _ in range(m)] for _ in range(n)]
        B = [[rng.randint(-2, 2) for _ in range(p)] for _ in range(m)]
        Ctrue = [[sum(A[i][jj] * B[jj][kk] for jj in range(m)) for kk in range(p)]
                 for i in range(n)]
        Ccomp = [[0.0 for _ in range(p)] for _ in range(n)]
        for i in range(n):
            for k in range(p):
                acc = 0.0
                for r in range(len(U[0])):
                    uvprod = 0.0
                    for jj in range(m):
                        uvprod += U[i*m+jj][r] * V[jj*p+k][r]
                    acc += uvprod * W[k*n+i][r]
                Ccomp[i][k] = acc
        for i in range(n):
            for k in range(p):
                if abs(Ccomp[i][k] - Ctrue[i][k]) > 1e-10:
                    return False
    return True

def _try_block_decomposition():
    """Try block decomposition: split p into blocks and use known smaller tensors."""
    n, m, p = DIMENSIONS
    
    # Try splitting p=8 into blocks
    # Block sizes: [4,4], [2,2,2,2], [3,3,2], etc.
    best = None
    best_rank = n * m * p  # schoolbook
    
    # For each block size, use schoolbook for that block
    # Total rank = sum of block ranks
    # This won't beat schoolbook unless we have better sub-tensors
    
    # Try to find better sub-tensors via search
    # For now, just return schoolbook as baseline
    return None

def _random_local_search(rank, seed, max_iters=500):
    """Random initialization with local hill-climbing for exact tensor decomposition."""
    n, m, p = DIMENSIONS
    rng = random.Random(seed)
    
    # Initialize with sparse 0/±1 entries
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    for i in range(n * m):
        for c in range(rank):
            if rng.random() < 0.4:
                U[i][c] = rng.choice([-1.0, 1.0])
    for i in range(m * p):
        for c in range(rank):
            if rng.random() < 0.4:
                V[i][c] = rng.choice([-1.0, 1.0])
    for i in range(p * n):
        for c in range(rank):
            if rng.random() < 0.4:
                W[i][c] = rng.choice([-1.0, 1.0])
    
    # Compute residual
    def _compute_residual():
        """Compute how well the decomposition matches the target tensor."""
        # Target: T[i,j,k] = 1 if (i,j,k) is a valid triple, else 0
        # Computed: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # We want this to equal 1 for all (i,j,k) in range
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
        return max_err
    
    current_err = _compute_residual()
    if current_err < 1e-9:
        return {"U": U, "V": V, "W": W}
    
    # Local search: try flipping single entries
    for iteration in range(max_iters):
        if time.time() - _START_TIME > _TIME_LIMIT - 5:
            break
        
        improved = False
        # Try random single-entry modifications
        for _ in range(20):
            # Choose which matrix to modify
            mat_idx = rng.choice(['U', 'V', 'W'])
            if mat_idx == 'U':
                row = rng.randint(0, n*m-1)
                col = rng.randint(0, rank-1)
                old_val = U[row][col]
                new_val = old_val + rng.choice([-1.0, 1.0])
                U[row][col] = new_val
            elif mat_idx == 'V':
                row = rng.randint(0, m*p-1)
                col = rng.randint(0, rank-1)
                old_val = V[row][col]
                new_val = old_val + rng.choice([-1.0, 1.0])
                V[row][col] = new_val
            else:
                row = rng.randint(0, p*n-1)
                col = rng.randint(0, rank-1)
                old_val = W[row][col]
                new_val = old_val + rng.choice([-1.0, 1.0])
                W[row][col] = new_val
            
            new_err = _compute_residual()
            if new_err < current_err - 1e-12:
                current_err = new_err
                improved = True
            else:
                # Revert
                if mat_idx == 'U':
                    U[row][col] = old_val
                elif mat_idx == 'V':
                    V[row][col] = old_val
                else:
                    W[row][col] = old_val
        
        if current_err < 1e-9:
            return {"U": U, "V": V, "W": W}
        if not improved and iteration > 100:
            break
    
    if current_err < 1e-9:
        return {"U": U, "V": V, "W": W}
    return None

def construct():
    n, m, p = DIMENSIONS
    elapsed = time.time() - _START_TIME
    budget = _TIME_LIMIT - elapsed
    if budget <= 10:
        return _build_schoolbook()
    
    best = _build_schoolbook()
    best_rank = len(best['U'][0])
    
    # Try various ranks from lower bound up
    # Lower bound: max(n*m, m*p, p*n) = max(12, 32, 24) = 32
    # But we know schoolbook is 96, so try from 32 upward
    
    # First, try structured approaches
    # For 3x4x8, try splitting into 4 blocks of 3x4x2
    # If R(3,4,2) <= 12, total rank <= 48
    
    # Try random local search for various ranks
    rng = random.Random(int(time.time() * 1000))
    
    # Try ranks from a reasonable lower bound
    # Theoretical lower bound is max(n*m, m*p, p*n) = 32
    # But practical lower bound might be higher
    # Try a few ranks: 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92
    
    ranks_to_try = [40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92]
    
    for target_rank in ranks_to_try:
        if time.time() - _START_TIME > _TIME_LIMIT - 10:
            break
        if target_rank >= best_rank:
            continue
        
        # Try a few seeds for this rank
        for seed_offset in range(3):
            if time.time() - _START_TIME > _TIME_LIMIT - 10:
                break
            
            seed = rng.randint(0, 10**9) + seed_offset
            result = _random_local_search(target_rank, seed, max_iters=200)
            
            if result is not None:
                if _validate_decomposition(result, ntest=2, seed=seed % 100000):
                    if _validate_exact(result):
                        best = result
                        best_rank = target_rank
                        break  # Found valid decomposition for this rank, try lower
        
        if best_rank <= 48:
            break
    
    return {
        "U": [[float(x) for x in row] for row in best['U']],
        "V": [[float(x) for x in row] for row in best['V']],
        "W": [[float(x) for x in row] for row in best['W']]
    }
# EVOLVE-BLOCK-END
