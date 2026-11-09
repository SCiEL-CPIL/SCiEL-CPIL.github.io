"""Matrix multiplication 2x4 x 4x6 - rank-minimal tensor decomposition search."""
import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [2, 4, 6]

def _schoolbook():
    n, m, p = DIMENSIONS
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

def _verify(U, V, W):
    """Check that U,V,W exactly decompose the matrix multiplication tensor."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if s != 1:
                    return False
    return True

def _verify_np(U, V, W):
    """Vectorized verification using numpy."""
    n, m, p = DIMENSIONS
    # Build index arrays
    idx_U = []
    idx_V = []
    idx_W = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_U.append(i*m+j)
                idx_V.append(j*p+k)
                idx_W.append(k*n+i)
    idx_U = np.array(idx_U)
    idx_V = np.array(idx_V)
    idx_W = np.array(idx_W)
    # Compute all products
    U_sel = U[idx_U, :]  # (nmp, rank)
    V_sel = V[idx_V, :]  # (nmp, rank)
    W_sel = W[idx_W, :]  # (nmp, rank)
    # Sum over rank dimension
    result = np.sum(U_sel * V_sel * W_sel, axis=1)
    return np.all(result == 1)

def _try_random_rank(r, max_attempts=500, seed=42, val_range=(-1,0,1)):
    """Try to find a rank-r decomposition via random search."""
    n, m, p = DIMENSIONS
    rng = random.Random(seed)
    vals = list(range(val_range[0], val_range[1]+1))
    
    for attempt in range(max_attempts):
        U = np.array([[rng.choice(vals) for _ in range(r)] for _ in range(n*m)], dtype=int)
        V = np.array([[rng.choice(vals) for _ in range(r)] for _ in range(m*p)], dtype=int)
        W = np.array([[rng.choice(vals) for _ in range(r)] for _ in range(p*n)], dtype=int)
        
        if _verify_np(U, V, W):
            return U.tolist(), V.tolist(), W.tolist()
    return None

def _structured_search(r, deadline):
    """Structured search: fix U and V patterns, solve for W."""
    n, m, p = DIMENSIONS
    # The constraint is: for all (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    # Group by (i,k): for fixed i,k, the constraints over j are:
    #   sum_j [sum_c U[i*m+j,c]*V[j*p+k,c]] * W[k*n+i,c] ... no, that's not right.
    # Actually: for each (i,j,k): sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    # For fixed i,k: we have m equations (one per j) in n*p... no, W has p*n entries.
    # W[k*n+i,c] is the same for all j when i,k are fixed.
    # So for fixed (i,k): sum_j sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = m? No.
    # Each equation is separate: for each j, sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    # These are m linear equations in the r variables W[k*n+i,0..r-1].
    
    # Strategy: randomly assign U and V, then for each (i,k) solve the linear system for W[k*n+i,:].
    rng = random.Random(12345)
    
    for attempt in range(200):
        if time.time() > deadline:
            break
        # Random U, V with entries in {-1, 0, 1}
        U = np.zeros((n*m, r), dtype=int)
        V = np.zeros((m*p, r), dtype=int)
        for _ in range(n*m * r // 3):
            U[rng.randrange(n*m), rng.randrange(r)] = rng.choice([-1, 1])
        for _ in range(m*p * r // 3):
            V[rng.randrange(m*p), rng.randrange(r)] = rng.choice([-1, 1])
        
        W = np.zeros((p*n, r), dtype=int)
        feasible = True
        for i in range(n):
            for k in range(p):
                # Build system: for each j in range(m): sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
                A_sys = np.zeros((m, r), dtype=float)
                for j in range(m):
                    A_sys[j, :] = U[i*m+j, :] * V[j*p+k, :]
                # Solve A_sys @ w_vec = ones(m)
                try:
                    sol, residuals, rank_sys, sv = np.linalg.lstsq(A_sys, np.ones(m), rcond=None)
                except:
                    feasible = False
                    break
                # Check if solution is integer (within tolerance)
                sol_rounded = np.round(sol)
                if np.max(np.abs(sol - sol_rounded)) > 1e-6:
                    # Try exact integer check
                    if not np.all(np.abs(A_sys @ sol_rounded - 1) < 1e-6):
                        feasible = False
                        break
                W[k*n+i, :] = sol_rounded
        if feasible:
            # Final verification
            if _verify_np(U, V, W):
                return U.tolist(), V.tolist(), W.tolist()
    return None

def _try_small_rank(r, deadline):
    """Try rank r with various strategies before deadline."""
    n, m, p = DIMENSIONS
    
    # Strategy 1: random with {-1,0,1}
    for seed in range(100):
        if time.time() > deadline:
            break
        result = _try_random_rank(r, max_attempts=50, seed=seed, val_range=(-1,1))
        if result:
            return result
    
    # Strategy 2: structured search
    if time.time() < deadline - 2.0:
        result = _structured_search(r, deadline - 1.0)
        if result:
            return result
    
    # Strategy 3: random with {-2,-1,0,1,2}
    for seed in range(50):
        if time.time() > deadline:
            break
        result = _try_random_rank(r, max_attempts=30, seed=seed+1000, val_range=(-2,2))
        if result:
            return result
    
    return None

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 48
    
    # Known lower bound: max(n*m, m*p, p*n) = max(8, 24, 12) = 24
    # Try ranks from lower bound up to schoolbook
    for r in range(24, schoolbook_rank):
        if time.time() > deadline - 5.0:
            break
        result = _try_small_rank(r, deadline - 2.0)
        if result:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
    
    # Fallback: schoolbook
    return _schoolbook()
# EVOLVE-BLOCK-END
