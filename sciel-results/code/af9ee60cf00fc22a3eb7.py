"""Matrix multiplication 3,4,8 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [3, 4, 8]
import time
_START_TIME = time.time()
_TIME_LIMIT = 115.0  # safety margin before 120s

# EVOLVE-BLOCK-START
import random
import numpy as np

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
    rank = len(U[0])
    if len(V[0]) != rank or len(W[0]) != rank:
        return False
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    
    # Check tensor identity: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1.0) > 1e-9:
                    return False
    
    # Also verify with random matrices for numerical stability
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
                for r in range(rank):
                    uvprod = 0.0
                    for jj in range(m):
                        uvprod += U[i*m+jj][r] * V[jj*p+k][r]
                    acc += uvprod * W[k*n+i][r]
                Ccomp[i][k] = acc
        for i in range(n):
            for k in range(p):
                if abs(Ccomp[i][k] - Ctrue[i][k]) > 1e-6 * max(1, abs(Ctrue[i][k])):
                    return False
    return True

def _try_random_search(n, m, p, rank, deadline, seed):
    """Try to find a valid decomposition of given rank using random sparse initialization and local search."""
    rng = random.Random(seed)
    Um = np.zeros((n*m, rank), dtype=float)
    Vm = np.zeros((m*p, rank), dtype=float)
    Wm = np.zeros((p*n, rank), dtype=float)
    
    # Sparse random initialization with small integer/half-integer values
    for i in range(n*m):
        for c in range(rank):
            if rng.random() < 0.3:
                Um[i, c] = rng.choice([-2.0, -1.5, -1.0, -0.5, 0.5, 1.0, 1.5, 2.0])
    for i in range(m*p):
        for c in range(rank):
            if rng.random() < 0.3:
                Vm[i, c] = rng.choice([-2.0, -1.5, -1.0, -0.5, 0.5, 1.0, 1.5, 2.0])
    for i in range(p*n):
        for c in range(rank):
            if rng.random() < 0.3:
                Wm[i, c] = rng.choice([-2.0, -1.5, -1.0, -0.5, 0.5, 1.0, 1.5, 2.0])
    
    # Local search: iteratively adjust entries to reduce residuals
    for it in range(200):
        if time.time() > deadline - 5:
            break
        
        # Find worst residual
        worst_ijk = None
        worst_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += Um[i*m+j, c] * Vm[j*p+k, c] * Wm[k*n+i, c]
                    err = abs(s - 1.0)
                    if err > worst_err:
                        worst_err = err
                        worst_ijk = (i, j, k)
        
        if worst_err < 1e-9:
            return {"U": Um.tolist(), "V": Vm.tolist(), "W": Wm.tolist()}
        
        i, j, k = worst_ijk
        # Try adjusting a random entry in U, V, or W
        mat = rng.choice(['U', 'V', 'W'])
        c = rng.randint(0, rank - 1)
        delta = rng.choice([-0.5, 0.5, -1.0, 1.0])
        
        old_val = 0.0
        if mat == 'U':
            old_val = Um[i*m+j, c]
            Um[i*m+j, c] += delta
        elif mat == 'V':
            old_val = Vm[j*p+k, c]
            Vm[j*p+k, c] += delta
        else:
            old_val = Wm[k*n+i, c]
            Wm[k*n+i, c] += delta
        
        # Check if this improved the worst residual
        new_err = 0.0
        for ii in range(n):
            for jj in range(m):
                for kk in range(p):
                    s = 0.0
                    for cc in range(rank):
                        s += Um[ii*m+jj, cc] * Vm[jj*p+kk, cc] * Wm[kk*n+ii, cc]
                    err = abs(s - 1.0)
                    if err > new_err:
                        new_err = err
        
        if new_err >= worst_err:
            # Revert
            if mat == 'U':
                Um[i*m+j, c] = old_val
            elif mat == 'V':
                Vm[j*p+k, c] = old_val
            else:
                Wm[k*n+i, c] = old_val
    
    return None

def construct():
    n, m, p = DIMENSIONS
    elapsed = time.time() - _START_TIME
    budget = _TIME_LIMIT - elapsed
    if budget <= 10:
        return _build_schoolbook()
    
    deadline = time.time() + budget - 5
    
    best = _build_schoolbook()
    best_rank = n * m * p
    
    # Try ranks from 70 up to 95 (below schoolbook 96)
    for rank in range(70, 96):
        if time.time() > deadline:
            break
        
        # Try multiple random seeds for this rank
        for seed in range(5):
            if time.time() > deadline:
                break
            
            result = _try_random_search(n, m, p, rank, deadline, seed)
            if result is not None:
                # Validate
                try:
                    if _validate_decomposition(result, ntest=2, seed=seed):
                        best = result
                        best_rank = rank
                        break  # Found valid decomposition for this rank
                except Exception:
                    pass
    
    return {
        "U": [[float(x) for x in row] for row in best['U']],
        "V": [[float(x) for x in row] for row in best['V']],
        "W": [[float(x) for x in row] for row in best['W']]
    }
# EVOLVE-BLOCK-END
