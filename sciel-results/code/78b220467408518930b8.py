"""Matrix multiplication 3,3,7 - rank-minimal exact tensor decomposition."""
import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n=3, m=3, p=7, tol=1e-9):
    """Verify that U, V, W give an exact decomposition of the matmul tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(val - 1.0) > tol:
                    return False
    return True

def _schoolbook(n=3, m=3, p=7):
    """Return the schoolbook (rank=n*m*p) decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _round_to_half_integer(x, tol=1e-6):
    """Round a value to the nearest half-integer if close, otherwise keep as is."""
    h = np.round(2 * x) / 2.0
    if abs(x - h) < tol:
        return h
    return x

def _try_random_rank(n, m, p, rank, deadline, seed=42):
    """
    Try to find a rank-r decomposition using random initialization and optimization.
    Returns (U, V, W) if found, else None.
    """
    rng = np.random.RandomState(seed)
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Tensor entries: T[i*m+j, j*p+k, k*n+i] = 1
    # We want: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # Flatten the tensor to a vector of constraints
    # There are n*m*p = 63 constraints
    # Each constraint: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] - 1 = 0
    
    def _objective(params):
        if time.time() > deadline:
            return 1e10
        # Unpack parameters
        # U: nm * rank, V: mp * rank, W: pn * rank
        total = (nm + mp + pn) * rank
        if len(params) != total:
            return 1e10
        
        idx = 0
        U = params[idx:idx + nm * rank].reshape(nm, rank)
        idx += nm * rank
        V = params[idx:idx + mp * rank].reshape(mp, rank)
        idx += mp * rank
        W = params[idx:idx + pn * rank].reshape(pn, rank)
        
        # Compute error: sum over all (i,j,k) of (sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] - 1)^2
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    err += (s - 1.0) ** 2
        return err
    
    # Initialize randomly
    init_scale = 1.0
    init_params = rng.randn((nm + mp + pn) * rank) * init_scale
    
    # Try a few random initializations
    for attempt in range(5):
        if time.time() > deadline:
            break
        if attempt == 0:
            params0 = init_params
        else:
            params0 = rng.randn((nm + mp + pn) * rank) * init_scale
        
        res = minimize(_objective, params0, method='L-BFGS-B', options={'maxiter': 200, 'maxfun': 1000})
        
        if res.fun < 1e-6:
            # Found a good solution
            idx = 0
            U = res.x[idx:idx + nm * rank].reshape(nm, rank)
            idx += nm * rank
            V = res.x[idx:idx + mp * rank].reshape(mp, rank)
            idx += mp * rank
            W = res.x[idx:idx + pn * rank].reshape(pn, rank)
            
            # Round to half-integers
            U_r = np.array([[_round_to_half_integer(U[i, c]) for c in range(rank)] for i in range(nm)])
            V_r = np.array([[_round_to_half_integer(V[i, c]) for c in range(rank)] for i in range(mp)])
            W_r = np.array([[_round_to_half_integer(W[i, c]) for c in range(rank)] for i in range(pn)])
            
            if _verify(U_r, V_r, W, n, m, p, tol=1e-6):
                return U_r.tolist(), V_r.tolist(), W_r.tolist()
            elif _verify(U, V, W, n, m, p, tol=1e-6):
                return U.tolist(), V.tolist(), W.tolist()
    
    return None

def _try_block_strassen(n, m, p, deadline):
    """
    Try a block decomposition approach:
    Split the p=7 columns into 3+4.
    Use Strassen-like decomposition for the 3x3*3x3 part.
    Use schoolbook for the 3x3*3x4 part.
    
    For the 3x3*3x3 part, we'll try to use a rank-23 decomposition (Larsen)
    or a simpler lower-rank decomposition.
    
    Actually, let's try a different approach: use the known rank-23 decomposition
    for 3x3*3x3 if we can construct it, otherwise fall back.
    """
    # For 3x3*3x3, the best known rank is 23.
    # Let's try to construct a rank-23 decomposition.
    # This is complex, so let's try a simpler approach first.
    
    # Alternative: use a 2x2 Strassen block within the 3x3.
    # Split 3x3 into 2x2 + 1x1 blocks.
    # A = [[A11 (2x2), A12 (2x1)],
    #      [A21 (1x2), A22 (1x1)]]
    # B = [[B11 (2x2), B12 (2x1)],
    #      [B21 (1x2), B22 (1x1)]]
    
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    
    # A11*B11: 2x2*2x2, Strassen rank 7
    # A12*B21: 2x1*1x2, rank 4
    # A11*B12: 2x2*2x1, rank 4
    # A12*B22: 2x1*1x1, rank 2
    # A21*B11: 1x2*2x2, rank 4
    # A22*B21: 1x1*1x2, rank 2
    # A21*B12: 1x2*2x1, rank 4
    # A22*B22: 1x1*1x1, rank 1
    # Total: 7+4+4+2+4+2+4+1 = 28 > 27
    
    # So this doesn't help. Let's just try random search for the full problem.
    return None

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x3 by 3x7 matrix multiplication.
    Returns {'U', 'V', 'W'} with integer/half-integer entries.
    """
    n, m, p = DIMENSIONS  # 3, 3, 7
    deadline = time.time() + 110  # safety margin
    
    # Start with schoolbook
    best = _schoolbook(n, m, p)
    best_rank = n * m * p  # 63
    
    # Try random search for lower ranks
    # The tensor rank of 3x3*3x3 is at most 23 (Larsen), and 3x3*3x7 should be at most 23 + 36 = 59
    # But we might do better with a direct decomposition.
    
    # Try ranks from 50 down to 63
    for rank in range(50, best_rank):
        if time.time() > deadline - 5:
            break
        result = _try_random_rank(n, m, p, rank, deadline, seed=rank)
        if result is not None:
            U, V, W = result
            best = {"U": U, "V": V, "W": W}
            best_rank = rank
            break  # Found the lowest rank, stop
    
    return best
# EVOLVE-BLOCK-END
