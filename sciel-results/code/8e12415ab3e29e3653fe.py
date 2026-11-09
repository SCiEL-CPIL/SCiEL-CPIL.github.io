"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import least_squares
import random
from fractions import Fraction

DIMENSIONS = [3, 3, 5]
SEED_TIME_BUDGET = 120.0  # seconds

def _verify_exact(U, V, W, n, m, p):
    """Check if decomposition is exact using rational arithmetic."""
    rank = len(U[0]) if isinstance(U, list) else U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0, 1)
                for c in range(rank):
                    u_val = U[i*m+j][c] if isinstance(U, list) else U[i*m+j, c]
                    v_val = V[j*p+k][c] if isinstance(V, list) else V[j*p+k, c]
                    w_val = W[k*n+i][c] if isinstance(W, list) else W[k*n+i, c]
                    if isinstance(u_val, Fraction):
                        uf = u_val
                    else:
                        uf = Fraction(u_val).limit_denominator(10000)
                    if isinstance(v_val, Fraction):
                        vf = v_val
                    else:
                        vf = Fraction(v_val).limit_denominator(10000)
                    if isinstance(w_val, Fraction):
                        wf = w_val
                    else:
                        wf = Fraction(w_val).limit_denominator(10000)
                    s += uf * vf * wf
                if s != Fraction(1, 1):
                    return False
    return True

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition."""
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

def _optimize_block(n, m, p, target_rank, seed=42, max_nfev=2000):
    """
    Try to find a rank-target_rank decomposition for n x m x p multiplication.
    Returns (U, V, W) as numpy arrays or None.
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # Initialize with small random values
    U_init = np.random.randn(n*m, target_rank) * 0.5
    V_init = np.random.randn(m*p, target_rank) * 0.5
    W_init = np.random.randn(p*n, target_rank) * 0.5
    
    def residuals(params):
        U_flat = params[:n*m*target_rank].reshape(n*m, target_rank)
        V_flat = params[n*m*target_rank:n*m*target_rank+m*p*target_rank].reshape(m*p, target_rank)
        W_flat = params[n*m*target_rank+m*p*target_rank:].reshape(p*n, target_rank)
        
        res = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(target_rank):
                        s += U_flat[i*m+j, c] * V_flat[j*p+k, c] * W_flat[k*n+i, c]
                    res.append(s - 1.0)
        return res
    
    x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
    
    try:
        result = least_squares(residuals, x0, method='lm', max_nfev=max_nfev)
        if result.success and np.max(np.abs(result.fun)) < 1e-6:
            U = result.x[:n*m*target_rank].reshape(n*m, target_rank)
            V = result.x[n*m*target_rank:n*m*target_rank+m*p*target_rank].reshape(m*p, target_rank)
            W = result.x[n*m*target_rank+m*p*target_rank:].reshape(p*n, target_rank)
            return U, V, W
    except Exception:
        pass
    return None

def _snap_and_verify(U, V, W, n, m, p):
    """
    Snap float matrices to half-integers and verify exactness.
    Returns (U_list, V_list, W_list) if successful, else None.
    """
    U_s = np.round(U * 2) / 2.0
    V_s = np.round(V * 2) / 2.0
    W_s = np.round(W * 2) / 2.0
    
    # Convert to list of lists of Fractions for exact verification
    U_frac = [[Fraction(x).limit_denominator(10) for x in row] for row in U_s]
    V_frac = [[Fraction(x).limit_denominator(10) for x in row] for row in V_s]
    W_frac = [[Fraction(x).limit_denominator(10) for x in row] for row in W_s]
    
    if _verify_exact(U_frac, V_frac, W_frac, n, m, p):
        # Convert back to simple numbers for output
        U_out = [[float(x) for x in row] for row in U_s]
        V_out = [[float(x) for x in row] for row in V_s]
        W_out = [[float(x) for x in row] for row in W_s]
        return U_out, V_out, W_out
    return None

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + SEED_TIME_BUDGET - 10.0
    
    best_result = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # We decompose 3x3x5 into 3x3x3 and 3x3x2
    # Block 1: 3x3x3
    # Block 2: 3x3x2
    
    U1_best, V1_best, W1_best = None, None, None
    rank1_best = 27
    
    U2_best, V2_best, W2_best = None, None, None
    rank2_best = 18
    
    # Try to optimize Block 1 (3x3x3)
    for rank in range(20, 28):
        if time.time() > deadline:
            break
        for seed in range(10):
            if time.time() > deadline:
                break
            sol = _optimize_block(3, 3, 3, rank, seed=seed)
            if sol is not None:
                snapped = _snap_and_verify(sol[0], sol[1], sol[2], 3, 3, 3)
                if snapped is not None:
                    U1_best, V1_best, W1_best = snapped
                    rank1_best = rank
                    break
    
    # Try to optimize Block 2 (3x3x2)
    for rank in range(12, 19):
        if time.time() > deadline:
            break
        for seed in range(10):
            if time.time() > deadline:
                break
            sol = _optimize_block(3, 3, 2, rank, seed=seed)
            if sol is not None:
                snapped = _snap_and_verify(sol[0], sol[1], sol[2], 3, 3, 2)
                if snapped is not None:
                    U2_best, V2_best, W2_best = snapped
                    rank2_best = rank
                    break
    
    # Combine if we found solutions for both blocks
    if U1_best is not None and U2_best is not None:
        total_rank = rank1_best + rank2_best
        if total_rank < best_rank:
            # Construct combined U, V, W
            # U: (9, rank1) and (9, rank2) -> (9, total_rank)
            U_comb = np.hstack([np.array(U1_best), np.array(U2_best)])
            
            # V: (9, rank1) for k=0,1,2 and (6, rank2) for k=3,4
            V_comb = np.zeros((15, total_rank))
            # Fill V_comb for k=0,1,2
            for j in range(3):
                for k in range(3):
                    local_idx = j * 3 + k
                    global_idx = j * 5 + k
                    V_comb[global_idx, :rank1_best] = np.array(V1_best)[local_idx, :]
            
            # Fill V_comb for k=3,4
            for j in range(3):
                for k_loc in range(2):
                    local_idx = j * 2 + k_loc
                    global_idx = j * 5 + (k_loc + 3)
                    V_comb[global_idx, rank1_best:] = np.array(V2_best)[local_idx, :]
            
            # W: (9, rank1) for k=0,1,2 and (6, rank2) for k=3,4
            W_comb = np.zeros((15, total_rank))
            # Fill W_comb for k=0,1,2
            for k in range(3):
                for i in range(3):
                    local_idx = k * 3 + i
                    global_idx = k * 3 + i
                    W_comb[global_idx, :rank1_best] = np.array(W1_best)[local_idx, :]
            
            # Fill W_comb for k=3,4
            for k_loc in range(2):
                for i in range(3):
                    local_idx = k_loc * 3 + i
                    global_idx = (k_loc + 3) * 3 + i
                    W_comb[global_idx, rank1_best:] = np.array(W2_best)[local_idx, :]
            
            # Verify combined solution
            U_list = [[float(x) for x in row] for row in U_comb]
            V_list = [[float(x) for x in row] for row in V_comb]
            W_list = [[float(x) for x in row] for row in W_comb]
            
            if _verify_exact(U_list, V_list, W_list, 3, 3, 5):
                best_result = {"U": U_list, "V": V_list, "W": W_list}
                best_rank = total_rank
    
    return best_result
# EVOLVE-BLOCK-END
