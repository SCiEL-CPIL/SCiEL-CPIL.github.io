"""Fresh program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

n, m, p = DIMENSIONS
NM, MP, PN = n * m, m * p, p * n

# Precompute the support of the target tensor
SUPPORT = [(i * m + j, j * p + k, k * n + i) for i in range(n) for j in range(m) for k in range(p)]

def build_target():
    """Build the target multiplication tensor T of shape (NM, MP, PN)."""
    T = np.zeros((NM, MP, PN), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T

def compute_error_fast(U, V, W):
    """Compute error only on the support of the target tensor."""
    err = 0.0
    for (a, b, c) in SUPPORT:
        val = 0.0
        for r in range(U.shape[1]):
            val += U[a, r] * V[b, r] * W[c, r]
        err += (val - 1.0) ** 2
    return err

def compute_error_full(U, V, W, T):
    """Compute Frobenius norm of error between target and reconstructed tensor."""
    T_rec = np.einsum('ar,br,cr->abc', U, V, W)
    return np.linalg.norm(T - T_rec)

def try_find_decomposition(R, T, time_budget, rng):
    """Try to find a rank-R decomposition via local search."""
    best_U, best_V, best_W = None, None, None
    best_err = np.inf
    
    n_restarts = 0
    while time.time() < time_budget:
        n_restarts += 1
        
        # Initialize: start from trivial solution and randomly perturb/merge
        if n_restarts == 1:
            # Start from a structured initialization
            U = np.zeros((NM, R), dtype=np.float64)
            V = np.zeros((MP, R), dtype=np.float64)
            W = np.zeros((PN, R), dtype=np.float64)
            
            # For each (i,j,k), assign a rank slot
            slot = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        if slot < R:
                            U[i * m + j, slot] = 1.0
                            V[j * p + k, slot] = 1.0
                            W[k * n + i, slot] = 1.0
                            slot += 1
            # If R < 36, we need to merge some terms
            # For now, just use the first R terms
            if R < 36:
                # Randomly set some entries to allow merging
                for r in range(R):
                    for _ in range(3):
                        a = rng.integers(0, NM)
                        b = rng.integers(0, MP)
                        c = rng.integers(0, PN)
                        U[a, r] = rng.choice([-1, 1])
                        V[b, r] = rng.choice([-1, 1])
                        W[c, r] = rng.choice([-1, 1])
        else:
            # Random initialization with small integers
            U = rng.integers(-1, 2, size=(NM, R)).astype(np.float64)
            V = rng.integers(-1, 2, size=(MP, R)).astype(np.float64)
            W = rng.integers(-1, 2, size=(PN, R)).astype(np.float64)
        
        err = compute_error_fast(U, V, W)
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            if err < 1e-9:
                break
        
        # Local search: coordinate descent with integer steps
        for iteration in range(500):
            if time.time() > time_budget:
                break
            improved = False
            
            # Try adjusting each entry of U, V, W by ±1
            for mat_name, mat in [('U', U), ('V', V), ('W', W)]:
                for idx in range(mat.size):
                    row, col = np.unravel_index(idx, mat.shape)
                    old_val = mat[row, col]
                    for new_val in [old_val - 1, old_val + 1]:
                        if new_val == old_val:
                            continue
                        mat[row, col] = new_val
                        new_err = compute_error_fast(U, V, W)
                        if new_err < err - 1e-12:
                            err = new_err
                            improved = True
                            if err < 1e-9:
                                break
                        else:
                            mat[row, col] = old_val
                    if err < 1e-9:
                        break
                if err < 1e-9:
                    break
            
            if err < 1e-9:
                break
            if not improved and iteration > 100:
                break
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            if best_err < 1e-9:
                break
    
    if best_err < 1e-9:
        # Convert to integers
        U_int = np.round(best_U).astype(int)
        V_int = np.round(best_V).astype(int)
        W_int = np.round(best_W).astype(int)
        # Verify
        T_rec = np.einsum('ar,br,cr->abc', U_int, V_int, W_int)
        if np.allclose(T_rec, T, atol=1e-9):
            return U_int.tolist(), V_int.tolist(), W_int.tolist()
    
    return None

def construct():
    start_time = time.time()
    time_limit = 145  # Keep under 150s
    
    T = build_target()
    
    # Try ranks from 36 downward first (we know 36 works), then try lower
    # Actually, let's try from a reasonable lower bound upward
    # Known lower bound: max(n*m, m*p, p*n) = max(6, 18, 12) = 18
    # But we need at least the number of independent "directions"
    
    # Try ranks: start with 36 (known to work), then try lower
    ranks_to_try = [36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18]
    
    for R in ranks_to_try:
        remaining = time_limit - (time.time() - start_time)
        if remaining < 10:
            break
        rng = np.random.default_rng(R * 1000 + 42)
        result = try_find_decomposition(R, T, start_time + remaining - 5, rng)
        if result is not None:
            U_list, V_list, W_list = result
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: schoolbook rank 36
    rank = n * m * p
    U = [[0] * rank for _ in range(NM)]
    V = [[0] * rank for _ in range(MP)]
    W = [[0] * rank for _ in range(PN)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
