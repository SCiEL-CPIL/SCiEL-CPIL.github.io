"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def _build_true_tensor(n, m, p):
    """Build the true matrix multiplication tensor."""
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0
    return T

def _verify_exact(U, V, W, n, m, p):
    """Verify exact decomposition with tolerance for floating point."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    T_true = _build_true_tensor(n, m, p)
    T_recon = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
    
    # Check exactness with tight tolerance
    if not np.allclose(T_recon, T_true, atol=1e-8):
        return False
    return True

def _round_to_half_integer(arr):
    """Round array to nearest half-integer."""
    return np.round(arr * 2) / 2.0

def _try_als(r, n, m, p, T_true, max_iters=200, seed=42, time_budget=30):
    """Try to find rank-r decomposition using Alternating Least Squares."""
    start_time = time.time()
    nm, mp, pn = n*m, m*p, p*n
    
    rng = np.random.RandomState(seed)
    
    # Initialize
    U = rng.randn(nm, r) * 0.1
    V = rng.randn(mp, r) * 0.1
    W = rng.randn(pn, r) * 0.1
    
    best_err = float('inf')
    best_solution = None
    
    for it in range(max_iters):
        if time.time() - start_time > time_budget:
            break
        
        # Solve for U given V, W
        # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        # For each r: T[:, :, r] ≈ U[:, r] outer V[:, r] outer W[:, r]
        # This is a bilinear problem. We can use ALS by fixing two and solving for the third.
        
        # Actually, for tensor decomposition T = sum_r U_r ⊗ V_r ⊗ W_r,
        # we can use the fact that for fixed V, W:
        # T = sum_r U_r * (V_r ⊗ W_r)
        # This is a linear system in U for each r... but it's not quite right.
        
        # Better approach: use gradient-based method with better initialization
        # Or use the ALS formulation properly.
        
        # Let's use a simple iterative refinement:
        # Compute residual
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        R = T_true - T_recon
        err = np.max(np.abs(R))
        
        if err < best_err:
            best_err = err
            best_solution = (U.copy(), V.copy(), W.copy())
        
        if err < 1e-6:
            # Try rounding
            U_r = _round_to_half_integer(U)
            V_r = _round_to_half_integer(V)
            W_r = _round_to_half_integer(W)
            T_recon_r = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
            err_r = np.max(np.abs(T_recon_r - T_true))
            if err_r < 1e-8:
                return (U_r, V_r, W_r)
        
        # Gradient step with adaptive learning rate
        eta = 0.05 / (1 + it * 0.005)
        
        grad_U = np.einsum('abc,br,cr->ar', R, V, W)
        grad_V = np.einsum('abc,ar,cr->br', R, U, W)
        grad_W = np.einsum('abc,ar,br->cr', R, U, V)
        
        U += eta * grad_U
        V += eta * grad_V
        W += eta * grad_W
    
    return best_solution

def _try_rank_search(r, n, m, p, T_true, max_iters=5000, seed=42, time_budget=60):
    """Try to find rank-r decomposition using multiple strategies."""
    start_time = time.time()
    nm, mp, pn = n*m, m*p, p*n
    
    best_solution = None
    best_err = float('inf')
    
    # Strategy 1: ALS-like approach with multiple initializations
    for init_seed in range(8):
        if time.time() - start_time > time_budget:
            break
        
        result = _try_als(r, n, m, p, T_true, max_iters=150, seed=seed + init_seed, 
                          time_budget=min(15, time_budget - (time.time() - start_time)))
        
        if result is not None:
            U, V, W = result
            T_recon = np.einsum('ar,br,cr->abc', U, V, W)
            err = np.max(np.abs(T_recon - T_true))
            
            if err < best_err:
                best_err = err
                best_solution = result
            
            if err < 1e-8:
                return result
    
    # Strategy 2: Direct gradient descent with careful initialization
    for init_seed in range(5):
        if time.time() - start_time > time_budget:
            break
        
        rng = np.random.RandomState(seed + 100 + init_seed)
        
        # Better initialization: use partial schoolbook structure
        U = np.zeros((nm, r))
        V = np.zeros((mp, r))
        W = np.zeros((pn, r))
        
        # Initialize first min(r, n*m*p) columns with schoolbook pattern
        min_cols = min(r, n * m * p)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if col >= r:
                        break
                    U[i*m + j, col] = 1.0
                    V[j*p + k, col] = 1.0
                    W[k*n + i, col] = 1.0
                    col += 1
        
        # Add small random perturbation to remaining columns
        if r > min_cols:
            U[:, min_cols:] = rng.randn(nm, r - min_cols) * 0.1
            V[:, min_cols:] = rng.randn(mp, r - min_cols) * 0.1
            W[:, min_cols:] = rng.randn(pn, r - min_cols) * 0.1
        
        for it in range(3000):
            if time.time() - start_time > time_budget:
                break
            
            T_recon = np.einsum('ar,br,cr->abc', U, V, W)
            R = T_true - T_recon
            err = np.max(np.abs(R))
            
            if err < best_err:
                best_err = err
                best_solution = (U.copy(), V.copy(), W.copy())
            
            if err < 1e-8:
                # Try rounding
                U_r = _round_to_half_integer(U)
                V_r = _round_to_half_integer(V)
                W_r = _round_to_half_integer(W)
                T_recon_r = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
                err_r = np.max(np.abs(T_recon_r - T_true))
                if err_r < 1e-8:
                    return (U_r, V_r, W_r)
                break
            
            # Adaptive learning rate
            eta = 0.02 / (1 + it * 0.001)
            
            grad_U = np.einsum('abc,br,cr->ar', R, V, W)
            grad_V = np.einsum('abc,ar,cr->br', R, U, W)
            grad_W = np.einsum('abc,ar,br->cr', R, U, V)
            
            U += eta * grad_U
            V += eta * grad_V
            W += eta * grad_W
    
    return best_solution

def _schoolbook(n, m, p):
    """Return schoolbook decomposition of rank n*m*p."""
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

def construct():
    """
    Constructs the rank-minimal tensor decomposition for 
    multiplying a 3x4 matrix by a 4x8 matrix.
    """
    n, m, p = DIMENSIONS
    T_true = _build_true_tensor(n, m, p)
    
    # Start with schoolbook (rank 96)
    best_result = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Deadline: must finish before 145s to be safe
    deadline = time.time() + 145
    
    # Try a range of ranks from high to low
    # Theoretical lower bound is max(n*m, m*p, p*n) = max(12, 32, 24) = 32
    # Practical bounds are higher; try 48, 44, 40, 36, 32
    for r in [48, 44, 40, 36, 32]:
        if time.time() > deadline - 10:  # Safety margin
            break
        
        print(f"Trying rank {r}...")
        time_budget = min(25, deadline - time.time() - 5)
        result = _try_rank_search(r, n, m, p, T_true, max_iters=5000, seed=42, time_budget=time_budget)
        
        if result is not None:
            U, V, W = result
            # Verify
            if _verify_exact(U, V, W, n, m, p):
                best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                best_rank = r
                print(f"Found rank {r} decomposition!")
                break
    
    return best_result
# EVOLVE-BLOCK-END
