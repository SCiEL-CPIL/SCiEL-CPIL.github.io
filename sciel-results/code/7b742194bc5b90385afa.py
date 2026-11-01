import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 2, 9]
RNG_SEED = 42

def _build_rank32():
    n, m, p = DIMENSIONS
    rank = 32
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    def add_strassen(k0, k1):
        nonlocal col
        U[0, col] = 1; U[3, col] = 1
        V[0*p + k0, col] = 1; V[1*p + k1, col] = 1
        W[k0*n + 0, col] = 1; W[k1*n + 1, col] = 1
        col += 1
        U[2, col] = 1; U[3, col] = 1
        V[0*p + k0, col] = 1
        W[k0*n + 1, col] = 1; W[k1*n + 1, col] = -1
        col += 1
        U[0, col] = 1
        V[0*p + k1, col] = 1; V[1*p + k1, col] = -1
        W[k1*n + 0, col] = 1; W[k1*n + 1, col] = 1
        col += 1
        U[3, col] = 1
        V[1*p + k0, col] = 1; V[0*p + k0, col] = -1
        W[k0*n + 0, col] = 1; W[k0*n + 1, col] = 1
        col += 1
        U[0, col] = 1; U[1, col] = 1
        V[1*p + k1, col] = 1
        W[k0*n + 0, col] = -1; W[k1*n + 0, col] = 1
        col += 1
        U[2, col] = 1; U[0, col] = -1
        V[0*p + k0, col] = 1; V[0*p + k1, col] = 1
        W[k1*n + 1, col] = 1
        col += 1
        U[1, col] = 1; U[3, col] = -1
        V[1*p + k0, col] = 1; V[1*p + k1, col] = 1
        W[k0*n + 0, col] = 1
        col += 1

    def add_trivial(k):
        nonlocal col
        U[0, col] = 1; V[0*p + k, col] = 1; W[k*n + 0, col] = 1
        col += 1
        U[1, col] = 1; V[1*p + k, col] = 1; W[k*n + 0, col] = 1
        col += 1
        U[2, col] = 1; V[0*p + k, col] = 1; W[k*n + 1, col] = 1
        col += 1
        U[3, col] = 1; V[1*p + k, col] = 1; W[k*n + 1, col] = 1
        col += 1

    add_strassen(0, 1)
    add_strassen(2, 3)
    add_strassen(4, 5)
    add_strassen(6, 7)
    add_trivial(8)
    
    return U, V, W

def _find_rank10_223():
    """Find a rank 10 decomposition for 2x2x3 matrix multiplication."""
    n, m, p = 2, 2, 3
    rank = 10
    nm, mp, pn = n*m, m*p, p*n
    
    # Tensor entries
    T_true = np.zeros((nm, mp, pn))
    for a in range(n):
        for b in range(m):
            for c in range(p):
                i = a * m + b
                j = b * p + c
                k = c * n + a
                T_true[i, j, k] = 1.0
                
    rng = np.random.default_rng(RNG_SEED)
    
    best_residual = float('inf')
    best_solution = None
    
    # Try multiple restarts
    for restart in range(50):
        # Initialize
        U0 = rng.normal(0, 1, (nm, rank))
        V0 = rng.normal(0, 1, (mp, rank))
        W0 = rng.normal(0, 1, (pn, rank))
        
        x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
        
        def objective(params):
            U_flat = params[:nm * rank]
            V_flat = params[nm * rank : nm * rank + mp * rank]
            W_flat = params[nm * rank + mp * rank : nm * rank + mp * rank + pn * rank]
            
            U = U_flat.reshape(nm, rank)
            V = V_flat.reshape(mp, rank)
            W = W_flat.reshape(pn, rank)
            
            M = np.einsum('ir,jr,kr->ijk', U, V, W)
            residual = M - T_true
            return np.sum(residual**2)
            
        try:
            result = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
            if result.fun < 1e-6:
                U_flat = result.x[:nm * rank]
                V_flat = result.x[nm * rank : nm * rank + mp * rank]
                W_flat = result.x[nm * rank + mp * rank : nm * rank + mp * rank + pn * rank]
                U_sol = U_flat.reshape(nm, rank)
                V_sol = V_flat.reshape(mp, rank)
                W_sol = W_flat.reshape(pn, rank)
                
                # Snap to half-integers
                U_snap = np.round(U_sol * 2) / 2
                V_snap = np.round(V_sol * 2) / 2
                W_snap = np.round(W_sol * 2) / 2
                
                M_snap = np.einsum('ir,jr,kr->ijk', U_snap, V_snap, W_snap)
                res_snap = np.max(np.abs(M_snap - T_true))
                
                if res_snap < 1e-6:
                    if res_snap < best_residual:
                        best_residual = res_snap
                        best_solution = (U_snap, V_snap, W_snap)
        except Exception:
            continue
            
    return best_solution

def _build_rank30_from_223(U223, V223, W223):
    """Build rank 30 solution for 2x2x9 from rank 10 solution for 2x2x3."""
    n, m, p = 2, 2, 9
    rank = 30
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # U is shared
    # Repeat U223 three times
    U[:, 0:10] = U223
    U[:, 10:20] = U223
    U[:, 20:30] = U223
    
    # V and W need to be placed in correct blocks
    # Block 0: cols 0-2 of B, cols 0-2 of C
    # Block 1: cols 3-5 of B, cols 3-5 of C
    # Block 2: cols 6-8 of B, cols 6-8 of C
    
    # V has 18 rows (2 * 9)
    # V223 has 6 rows (2 * 3)
    
    # Block 0
    V[0:6, 0:10] = V223
    W[0:6, 0:10] = W223
    
    # Block 1
    V[6:12, 10:20] = V223
    W[6:12, 10:20] = W223
    
    # Block 2
    V[12:18, 20:30] = V223
    W[12:18, 20:30] = W223
    
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    n, m, p = DIMENSIONS
    
    # Baseline rank 32
    U32, V32, W32 = _build_rank32()
    
    def to_list(arr):
        return arr.tolist()
        
    best_U, best_V, best_W = U32, V32, W32
    best_rank = 32
    
    # Try to find rank 30
    if time.time() < deadline - 20.0:
        sol_223 = _find_rank10_223()
        if sol_223 is not None:
            U30, V30, W30 = _build_rank30_from_223(*sol_223)
            # Verify
            T_true = np.zeros((n * m, m * p, p * n))
            for a in range(n):
                for b in range(m):
                    for c in range(p):
                        i = a * m + b
                        j = b * p + c
                        k = c * n + a
                        T_true[i, j, k] = 1.0
            M = np.einsum('ir,jr,kr->ijk', U30, V30, W30)
            res = np.max(np.abs(M - T_true))
            if res < 1e-6:
                best_U, best_V, best_W = U30, V30, W30
                best_rank = 30
    
    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
