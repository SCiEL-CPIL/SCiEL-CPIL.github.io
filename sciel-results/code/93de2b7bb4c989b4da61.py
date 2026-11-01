import time
import numpy as np

DIMENSIONS = [2, 2, 3]
n, m, p = DIMENSIONS

# Precomputed indices for vectorized residual calculation
# Residual (i,j,k) corresponds to U[2i+j], V[3j+k], W[2k+i]
_U_IDX = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]
_V_IDX = [0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5]
_W_IDX = [0, 2, 4, 1, 3, 5, 2, 4, 6, 3, 5, 7]

def _residuals_rank10(params):
    """Compute residuals for rank-10 tensor decomposition (vectorized)."""
    U_flat = params[:40]
    V_flat = params[40:100]
    W_flat = params[100:160]
    
    U_mat = U_flat.reshape(4, 10)
    V_mat = V_flat.reshape(6, 10)
    W_mat = W_flat.reshape(6, 10)
    
    res = np.empty(12)
    for m in range(12):
        res[m] = np.sum(U_mat[_U_IDX[m], :] * V_mat[_V_IDX[m], :] * W_mat[_W_IDX[m], :]) - 1.0
    return res

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _rank11_decomposition():
    """
    Rank-11 decomposition: Strassen (rank 7) for 2x2 block + schoolbook (rank 4) for col 2.
    """
    n, m, p = 2, 2, 3
    rank = 11
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    def set_u(i, j, r, val):
        U[i * m + j][r] = val

    def set_v(j, k, r, val):
        V[j * p + k][r] = val

    def set_w(k, i, r, val):
        W[k * n + i][r] = val

    # Term 0: (A00+A11)(B00+B11) -> contributes to C[0,0] and C[1,1]
    set_u(0, 0, 0, 1); set_u(1, 1, 0, 1)
    set_v(0, 0, 0, 1); set_v(1, 1, 0, 1)
    set_w(0, 0, 0, 1)
    set_w(1, 1, 0, 1)

    # Term 1: (A10+A11)B00 -> contributes to C[0,1] and C[1,1]
    set_u(1, 0, 1, 1); set_u(1, 1, 1, 1)
    set_v(0, 0, 1, 1)
    set_w(0, 1, 1, 1)
    set_w(1, 1, 1, -1)

    # Term 2: A00(B01-B11) -> contributes to C[1,0] and C[1,1]
    set_u(0, 0, 2, 1)
    set_v(0, 1, 2, 1); set_v(1, 1, 2, -1)
    set_w(1, 0, 2, 1)
    set_w(1, 1, 2, 1)

    # Term 3: A11(B10-B00) -> contributes to C[0,0] and C[0,1]
    set_u(1, 1, 3, 1)
    set_v(1, 0, 3, 1); set_v(0, 0, 3, -1)
    set_w(0, 0, 3, 1)
    set_w(0, 1, 3, 1)

    # Term 4: (A00+A01)B11 -> contributes to C[0,0] and C[1,0]
    set_u(0, 0, 4, 1); set_u(0, 1, 4, 1)
    set_v(1, 1, 4, 1)
    set_w(0, 0, 4, -1)
    set_w(1, 0, 4, 1)

    # Term 5: (A10-A00)(B00+B01) -> contributes to C[1,1]
    set_u(1, 0, 5, 1); set_u(0, 0, 5, -1)
    set_v(0, 0, 5, 1); set_v(0, 1, 5, 1)
    set_w(1, 1, 5, 1)

    # Term 6: (A01-A11)(B10+B11) -> contributes to C[0,0]
    set_u(0, 1, 6, 1); set_u(1, 1, 6, -1)
    set_v(1, 0, 6, 1); set_v(1, 1, 6, 1)
    set_w(0, 0, 6, 1)

    # Term 7: A00*B02
    set_u(0, 0, 7, 1)
    set_v(0, 2, 7, 1)
    set_w(2, 0, 7, 1)

    # Term 8: A01*B12
    set_u(0, 1, 8, 1)
    set_v(1, 2, 8, 1)
    set_w(2, 0, 8, 1)

    # Term 9: A10*B02
    set_u(1, 0, 9, 1)
    set_v(0, 2, 9, 1)
    set_w(2, 1, 9, 1)

    # Term 10: A11*B12
    set_u(1, 1, 10, 1)
    set_v(1, 2, 10, 1)
    set_w(2, 1, 10, 1)

    return {"U": U, "V": V, "W": W}

def _schoolbook():
    """Schoolbook decomposition of rank n*m*p = 12."""
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

def _snap_to_rationals(params, denominators=[1, 2, 4, 8]):
    """Try to snap parameters to rational values with small denominators."""
    U_flat = params[:40]
    V_flat = params[40:100]
    W_flat = params[100:160]
    
    U_mat = U_flat.reshape(4, 10)
    V_mat = V_flat.reshape(6, 10)
    W_mat = W_flat.reshape(6, 10)
    
    for denom in denominators:
        def snap(x):
            return round(x * denom) / denom
        
        U = [[snap(u) for u in row] for row in U_mat.tolist()]
        V = [[snap(v) for v in row] for row in V_mat.tolist()]
        W = [[snap(w) for w in row] for row in W_mat.tolist()]
        
        if _verify(U, V, W, 2, 2, 3):
            return {"U": U, "V": V, "W": W}
    
    return None

def _try_rank10_structured(deadline):
    """
    Try structured rank-10 decompositions based on Strassen-like patterns.
    """
    if time.time() > deadline - 20:
        return None
    
    # Try different structured patterns for the 10 terms
    # Pattern 1: 7 Strassen terms for 2x2 + 3 terms for column 2
    # We need to find if 3 terms can handle column 2 contributions
    
    # The key insight: for rank 10, we might be able to share some
    # Strassen terms with column 2 contributions
    
    # Try a parameterized family where some Strassen terms also contribute to column 2
    from scipy.optimize import least_squares
    
    # Define a structured parameterization
    # Let's try: first 7 terms are Strassen-like for 2x2, last 3 handle col 2
    # But allow some mixing
    
    best_result = None
    best_cost = float('inf')
    
    # Try multiple initial guesses
    rng = np.random.RandomState(42)
    
    for trial in range(20):
        if time.time() > deadline - 25:
            break
        
        # Start from rank-11 and try to merge two terms
        U11 = np.array(_rank11_decomposition()["U"], dtype=float)
        V11 = np.array(_rank11_decomposition()["V"], dtype=float)
        W11 = np.array(_rank11_decomposition()["W"], dtype=float)
        
        # Try to merge terms 7 and 8 (both contribute to W[2,0])
        # or terms 9 and 10 (both contribute to W[2,1])
        # Create a perturbed version with rank 10
        
        # Initialize with first 8 terms of rank-11, then try to fit
        U_init = np.zeros((4, 10))
        V_init = np.zeros((6, 10))
        W_init = np.zeros((6, 10))
        
        # Copy first 7 Strassen terms
        U_init[:, :7] = U11[:, :7]
        V_init[:, :7] = V11[:, :7]
        W_init[:, :7] = W11[:, :7]
        
        # For terms 7-9, start from the schoolbook column-2 terms with perturbation
        U_init[:, 7] = U11[:, 7] + rng.randn(4) * 0.1
        V_init[:, 7] = V11[:, 7] + rng.randn(6) * 0.1
        W_init[:, 7] = W11[:, 7] + rng.randn(6) * 0.1
        
        U_init[:, 8] = U11[:, 8] + rng.randn(4) * 0.1
        V_init[:, 8] = V11[:, 8] + rng.randn(6) * 0.1
        W_init[:, 8] = W11[:, 8] + rng.randn(6) * 0.1
        
        U_init[:, 9] = U11[:, 9] + rng.randn(4) * 0.1
        V_init[:, 9] = V11[:, 9] + rng.randn(6) * 0.1
        W_init[:, 9] = W11[:, 9] + rng.randn(6) * 0.1
        
        params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        try:
            result = least_squares(
                _residuals_rank10,
                params_init,
                method='lm',
                max_nfev=10000
            )
            
            if result.cost < best_cost:
                best_cost = result.cost
                best_params = result.x
            
            if result.cost < 1e-10:
                snapped = _snap_to_rationals(result.x)
                if snapped is not None:
                    return snapped
        except Exception:
            continue
    
    if best_cost < 1e-8 and best_params is not None:
        snapped = _snap_to_rationals(best_params)
        if snapped is not None:
            return snapped
    
    return None

def _try_rank10_search(deadline):
    """
    Attempt to find a rank-10 decomposition using Differential Evolution.
    """
    if time.time() > deadline - 30:
        return None
    
    try:
        from scipy.optimize import differential_evolution
    except ImportError:
        return None
    
    def objective(params):
        res = _residuals_rank10(params)
        return np.sum(res ** 2)
    
    # Bounds for coefficients
    bounds = [(-5, 5)] * 160
    
    # Try multiple seeds
    best_result = None
    best_cost = float('inf')
    
    for seed in [42, 123, 456]:
        if time.time() > deadline - 35:
            break
        
        try:
            result = differential_evolution(
                objective, 
                bounds, 
                maxiter=50, 
                popsize=20, 
                tol=1e-10, 
                seed=seed,
                polish=False
            )
            
            if result.fun < best_cost:
                best_cost = result.fun
                best_params = result.x
            
            if result.fun < 1e-8:
                from scipy.optimize import least_squares
                result_lm = least_squares(_residuals_rank10, result.x, method='lm', max_nfev=5000)
                if result_lm.cost < best_cost:
                    best_cost = result_lm.cost
                    best_params = result_lm.x
                
                if result_lm.cost < 1e-10:
                    snapped = _snap_to_rationals(result_lm.x)
                    if snapped is not None:
                        return snapped
        except Exception:
            continue
    
    if best_cost < 1e-8 and best_params is not None:
        snapped = _snap_to_rationals(best_params)
        if snapped is not None:
            return snapped
    
    return None

def construct():
    deadline = time.time() + 135
    
    result11 = _rank11_decomposition()
    if _verify(result11["U"], result11["V"], result11["W"], n, m, p):
        best = result11
        best_rank = 11
    else:
        best = _schoolbook()
        best_rank = 12
    
    if best_rank > 10 and time.time() < deadline - 40:
        # First try structured search (faster, more targeted)
        result10_struct = _try_rank10_structured(deadline)
        if result10_struct is not None:
            best = result10_struct
            best_rank = 10
        elif time.time() < deadline - 30:
            # Then try DE search
            result10 = _try_rank10_search(deadline)
            if result10 is not None:
                best = result10
                best_rank = 10
            
    return best
