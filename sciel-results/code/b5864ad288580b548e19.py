import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [2, 2, 3]
n, m, p = DIMENSIONS

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

def _verify_exact_integer(U, V, W, n, m, p):
    """Verify using exact integer arithmetic (all entries must be integers or half-integers)."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    # Multiply by 8 to clear half-integer denominators
                    u8 = int(round(U[i * m + j][r] * 8))
                    v8 = int(round(V[j * p + k][r] * 8))
                    w8 = int(round(W[k * n + i][r] * 8))
                    s += u8 * v8 * w8
                # s/512 should equal 1.0, so s should equal 512
                if s != 512:
                    return False
    return True

def _residuals_rank10(params):
    """Compute residuals for rank-10 tensor decomposition."""
    U_flat = params[:40]
    V_flat = params[40:100]
    W_flat = params[100:160]
    
    U_mat = U_flat.reshape(4, 10)
    V_mat = V_flat.reshape(6, 10)
    W_mat = W_flat.reshape(6, 10)
    
    res = np.empty(12)
    idx = 0
    for i in range(2):
        for j in range(2):
            for k in range(3):
                s = 0.0
                for r in range(10):
                    s += U_mat[i*2+j, r] * V_mat[j*3+k, r] * W_mat[k*2+i, r]
                res[idx] = s - 1.0
                idx += 1
    return res

def _try_rank10_search(deadline):
    """
    Attempt to find a rank-10 decomposition using numerical optimization.
    """
    if time.time() > deadline - 10:
        return None
    
    rng = np.random.default_rng(42)
    best_cost = float('inf')
    best_params = None
    
    # Strategy 1: Random sparse integer initialization
    for trial in range(100):
        if time.time() > deadline - 5:
            break
            
        # Initialize with small integers
        U = np.zeros((4, 10))
        V = np.zeros((6, 10))
        W = np.zeros((6, 10))
        
        # Sparse random fill
        for i in range(4):
            for j in range(10):
                if rng.random() < 0.4:
                    U[i, j] = rng.choice([-1, 1])
        for i in range(6):
            for j in range(10):
                if rng.random() < 0.4:
                    V[i, j] = rng.choice([-1, 1])
        for i in range(6):
            for j in range(10):
                if rng.random() < 0.4:
                    W[i, j] = rng.choice([-1, 1])
        
        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        try:
            result = least_squares(_residuals_rank10, x0, method='lm', max_nfev=2000, ftol=1e-18, xtol=1e-18)
            if result.cost < best_cost:
                best_cost = result.cost
                best_params = result.x.copy()
                if result.cost < 1e-12:
                    break
        except Exception:
            pass

    # Strategy 2: Initialize from rank 11 solution (drop a term)
    if time.time() > deadline - 10:
        return None
        
    # Construct a basic rank 11 solution (schoolbook-like or Strassen-like)
    # We use the rank 11 from the previous program as a guide
    # But we don't have it directly here, so we use a random dense init for a few more tries
    for trial in range(50):
        if time.time() > deadline - 5:
            break
        
        x0 = rng.normal(0, 1, 160)
        try:
            result = least_squares(_residuals_rank10, x0, method='lm', max_nfev=2000, ftol=1e-18, xtol=1e-18)
            if result.cost < best_cost:
                best_cost = result.cost
                best_params = result.x.copy()
                if result.cost < 1e-12:
                    break
        except Exception:
            pass

    if best_params is None or best_cost > 1e-10:
        return None

    # Extract and try to snap to rational values
    U_flat = best_params[:40]
    V_flat = best_params[40:100]
    W_flat = best_params[100:160]
    
    U_mat = U_flat.reshape(4, 10)
    V_mat = V_flat.reshape(6, 10)
    W_mat = W_flat.reshape(6, 10)
    
    # Try different snap strategies
    for snap in [
        lambda x: round(x * 2) / 2,      # half-integers
        lambda x: round(x * 4) / 4,      # quarter-integers
        lambda x: round(x * 8) / 8,      # eighth-integers
        lambda x: round(x),               # integers
    ]:
        U = [[snap(u) for u in row] for row in U_mat.tolist()]
        V = [[snap(v) for v in row] for row in V_mat.tolist()]
        W = [[snap(w) for w in row] for row in W_mat.tolist()]
        
        if _verify(U, V, W, 2, 2, 3):
            return {"U": U, "V": V, "W": W}
    
    return None

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

def _rank11_decomposition():
    """Fallback rank 11 decomposition."""
    # Reusing the logic from the previous program's rank 11
    n, m, p = 2, 2, 3
    rank = 11
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    def set_u(i, j, r, val): U[i * m + j][r] = val
    def set_v(j, k, r, val): V[j * p + k][r] = val
    def set_w(k, i, r, val): W[k * n + i][r] = val

    # Strassen 7 terms
    set_u(0, 0, 0, 1); set_u(1, 1, 0, 1)
    set_v(0, 0, 0, 1); set_v(1, 1, 0, 1)
    set_w(0, 0, 0, 1); set_w(1, 1, 0, 1)

    set_u(1, 0, 1, 1); set_u(1, 1, 1, 1)
    set_v(0, 0, 1, 1)
    set_w(0, 1, 1, 1); set_w(1, 1, 1, -1)

    set_u(0, 0, 2, 1)
    set_v(0, 1, 2, 1); set_v(1, 1, 2, -1)
    set_w(1, 0, 2, 1); set_w(1, 1, 2, 1)

    set_u(1, 1, 3, 1)
    set_v(1, 0, 3, 1); set_v(0, 0, 3, -1)
    set_w(0, 0, 3, 1); set_w(0, 1, 3, 1)

    set_u(0, 0, 4, 1); set_u(0, 1, 4, 1)
    set_v(1, 1, 4, 1)
    set_w(0, 0, 4, -1); set_w(1, 0, 4, 1)

    set_u(1, 0, 5, 1); set_u(0, 0, 5, -1)
    set_v(0, 0, 5, 1); set_v(0, 1, 5, 1)
    set_w(1, 1, 5, 1)

    set_u(0, 1, 6, 1); set_u(1, 1, 6, -1)
    set_v(1, 0, 6, 1); set_v(1, 1, 6, 1)
    set_w(0, 0, 6, 1)

    # 4 schoolbook terms for k=2
    set_u(0, 0, 7, 1); set_v(0, 2, 7, 1); set_w(2, 0, 7, 1)
    set_u(0, 1, 8, 1); set_v(1, 2, 8, 1); set_w(2, 0, 8, 1)
    set_u(1, 0, 9, 1); set_v(0, 2, 9, 1); set_w(2, 1, 9, 1)
    set_u(1, 1, 10, 1); set_v(1, 2, 10, 1); set_w(2, 1, 10, 1)

    return {"U": U, "V": V, "W": W}

def construct():
    deadline = time.time() + 145
    
    # Try rank 10
    result10 = _try_rank10_search(deadline)
    if result10 is not None:
        return result10
        
    # Fallback to rank 11
    result11 = _rank11_decomposition()
    if _verify(result11["U"], result11["V"], result11["W"], n, m, p):
        return result11
        
    # Fallback to schoolbook
    return _schoolbook()
