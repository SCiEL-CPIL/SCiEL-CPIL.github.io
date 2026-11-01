import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [2, 2, 4]

def _build_strassen_block():
    """Return U, V, W for a 2x2 * 2x2 Strassen decomposition (rank 7)."""
    U_s = np.array([
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ], dtype=float)
    
    V_s = np.array([
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ], dtype=float)
    
    P_s = [
        np.array([[1, 0], [0, 1]]),
        np.array([[0, 0], [1, -1]]),
        np.array([[0, 1], [0, 1]]),
        np.array([[1, 0], [1, 0]]),
        np.array([[-1, 1], [0, 0]]),
        np.array([[0, 0], [0, 1]]),
        np.array([[1, 0], [0, 0]])
    ]
    
    return U_s, V_s, P_s

def _build_rank14():
    """Build the rank-14 decomposition using Strassen on two 2x2 blocks."""
    n, m, p = 2, 2, 4
    rank = 14
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    U_s, V_s, P_s = _build_strassen_block()
    
    for s in range(7):
        # Block 1 (k in 0,1)
        r1 = s
        U[:, r1] = U_s[s]
        for j in range(2):
            for kp in range(2):
                V[j * 4 + kp, r1] = V_s[s][j * 2 + kp]
        for k in range(2):
            for i in range(2):
                W[k * 2 + i, r1] = P_s[s][i, k]
        
        # Block 2 (k in 2,3)
        r2 = 7 + s
        U[:, r2] = U_s[s]
        for j in range(2):
            for kp in range(2):
                V[j * 4 + (kp + 2), r2] = V_s[s][j * 2 + kp]
        for k in range(2):
            for i in range(2):
                W[(k + 2) * 2 + i, r2] = P_s[s][i, k]
    
    return U, V, W

def _verify_decomposition(U, V, W, tol=1e-10):
    """Verify that the decomposition correctly computes matrix multiplication."""
    n, m, p = 2, 2, 4
    rank = U.shape[1]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
                if abs(val - 1.0) > tol:
                    return False
    return True

def _try_rank13_systematic(time_budget):
    """
    Try to find a rank-13 decomposition by directly solving the underdetermined system.
    We have 16 constraints and 260 unknowns for rank 13.
    Use random initialization + optimization to find a solution.
    """
    n, m, p = 2, 2, 4
    t_start = time.time()
    deadline = t_start + time_budget
    
    rank = 13
    U_size = n * m * rank  # 4*13 = 52
    V_size = m * p * rank  # 8*13 = 104
    W_size = p * n * rank  # 8*13 = 104
    total = U_size + V_size + W_size
    
    def compute_error(x):
        Ux = x[:U_size].reshape(n*m, rank)
        Vx = x[U_size:U_size+V_size].reshape(m*p, rank)
        Wx = x[U_size+V_size:total].reshape(p*n, rank)
        
        error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += Ux[i*m + j, r] * Vx[j*p + k, r] * Wx[k*n + i, r]
                    error += (val - 1.0) ** 2
        return error
    
    # Try multiple random initializations
    for attempt in range(20):
        if time.time() > deadline:
            break
        
        # Random initialization with small values
        np.random.seed(42 + attempt)
        x0 = np.random.randn(total) * 0.5
        
        try:
            result = minimize(compute_error, x0, method='L-BFGS-B', 
                            options={'maxiter': 100, 'ftol': 1e-15, 'gtol': 1e-15})
            
            if result.fun < 1e-12:
                Ux = result.x[:U_size].reshape(n*m, rank)
                Vx = result.x[U_size:U_size+V_size].reshape(m*p, rank)
                Wx = result.x[U_size+V_size:total].reshape(p*n, rank)
                
                if _verify_decomposition(Ux, Vx, Wx, tol=1e-9):
                    return Ux, Vx, Wx
        except Exception:
            continue
    
    return None

def _try_rank13_from_rank14(U14, V14, W14, time_budget):
    """
    Try to find a rank-13 decomposition by perturbing the rank-14 solution.
    Remove one column and optimize the remaining 13.
    """
    n, m, p = 2, 2, 4
    t_start = time.time()
    deadline = t_start + time_budget
    
    rank = 13
    U_size = n * m * rank
    V_size = m * p * rank
    W_size = p * n * rank
    total = U_size + V_size + W_size
    
    def compute_error(x):
        Ux = x[:U_size].reshape(n*m, rank)
        Vx = x[U_size:U_size+V_size].reshape(m*p, rank)
        Wx = x[U_size+V_size:total].reshape(p*n, rank)
        
        error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += Ux[i*m + j, r] * Vx[j*p + k, r] * Wx[k*n + i, r]
                    error += (val - 1.0) ** 2
        return error
    
    # Try removing each of the 14 columns
    for zero_col in range(14):
        if time.time() > deadline:
            break
        
        mask = np.ones(14, dtype=bool)
        mask[zero_col] = False
        
        U = U14[:, mask]
        V = V14[:, mask]
        W = W14[:, mask]
        
        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        try:
            result = minimize(compute_error, x0, method='L-BFGS-B',
                            options={'maxiter': 50, 'ftol': 1e-15, 'gtol': 1e-15})
            
            if result.fun < 1e-12:
                Ux = result.x[:U_size].reshape(n*m, rank)
                Vx = result.x[U_size:U_size+V_size].reshape(m*p, rank)
                Wx = result.x[U_size+V_size:total].reshape(p*n, rank)
                
                if _verify_decomposition(Ux, Vx, Wx, tol=1e-9):
                    return Ux, Vx, Wx
        except Exception:
            continue
    
    return None

def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = t0 + 130  # 130s deadline, 20s safety margin
    
    # Start with rank 14
    U14, V14, W14 = _build_rank14()
    if not _verify_decomposition(U14, V14, W14):
        raise RuntimeError("Rank 14 decomposition failed verification")
    
    best_rank = 14
    best_U, best_V, best_W = U14, V14, W14
    
    # Try to find rank 13 via optimization from rank-14
    remaining_time = deadline - time.time()
    if remaining_time > 10:
        result = _try_rank13_from_rank14(U14, V14, W14, min(remaining_time - 5, 60))
        if result is not None:
            best_U, best_V, best_W = result
            best_rank = 13
    
    # If still time, try systematic search for rank 13
    if best_rank == 14:
        remaining_time = deadline - time.time()
        if remaining_time > 10:
            result = _try_rank13_systematic(min(remaining_time - 5, 50))
            if result is not None:
                best_U, best_V, best_W = result
                best_rank = 13
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
