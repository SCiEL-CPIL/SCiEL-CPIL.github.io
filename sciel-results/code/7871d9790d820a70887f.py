"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def _schoolbook():
    """Return the rank-27 schoolbook decomposition."""
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

def _verify(U, V, W, tol=1e-9):
    """Check that the decomposition exactly reconstructs the matmul tensor."""
    n, m, p = DIMENSIONS
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    rank = Ua.shape[1]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += Ua[i*m+j, c] * Va[j*p+k, c] * Wa[k*n+i, c]
                if abs(val - 1.0) > tol:
                    return False
                    
    for a in range(n):
        for b in range(m):
            for c in range(m):
                for d in range(p):
                    for e in range(p):
                        for f in range(n):
                            if b == c and d == e:
                                continue
                            val = 0.0
                            for c_idx in range(rank):
                                val += Ua[a*m+b, c_idx] * Va[c*p+d, c_idx] * Wa[e*n+f, c_idx]
                            if abs(val) > tol:
                                return False
    return True

def _check_half_ints(U, V, W, tol=1e-9):
    """Check all entries are half-integers."""
    for mat in [U, V, W]:
        for row in mat:
            for v in row:
                if abs(v * 2 - round(v * 2)) > tol:
                    return False
    return True

def _find_rank19():
    """
    Find a rank-19 decomposition for 3x3 matrix multiplication.
    Uses iterative numerical optimization with half-integer rounding.
    """
    start_time = time.time()
    deadline = start_time + 115  # safety margin
    
    n, m, p = DIMENSIONS
    rank = 19
    rng = np.random.RandomState(42)
    
    best_err = float('inf')
    best_U = best_V = best_W = None
    
    for trial in range(50):
        if time.time() > deadline - 5:
            break
            
        # Initialize with random values
        Ua = rng.randn(n*m, rank)
        Va = rng.randn(m*p, rank)
        Wa = rng.randn(p*n, rank)
        
        # Iterative refinement
        for it in range(200):
            if time.time() > deadline - 5:
                    break
                    
            # Compute errors
            err = 0.0
            # On-support: should be 1
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(Ua[i*m+j], Va[j*p+k] * Wa[k*n+i])
                        err += (val - 1.0) ** 2
            # Off-support: should be 0
            for a in range(n):
                for b in range(m):
                    for c in range(m):
                        for d in range(p):
                            for e in range(p):
                                for f in range(n):
                                    if b == c and d == e:
                                        continue
                                    val = np.dot(Ua[a*m+b], Va[c*p+d] * Wa[e*n+f])
                                    err += val ** 2
            
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = Ua.copy(), Va.copy(), Wa.copy()
            
            if err < 1e-10:
                break
            
            # Gradient-free update: perturb randomly
            scale = max(0.1, 0.01 * (err ** 0.25))
            Ua += rng.randn(n*m, rank) * scale
            Va += rng.randn(m*p, rank) * scale
            Wa += rng.randn(p*n, rank) * scale
        
        if best_err < 1e-6:
            break
    
    if best_U is None:
        return None
    
    # Round to half-integers
    def r(x):
        return np.round(x * 2) / 2.0
    
    U_r = r(best_U)
    V_r = r(best_V)
    W_r = r(best_W)
    
    U_list = [[float(v) for v in row] for row in U_r]
    V_list = [[float(v) for v in row] for row in V_r]
    W_list = [[float(v) for v in row] for row in W_r]
    
    # Verify
    if _verify(U_list, V_list, W_list) and _check_half_ints(U_list, V_list, W_list):
        return {"U": U_list, "V": V_list, "W": W_list}
    
    # Try without rounding if close
    if best_err < 1e-6:
        U_list = [[float(v) for v in row] for row in best_U]
        V_list = [[float(v) for v in row] for row in best_V]
        W_list = [[float(v) for v in row] for row in best_W]
        if _verify(U_list, V_list, W_list):
            return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 120
    
    # Try rank-19 first
    result = _find_rank19()
    if result is not None:
        return result
    
    # Fall back to schoolbook
    return _schoolbook()
# EVOLVE-BLOCK-END
