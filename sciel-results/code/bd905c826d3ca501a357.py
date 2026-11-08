"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import least_squares

def _validate(U, V, W, n, m, p):
    """Check exact decomposition.
    U shape (n*m, rank), V shape (m*p, rank), W shape (p*n, rank).
    W index: W[k*n + i, c].
    Condition: for all (i,j,k), sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    """
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Vectorized check
    # U3: (n, m, R)
    # V3: (m, p, R)
    # W3: (p, n, R)
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_vec = U[i*m+j, :]
                v_vec = V[j*p+k, :]
                w_vec = W[k*n+i, :]
                s = np.dot(u_vec, v_vec * w_vec)
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _try_rank(R, n, m, p, seed=42, time_limit=10.0):
    """Try to find a rank-R decomposition using structured ansatz + least_squares."""
    start_time = time.time()
    rng = np.random.RandomState(seed)
    
    n_m = n * m
    m_p = m * p
    p_n = p * n
    total_vars = R * (n_m + m_p + p_n)
    
    def residuals(x):
        U_flat = x[:R*n_m]
        V_flat = x[R*n_m:R*n_m + R*m_p]
        W_flat = x[R*n_m + R*m_p:]
        
        U3 = U_flat.reshape(n, m, R)
        V3 = V_flat.reshape(m, p, R)
        W3 = W_flat.reshape(p, n, R)
        
        res = np.zeros(n*m*p)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U3[i,j,:], V3[j,k,:] * W3[k,i,:])
                    res[idx] = s - 1.0
                    idx += 1
        return res

    best_solution = None
    best_err = float('inf')
    
    def _polish(x0, max_nfev=3000):
        try:
            sol = least_squares(residuals, x0, method='trf', max_nfev=max_nfev, ftol=1e-15, xtol=1e-15, gtol=1e-15)
            return sol.x, sol.cost
        except Exception:
            return None, float('inf')

    def _try_rounding(U_out, V_out, W_out):
        """Try rounding to integer or half-integer and validate."""
        # Integer
        U_int = np.round(U_out).astype(int)
        V_int = np.round(V_out).astype(int)
        W_int = np.round(W_out).astype(int)
        if _validate(U_int.tolist(), V_int.tolist(), W_int.tolist(), n, m, p):
            return U_int.tolist(), V_int.tolist(), W_int.tolist()
        
        # Half-integer
        U_half = np.round(U_out * 2) / 2.0
        V_half = np.round(V_out * 2) / 2.0
        W_half = np.round(W_out * 2) / 2.0
        if _validate(U_half.tolist(), V_half.tolist(), W_half.tolist(), n, m, p):
            return U_half.tolist(), V_half.tolist(), W_half.tolist()
        
        return None

    def _extract_and_check(x):
        U_flat = x[:R*n_m].reshape(n, m, R)
        V_flat = x[R*n_m:R*n_m + R*m_p].reshape(m, p, R)
        W_flat = x[R*n_m + R*m_p:].reshape(p, n, R)
        
        U_out = U_flat.reshape(n*m, R)
        V_out = V_flat.reshape(m*p, R)
        W_out = W_flat.reshape(p*n, R)
        
        return _try_rounding(U_out, V_out, W_out)

    # Strategy 1: Kronecker/block structure
    # For matrix multiplication (n x m) * (m x p), we can use block structure.
    # If we decompose m into blocks, we can share factors.
    
    # Strategy 2: Random structured init with low-rank projection
    max_restarts = 30
    for restart in range(max_restarts):
        if time.time() - start_time > time_limit:
            break
        
        x0 = np.zeros(total_vars)
        
        # Pick R random "centers" but with structured sharing
        # Group centers to encourage sharing
        if R <= 24:
            # Use a pattern where each rank index covers multiple (i,j,k) triples
            # by having non-zero entries in multiple positions
            for c in range(R):
                i_c = rng.randint(n)
                j_c = rng.randint(m)
                k_c = rng.randint(p)
                
                idx_u = ((i_c * m) + j_c) * R + c
                idx_v = R*n_m + ((j_c * p) + k_c) * R + c
                idx_w = R*n_m + R*m_p + ((k_c * n) + i_c) * R + c
                
                x0[idx_u] = 1.0
                x0[idx_v] = 1.0
                x0[idx_w] = 1.0
                
                # Add structured sharing: connect to nearby indices
                for di in range(-1, 2):
                    for dj in range(-1, 2):
                        for dk in range(-1, 2):
                            i2 = (i_c + di) % n
                            j2 = (j_c + dj) % m
                            k2 = (k_c + dk) % p
                            if i2 == i_c and j2 == j_c and k2 == k_c:
                                continue
                            idx_u2 = ((i2 * m) + j2) * R + c
                            idx_v2 = R*n_m + ((j2 * p) + k2) * R + c
                            idx_w2 = R*n_m + R*m_p + ((k2 * n) + i2) * R + c
                            if rng.random() < 0.3:
                                x0[idx_u2] += rng.uniform(0.1, 0.3)
                                x0[idx_v2] += rng.uniform(0.1, 0.3)
                                x0[idx_w2] += rng.uniform(0.1, 0.3)
        else:
            # For higher ranks, use sparser init
            for c in range(R):
                i_c = rng.randint(n)
                j_c = rng.randint(m)
                k_c = rng.randint(p)
                
                idx_u = ((i_c * m) + j_c) * R + c
                idx_v = R*n_m + ((j_c * p) + k_c) * R + c
                idx_w = R*n_m + R*m_p + ((k_c * n) + i_c) * R + c
                
                x0[idx_u] = 1.0
                x0[idx_v] = 1.0
                x0[idx_w] = 1.0
                
                # Small random perturbations
                for _ in range(3):
                    ru = rng.randint(R*n_m)
                    rv = rng.randint(R*m_p)
                    rw = rng.randint(R*p_n)
                    x0[ru] += rng.uniform(-0.05, 0.05)
                    x0[R*n_m + rv] += rng.uniform(-0.05, 0.05)
                    x0[R*n_m + R*m_p + rw] += rng.uniform(-0.05, 0.05)
        
        x_sol, err = _polish(x0, max_nfev=2000)
        if x_sol is not None and err < best_err:
            best_err = err
            best_solution = x_sol
            
            # Try rounding immediately
            result = _extract_and_check(x_sol)
            if result is not None:
                return result
    
    # Final polish with more iterations if we have a good solution
    if best_solution is not None and best_err < 1e-6:
        x_final, err_final = _polish(best_solution, max_nfev=5000)
        if x_final is not None:
            result = _extract_and_check(x_final)
            if result is not None:
                return result
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Lower bound
    lower_bound = max(n*m, m*p, p*n)
    
    # Try ranks from lower bound up to schoolbook
    schoolbook_rank = n * m * p
    
    for R in range(lower_bound, schoolbook_rank + 1):
        if time.time() > deadline - 5:
            break
            
        time_limit = min(12.0, deadline - time.time() - 2)
        if time_limit <= 0:
            break
            
        result = _try_rank(R, n, m, p, seed=R, time_limit=time_limit)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
    
    # Fallback: schoolbook decomposition
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
# EVOLVE-BLOCK-END
