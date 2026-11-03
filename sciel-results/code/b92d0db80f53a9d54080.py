"""Matrix multiplication 3x3 * 3x4 rank-minimal exact tensor decomposition."""
import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    n_m = n * m
    m_p = m * p
    p_n = p * n
    schoolbook_rank = n_m * m_p
    
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    def verify(U, V, W):
        """Verify exact decomposition. U: (n*m, r), V: (m*p, r), W: (p*n, r)"""
        r = U.shape[1]
        # Check all constraints
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Coefficient of A_{ij} B_{jk} in C_{ik} must be 1
                    val = 0.0
                    for c in range(r):
                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(val - 1.0) > 1e-10:
                        return False
                    # Coefficient of A_{ij} B_{jk'} in C_{ik} must be 0 for k' != k
                    for k2 in range(p):
                        if k2 == k:
                            continue
                        val = 0.0
                        for c in range(r):
                            val += U[i*m+j, c] * V[j*p+k2, c] * W[k*n+i, c]
                        if abs(val) > 1e-10:
                            return False
        return True
    
    def schoolbook():
        rank = n_m * m_p
        U = np.zeros((n_m, rank))
        V = np.zeros((m_p, rank))
        W = np.zeros((p_n, rank))
        for col in range(rank):
            i = col // (m * p)
            j = (col // p) % m
            k = col % p
            U[i * m + j, col] = 1
            V[j * p + k, col] = 1
            W[k * n + i, col] = 1
        return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
    
    def error_fn(params, rank):
        """Compute decomposition error from flattened parameters."""
        U_p = params[:n_m*rank].reshape(n_m, rank)
        V_p = params[n_m*rank:2*n_m*rank].reshape(m_p, rank)
        W_p = params[2*n_m*rank:].reshape(p_n, rank)
        
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(rank):
                        val += U_p[i*m+j, c] * V_p[j*p+k, c] * W_p[k*n+i, c]
                    err += (val - 1.0)**2
                    for k2 in range(p):
                        if k2 == k:
                            continue
                        val = 0.0
                        for c in range(rank):
                            val += U_p[i*m+j, c] * V_p[j*p+k2, c] * W_p[k*n+i, c]
                        err += val**2
        return err
    
    def snap_to_half_int(matrix):
        """Snap entries to nearest half-integer."""
        return np.round(np.array(matrix) * 2) / 2.0
    
    def try_rank(rank, max_restarts=10):
        """Try to find a valid decomposition of given rank."""
        for restart in range(max_restarts):
            if time.time() > deadline:
                return None
            
            # Random initialization with small values
            U = np.random.randn(n_m, rank) * 0.3
            V = np.random.randn(m_p, rank) * 0.3
            W = np.random.randn(p_n, rank) * 0.3
            
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            # Try L-BFGS-B optimization
            try:
                res = minimize(error_fn, x0, args=(rank,), method='L-BFGS-B',
                              options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10})
                
                if res.fun < 1e-8:
                    U_p = res.x[:n_m*rank].reshape(n_m, rank)
                    V_p = res.x[n_m*rank:2*n_m*rank].reshape(m_p, rank)
                    W_p = res.x[2*n_m*rank:].reshape(p_n, rank)
                    
                    # Snap to half-integers
                    U_s = snap_to_half_int(U_p)
                    V_s = snap_to_half_int(V_p)
                    W_s = snap_to_half_int(W_p)
                    
                    if verify(U_s, V_s, W_s):
                        return {'U': U_s.tolist(), 'V': V_s.tolist(), 'W': W_s.tolist()}
                        
                    # Try without snapping if close enough
                    if error_fn(np.concatenate([U_p.flatten(), V_p.flatten(), W_p.flatten()]), rank) < 1e-12:
                        # Try rounding to integers first
                        U_i = np.round(U_p)
                        V_i = np.round(V_p)
                        W_i = np.round(W_p)
                        if verify(U_i, V_i, W_i):
                            return {'U': U_i.tolist(), 'V': V_i.tolist(), 'W': W_i.tolist()}
            except Exception:
                pass
        
        return None
    
    # Try ranks from 32 up to schoolbook rank
    # Based on known results, rank 34 should be achievable for 3x3x4
    for rank in range(32, schoolbook_rank + 1):
        if time.time() > deadline:
            break
        
        result = try_rank(rank, max_restarts=8)
        if result is not None:
            return result
    
    # Fallback to schoolbook
    return schoolbook()

# EVOLVE-BLOCK-END
