import numpy as np
import time
from scipy.optimize import least_squares

DIMENSIONS = [2, 2, 7]

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U,V,W correctly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    # T[a, b, c] = 1 if indices match multiplication rule
    # a = i*m + j
    # b = jp*p + kp
    # c = kp*n + ip
    # Rule: i==ip, j==jp, k==kp
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    for kp in range(p):
                        for ip in range(n):
                            lhs = 1.0 if (i == ip and j == jp and k == kp) else 0.0
                            rhs = 0.0
                            for r in range(rank):
                                rhs += U[i*m+j, r] * V[jp*p+kp, r] * W[kp*n+ip, r]
                            if abs(lhs - rhs) > 1e-9:
                                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS
    n_m = n * m
    m_p = m * p
    p_n = p * n
    
    # Build target tensor T
    T = np.zeros((n_m, m_p, p_n))
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    for kp in range(p):
                        for ip in range(n):
                            if i == ip and j == jp and k == kp:
                                a = i*m + j
                                b = jp*p + k
                                c = kp*n + ip
                                T[a, b, c] = 1.0
    T_flat = T.flatten()
    
    # Schoolbook fallback
    def make_schoolbook():
        rank = n * m * p
        U = np.zeros((n_m, rank))
        V = np.zeros((m_p, rank))
        W = np.zeros((p_n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W
    
    best_U, best_V, best_W = make_schoolbook()
    best_rank = n * m * p
    
    # Try to find lower rank
    # Known lower bound is likely 25 based on inspiration
    for r in range(25, best_rank):
        if time.time() > deadline - 10:
            break
            
        # Multiple trials for this rank
        found = False
        for trial in range(50):
            if time.time() > deadline - 10:
                break
                
            # Random initialization
            # Use small random values to avoid overflow initially
            U = np.random.randn(n_m, r) * 0.5
            V = np.random.randn(m_p, r) * 0.5
            W = np.random.randn(p_n, r) * 0.5
            
            def residuals(params):
                U_r = params[:n_m*r].reshape(n_m, r)
                V_r = params[n_m*r:n_m*r + m_p*r].reshape(m_p, r)
                W_r = params[n_m*r + m_p*r:].reshape(p_n, r)
                
                # Compute tensor product sum
                # T_calc[a, b, c] = sum_k U[a,k] V[b,k] W[c,k]
                # Efficiently: 
                # Let M_ab = U @ V.T (n_m x m_p)
                # Then T_calc is not directly M_ab @ W.T
                # T_calc[a,b,c] = sum_k (U[a,k] V[b,k]) W[c,k]
                # This is a 3rd order tensor.
                # We can compute it as:
                # Reshape U, V, W to (n_m, 1, r), (1, m_p, r), (1, 1, p_n)? No.
                # Use einsum
                try:
                    T_calc = np.einsum('ak,bk,ck->abc', U_r, V_r, W_r)
                except:
                    return np.ones_like(T_flat)
                
                return T_calc.flatten() - T_flat
            
            try:
                # Use least_squares
                # x0 is flattened U, V, W
                x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
                result = least_squares(residuals, x0, method='lm', max_nfev=1000)
                
                if result.success or np.linalg.norm(result.fun) < 1e-8:
                    # Extract U, V, W
                    U_new = result.x[:n_m*r].reshape(n_m, r)
                    V_new = result.x[n_m*r:n_m*r + m_p*r].reshape(m_p, r)
                    W_new = result.x[n_m*r + m_p*r:].reshape(p_n, r)
                    
                    # Verify exactly
                    if _verify_decomposition(U_new, V_new, W_new, n, m, p):
                        best_U = U_new
                        best_V = V_new
                        best_W = W_new
                        best_rank = r
                        found = True
                        break
            except Exception:
                pass
        
        if found:
            break
    
    # Final check
    if not _verify_decomposition(best_U, best_V, best_W, n, m, p):
        best_U, best_V, best_W = make_schoolbook()
        best_rank = n * m * p
        
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
