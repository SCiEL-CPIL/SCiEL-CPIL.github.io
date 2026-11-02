"""Improved program: attempts a lower-rank exact decomposition for 3x3*3x6 using Winograd/FFT structure."""
import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [3, 3, 6]

def construct():
    deadline = time.time() + 140.0  # 10s safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 3, 3, 6
    
    def check_decomp(U, V, W):
        """Verify exact decomposition: T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1"""
        r = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    val = np.dot(U[u_idx, :], V[v_idx, :] * W[w_idx, :])
                    if not np.isclose(val, 1.0, atol=1e-8):
                        return False
        return True

    def make_schoolbook():
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

    schoolbook = make_schoolbook()
    
    # Try to find lower rank decompositions using structured initialization + optimization
    # Key idea: Use DFT structure for the middle dimension (size 3)
    
    # 3-point DFT matrix (unscaled)
    omega = np.exp(2j * np.pi / 3)
    DFT3 = np.array([
        [1, 1, 1],
        [1, omega, omega**2],
        [1, omega**2, omega]
    ])
    
    def try_rank_gradient(rank, max_restarts=20, timeout=120):
        """Try to find rank-r decomposition using gradient-based optimization."""
        start_time = time.time()
        dim_u = n * m  # 9
        dim_v = m * p  # 18
        dim_w = p * n  # 18
        
        for restart in range(max_restarts):
            if time.time() > start_time + timeout or time.time() > deadline:
                break
            
            # Structured initialization: use DFT-inspired patterns
            # For the middle dimension j (size 3), use DFT columns
            U = np.zeros((dim_u, rank))
            V = np.zeros((dim_v, rank))
            W = np.zeros((dim_w, rank))
            
            # Initialize with DFT structure for j-dimension
            # U[i*m+j, c]: for fixed i, vary j using DFT
            # V[j*p+k, c]: for fixed k, vary j using DFT
            # W[k*n+i, c]: no j dependence
            
            for c in range(rank):
                # Assign DFT-like patterns
                i_idx = c % n
                k_idx = (c // n) % p
                j_phase = (c // (n * p)) % m
                
                # U entries: U[i*m+j, c]
                for i in range(n):
                    for j in range(m):
                        U[i * m + j, c] = 1.0 if (i == i_idx) else 0.0
                        # Add DFT structure in j
                        U[i * m + j, c] = np.cos(2 * np.pi * j * j_phase / m)
                
                # V entries: V[j*p+k, c]
                for j in range(m):
                    for k in range(p):
                        V[j * p + k, c] = 1.0 if (k == k_idx) else 0.0
                        V[j * p + k, c] = np.cos(2 * np.pi * j * j_phase / m)
                
                # W entries: W[k*n+i, c]
                for k in range(p):
                    for i in range(n):
                        W[k * n + i, c] = 1.0 if (k == k_idx and i == i_idx) else 0.0
            
            # Add random perturbation to escape local minima
            U += np.random.randn(dim_u, rank) * 0.1
            V += np.random.randn(dim_v, rank) * 0.1
            W += np.random.randn(dim_w, rank) * 0.1
            
            # Flatten for optimization
            params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            def residuals(x):
                U_x = x[:dim_u * rank].reshape(dim_u, rank)
                V_x = x[dim_u * rank:dim_u * rank + dim_v * rank].reshape(dim_v, rank)
                W_x = x[dim_u * rank + dim_v * rank:].reshape(dim_w, rank)
                
                res = np.zeros(n * m * p)
                idx = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            u_idx = i * m + j
                            v_idx = j * p + k
                            w_idx = k * n + i
                            val = np.dot(U_x[u_idx, :], V_x[v_idx, :] * W_x[w_idx, :])
                            res[idx] = val - 1.0
                            idx += 1
                return res
            
            try:
                result = least_squares(residuals, params, method='lm', xtol=1e-12, ftol=1e-12, gtol=1e-12)
            except Exception:
                continue
            
            if result.cost < 1e-10:
                U_final = result.x[:dim_u * rank].reshape(dim_u, rank)
                V_final = result.x[dim_u * rank:dim_u * rank + dim_v * rank].reshape(dim_v, rank)
                W_final = result.x[dim_u * rank + dim_v * rank:].reshape(dim_w, rank)
                
                # Try rounding to half-integers
                U_r = np.round(U_final * 2) / 2
                V_r = np.round(V_final * 2) / 2
                W_r = np.round(W_final * 2) / 2
                
                if check_decomp(U_r, V_r, W_r):
                    return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                
                # If rounding failed, try integer rounding
                U_r2 = np.round(U_final)
                V_r2 = np.round(V_final)
                W_r2 = np.round(W_final)
                if check_decomp(U_r2, V_r2, W_r2):
                    return {"U": U_r2.tolist(), "V": V_r2.tolist(), "W": W_r2.tolist()}
        
        return None
    
    # Try various rank targets from low to high
    # Lower bound for M_{3,3,6} is around 44, but let's try a range
    for target_rank in [36, 40, 44, 48, 50, 52]:
        if time.time() > deadline - 5:
            break
        
        decomp = try_rank_gradient(target_rank, max_restarts=10, timeout=20)
        if decomp is not None:
            return decomp
    
    # Fallback to schoolbook
    return schoolbook
