"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import sys
from scipy.optimize import least_squares

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 135  # Leave 15s margin before 150s kill
    
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 72
    NM, MP, PN = n * m, m * p, p * n  # 12, 24, 18
    
    # Build target tensor T[i,j,k] = 1 if (i,j,k) is valid
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0
    
    T_flat = T.flatten()
    
    def check_exact(U_flat, V_flat, W_flat, rank):
        """Check if U, V, W with given rank exactly reconstruct T"""
        U_r = np.array(U_flat).reshape(n, m, rank)
        V_r = np.array(V_flat).reshape(m, p, rank)
        W_r = np.array(W_flat).reshape(p, n, rank)
        T_rec = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
        return np.allclose(T_rec.flatten(), T_flat, atol=1e-9)
    
    def objective(params, rank):
        """Objective function for least_squares: returns residual vector"""
        U_size = n * m * rank
        V_size = m * p * rank
        W_size = p * n * rank
        
        U_flat = params[:U_size]
        V_flat = params[U_size:U_size + V_size]
        W_flat = params[U_size + V_size:]
        
        U_r = U_flat.reshape(n, m, rank)
        V_r = V_flat.reshape(m, p, rank)
        W_r = W_flat.reshape(p, n, rank)
        
        T_rec = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
        return (T_rec - T).flatten()
    
    def find_rank_decomposition(rank, num_trials=5, max_nfev=2000):
        """Try to find exact decomposition of given rank using least_squares"""
        total_params = (n * m + m * p + p * n) * rank
        
        for trial in range(num_trials):
            if time.time() > deadline - 5:
                return None
            
            np.random.seed(trial * 1000 + 42)
            
            # Initialize with small random values
            params0 = np.random.randn(total_params) * 0.3
            
            # Run least_squares
            try:
                result = least_squares(
                    objective, 
                    params0, 
                    args=(rank,),
                    method='trf',  # Trust region reflective
                    max_nfev=max_nfev,
                    xtol=1e-12,
                    ftol=1e-12,
                    gtol=1e-12
                )
                
                if result.cost < 1e-10:
                    # Found a good solution, now try rounding
                    U_flat = result.x[:n*m*rank]
                    V_flat = result.x[n*m*rank:(n*m+m*p)*rank]
                    W_flat = result.x[(n*m+m*p)*rank:]
                    
                    # Try half-integer rounding
                    U_half = np.round(U_flat * 2) / 2
                    V_half = np.round(V_flat * 2) / 2
                    W_half = np.round(W_flat * 2) / 2
                    if check_exact(U_half, V_half, W_half, rank):
                        return (U_half.tolist(), V_half.tolist(), W_half.tolist())
                    
                    # Try integer rounding
                    U_int = np.round(U_flat)
                    V_int = np.round(V_flat)
                    W_int = np.round(W_flat)
                    if check_exact(U_int, V_int, W_int, rank):
                        return (U_int.tolist(), V_int.tolist(), W_int.tolist())
                    
                    # Try quarter rounding
                    U_q = np.round(U_flat * 4) / 4
                    V_q = np.round(V_flat * 4) / 4
                    W_q = np.round(W_flat * 4) / 4
                    if check_exact(U_q, V_q, W_q, rank):
                        return (U_q.tolist(), V_q.tolist(), W_q.tolist())
                    
                    # If no rounding works, return the real-valued solution (should be exact enough)
                    return (U_flat.tolist(), V_flat.tolist(), W_flat.tolist())
                    
            except Exception as e:
                continue
        
        return None
    
    # Try ranks in increasing order from a reasonable lower bound
    # Known lower bounds for 3x4x6 matrix multiplication tensor suggest rank >= 50
    # Search from 50 upward to find the smallest achievable rank
    best_result = None
    
    # First try to find any rank below schoolbook
    for rank in range(50, min(72, 70), 1):
        if time.time() > deadline - 10:
            break
        
        result = find_rank_decomposition(rank, num_trials=3, max_nfev=1500)
        if result is not None:
            U, V, W = result
            best_result = {"U": U, "V": V, "W": W}
            break
    
    # If no rank found in first pass, try a few more with more effort
    if best_result is None:
        for rank in range(48, 55, 1):
            if time.time() > deadline - 10:
                break
            
            result = find_rank_decomposition(rank, num_trials=5, max_nfev=2000)
            if result is not None:
                U, V, W = result
                best_result = {"U": U, "V": V, "W": W}
                break
    
    # Fallback to schoolbook
    if best_result is None:
        rank = rank_schoolbook
        U = [[0] * rank for _ in range(NM)]
        V = [[0] * rank for _ in range(MP)]
        W = [[0] * rank for _ in range(PN)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        best_result = {"U": U, "V": V, "W": W}
    
    return best_result
# EVOLVE-BLOCK-END
