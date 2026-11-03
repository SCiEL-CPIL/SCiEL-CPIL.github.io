import numpy as np
from scipy.optimize import least_squares
import time

DIMENSIONS = [3, 5, 7]

def construct():
    n, m, p = DIMENSIONS
    target_rank = n * m * p
    
    # Target tensor T[i, j, k] = 1
    
    # Fast objective function using einsum
    def objective(params, r):
        size_u = n * m * r
        size_v = m * p * r
        size_w = p * n * r
        
        U = params[:size_u].reshape(n, m, r)
        V = params[size_u:size_u + size_v].reshape(m, p, r)
        W = params[size_u + size_v:].reshape(p, n, r)
        
        # P[i, j, k] = sum_r U[i, j, r] * V[j, k, r] * W[k, i, r]
        # Note: W is indexed by (k, i) in the decomposition formula, so W_3d[k, i, r]
        # In the reshape above, W is (p, n, r) which corresponds to indices (k, i, r)
        # So W[k, i, r] is W_3d[k, i, r].
        
        P = np.einsum('ijr,jkr,kir->ijk', U, V, W)
        
        return (P - 1.0).ravel()

    def verify(U, V, W, r):
        U_3d = U.reshape(n, m, r)
        V_3d = V.reshape(m, p, r)
        W_3d = W.reshape(p, n, r)
        P = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
        return np.max(np.abs(P - 1.0)) < 1e-9

    # Try to find lower rank
    best_result = None
    start_time = time.time()
    
    # We attempt ranks from target_rank - 1 down to target_rank - 10
    # If time is running out, we stop.
    
    for r in range(target_rank - 1, max(target_rank - 10, 1), -1):
        if time.time() - start_time > 110:
            break
            
        # Try a few random initializations
        for attempt in range(20):
            if time.time() - start_time > 115:
                break
                
            # Random seed for reproducibility in debugging, but varied for search
            np.random.seed(attempt * 1000 + r)
            
            # Initialize with small random values
            # Scale is important. The product of three terms should be ~1.
            # If U, V, W are ~1/sqrt(r) or similar, it helps convergence.
            scale = 1.0 / np.sqrt(r)
            params = np.random.randn(n * m * r + m * p * r + p * n * r) * scale
            
            try:
                # Use 'lm' (Levenberg-Marquardt) which is good for least squares
                # max_nfev is limited to prevent hanging
                res = least_squares(objective, params, args=(r,), method='lm', max_nfev=500, ftol=1e-10, xtol=1e-10, gtol=1e-10)
                
                if res.cost < 1e-6:
                    size_u = n * m * r
                    size_v = m * p * r
                    size_w = p * n * r
                    
                    U_opt = res.x[:size_u].reshape(n * m, r)
                    V_opt = res.x[size_u:size_u + size_v].reshape(m * p, r)
                    W_opt = res.x[size_u + size_v:].reshape(p * n, r)
                    
                    # Try rounding to nearest half-integer
                    # The solution might need scaling. 
                    # However, the equations are homogeneous in the sense that if (U, V, W) is a solution, 
                    # (aU, V, W) is not generally a solution unless we scale all three.
                    # Actually, T = sum U V W. If we scale U by a, V by b, W by c, T scales by abc.
                    # We need abc = 1.
                    # Usually, we can fix the scale of one matrix.
                    # Let's just try rounding to nearest 0.5 directly.
                    
                    U_round = np.round(U_opt * 2) / 2
                    V_round = np.round(V_opt * 2) / 2
                    W_round = np.round(W_opt * 2) / 2
                    
                    if verify(U_round, V_round, W_round, r):
                        best_result = {"U": U_round.tolist(), "V": V_round.tolist(), "W": W_round.tolist()}
                        break
                        
                    # If direct rounding fails, try to find a scaling factor for each column?
                    # Or just accept if the residual is extremely small, implying the exact solution is nearby.
                    # But the task requires EXACT decomposition.
                    # So we must find integer/half-integer entries that work.
                    
            except Exception:
                pass
                
        if best_result:
            break

    if best_result:
        return best_result
    else:
        # Fallback to schoolbook
        rank = target_rank
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
