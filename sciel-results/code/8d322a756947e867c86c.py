import numpy as np
import time
from scipy.optimize import least_squares

n, m, p = 2, 5, 6

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    def check_decomposition(U, V, W):
        R = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    val = np.sum(U[u_idx, :] * V[v_idx, :] * W[w_idx, :])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    def try_optimize_rank(R):
        """
        Try to find a decomposition of rank R using continuous optimization.
        We minimize the error of the tensor identity.
        """
        if time.time() > deadline - 20:
            return None
            
        # Initialize with random values
        # To help convergence, we can start near schoolbook if R is close to 60
        # But for R < 60, we must drop some terms.
        
        # We represent the variables as a flat vector
        # U: n*m x R, V: m*p x R, W: p*n x R
        size_U = n * m * R
        size_V = m * p * R
        size_W = p * n * R
        total_size = size_U + size_V + size_W
        
        # We need to map the tensor equation to residuals.
        # Residual for (i, j, k) is sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] - 1
        # There are n*m*p = 60 residuals.
        
        def residuals(x):
            U = x[:size_U].reshape((n*m, R))
            V = x[size_U:size_U+size_V].reshape((m*p, R))
            W = x[size_U+size_V:].reshape((p*n, R))
            
            res = np.zeros(n*m*p)
            idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u_idx = i * m + j
                        v_idx = j * p + k
                        w_idx = k * n + i
                        val = np.sum(U[u_idx, :] * V[v_idx, :] * W[w_idx, :])
                        res[idx] = val - 1.0
                        idx += 1
            return res

        # Initial guess: Random small values
        # Better: Use schoolbook and project to lower rank?
        # For now, random initialization with multiple restarts.
        
        for restart in range(20):
            if time.time() > deadline - 20:
                return None
                
            # Random initialization
            x0 = np.random.randn(total_size) * 0.5
            
            try:
                result = least_squares(residuals, x0, method='lm', max_nfev=1000)
            except Exception:
                continue
                
            if result.cost < 1e-10:
                # Found a real-valued solution
                U = result.x[:size_U].reshape((n*m, R))
                V = result.x[size_U:size_U+size_V].reshape((m*p, R))
                W = result.x[size_U+size_U+size_W].reshape((p*n, R))
                
                # Try to snap to integers/half-integers
                # The solution might be close to integers.
                # Scale? The problem is homogeneous, so scale doesn't matter for exactness,
                # but we need integer/half-integer.
                
                # Check if U, V, W are close to integers
                # We can try to find a scaling factor or just check proximity.
                
                # Actually, if the solution is valid, we can try to find integer representations.
                # This is a Diophantine problem.
                # However, for small ranks, the solutions are often simple integers.
                
                # Let's check if the entries are close to half-integers.
                U_int = np.round(U * 2) / 2
                V_int = np.round(V * 2) / 2
                W_int = np.round(W * 2) / 2
                
                if check_decomposition(U_int, V_int, W_int):
                    return U_int, V_int, W_int
                
                # If not, try rounding to integers
                U_int = np.round(U)
                V_int = np.round(V)
                W_int = np.round(W)
                
                if check_decomposition(U_int, V_int, W_int):
                    return U_int, V_int, W_int
        return None

    # Try ranks from 59 down to 50
    for R in range(59, 49, -1):
        if time.time() > deadline - 30:
            break
        result = try_optimize_rank(R)
        if result is not None:
            U, V, W = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
    # Fallback
    U, V, W = get_schoolbook()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
