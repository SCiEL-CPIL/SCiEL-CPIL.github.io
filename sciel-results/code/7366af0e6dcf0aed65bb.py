import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [3, 3, 6]
START_TIME = time.time()
TIME_LIMIT = 140  # safety margin

def construct():
    n, m, p = DIMENSIONS
    # Shapes:
    # U: (n*m, R) = (9, R)
    # V: (m*p, R) = (18, R)
    # W: (p*n, R) = (18, R)
    
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
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    def verify(U, V, W, tol=1e-9):
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        rank = U.shape[1]
        
        errors = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_val = U[i * m + j, :]
                    v_val = V[j * p + k, :]
                    w_val = W[k * n + i, :]
                    s = np.dot(u_val, v_val * w_val)
                    errors.append(s - 1.0)
        
        return max(abs(e) for e in errors) < tol

    def residual_func(x, R):
        # x is flattened U, V, W
        # U: 9*R, V: 18*R, W: 18*R
        U = x[:9*R].reshape((9, R))
        V = x[9*R:27*R].reshape((18, R))
        W = x[27*R:].reshape((18, R))
        
        # Calculate tensor T[i, j, k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
        # We want T[i, j, k] = 1
        # Residuals are T[i, j, k] - 1
        
        res = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # u_val is vector of length R
                    # v_val is vector of length R
                    # w_val is vector of length R
                    # sum(u_val * v_val * w_val)
                    u_val = U[i * m + j, :]
                    v_val = V[j * p + k, :]
                    w_val = W[k * n + i, :]
                    res[i, j, k] = np.dot(u_val, v_val * w_val)
        
        return (res - 1.0).flatten()

    def jacobian_func(x, R):
        # Finite difference is okay for small problems, but analytic is better if possible.
        # However, scipy's least_squares can compute finite difference Jacobian.
        # Given the small size, let's let scipy handle it or provide a sparse structure if needed.
        # For now, rely on scipy's internal FD or provide a dense Jacobian if we derive it.
        # Deriving analytic Jacobian:
        # f_ijk = sum_r U_imj_r V_jpk_r W_kni_r - 1
        # df_ijk / dU_imj_r = V_jpk_r W_kni_r
        # This is complex to vectorize efficiently for arbitrary R in a general solver call without overhead.
        # Let's try without providing Jacobian first (FD).
        pass

    def try_find_decomposition(R, num_restarts=50, deadline=None):
        rng = np.random.RandomState(42 + R)
        
        for r in range(num_restarts):
            if deadline is not None and time.time() > deadline - 5.0:
                return None, None, None
            
            # Random initialization
            # Scale matters. Let's use normal distribution with small std.
            # If std is too small, it might be stuck. If too large, it might diverge or take long.
            # Let's try std=1.
            x0 = rng.randn(45 * R)
            
            try:
                # Use Levenberg-Marquardt
                # x_scale='jac' might help
                result = least_squares(
                    residual_func, 
                    x0, 
                    args=(R,), 
                    method='lm', 
                    max_nfev=2000, 
                    xtol=1e-12, 
                    ftol=1e-12, 
                    gtol=1e-12
                )
                
                if result.success and result.cost < 1e-10:
                    # Extract U, V, W
                    x_opt = result.x
                    U = x_opt[:9*R].reshape((9, R))
                    V = x_opt[9*R:27*R].reshape((18, R))
                    W = x_opt[27*R:].reshape((18, R))
                    
                    # Verify
                    if verify(U, V, W, tol=1e-9):
                        return U, V, W
            except Exception as e:
                pass
                
        return None, None, None

    # Fallback to schoolbook
    schoolbook_sol = get_schoolbook()
    
    if time.time() - START_TIME < TIME_LIMIT - 10:
        deadline = START_TIME + TIME_LIMIT - 5
        school_rank = n * m * p  # 54
        
        # Try ranks in ascending order
        # Lower bound is max(n*m, m*p, p*n) = 18
        for R in range(18, school_rank):
            if time.time() > deadline - 10:
                break
            
            # Decrease restarts as we go higher in rank if time is tight, 
            # but lower ranks are harder so we might want more restarts there.
            # Let's keep restarts constant for simplicity, but check time.
            num_restarts = 100
            
            U, V, W = try_find_decomposition(R, num_restarts=num_restarts, deadline=deadline)
            
            if U is not None:
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return schoolbook_sol
