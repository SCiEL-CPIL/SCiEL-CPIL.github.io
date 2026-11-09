"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0
    
    # Schoolbook solution
    rank_school = n * m * p
    U = np.zeros((n * m, rank_school))
    V = np.zeros((m * p, rank_school))
    W = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    best = {'U': U, 'V': V, 'W': W}
    best_rank = rank_school
    
    # Try to find a rank 18 solution
    # The rank of <2,3,6> is 18.
    # We use scipy.optimize.least_squares to find U, V, W
    # such that the tensor identity holds.
    
    try:
        from scipy.optimize import least_squares
        
        def tensor_residuals(params):
            # params: flattened U, V, W
            # Sizes: n*m*R, m*p*R, p*n*R
            R = 18
            size_U = n * m * R
            size_V = m * p * R
            size_W = p * n * R
            
            U_flat = params[:size_U]
            V_flat = params[size_U:size_U + size_V]
            W_flat = params[size_U + size_V:]
            
            U = U_flat.reshape(n * m, R)
            V = V_flat.reshape(m * p, R)
            W = W_flat.reshape(p * n, R)
            
            residuals = []
            # Check identity: sum_c U[i*m+j, c] V[j'*p+k', c] W[k'*n+i', c] == delta_{i,i'} delta_{j,j'} delta_{k,k'}
            # There are (n*m*p) * (n*m*p) terms.
            # We only need to check the terms where the sum should be 1, and a few where it should be 0.
            # Actually, we need to check all to ensure exact decomposition.
            # But for optimization, we can check a subset or all.
            # 36*36 = 1296 residuals.
            
            idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        for ip in range(n):
                            for jp in range(m):
                                for kp in range(p):
                                    expected = 1.0 if (i == ip and j == jp and k == kp) else 0.0
                                    val = np.dot(U[i * m + j, :], V[jp * p + kp, :] * W[kp * n + ip, :])
                                    residuals.append(val - expected)
                                    idx += 1
                                  
            return residuals

        # Initial guess: random
        np.random.seed(42)
        R = 18
        size_U = n * m * R
        size_V = m * p * R
        size_W = p * n * R
        total_size = size_U + size_V + size_W
        
        best_residual = np.inf
        best_params = None
        
        # Try a few random restarts
        for restart in range(5):
            if time.time() > deadline:
                break
                
            x0 = np.random.randn(total_size)
            
            # Scale to avoid large numbers
            x0 = x0 / 10.0
            
            try:
                res = least_squares(tensor_residuals, x0, method='lm', max_nfev=1000)
                if res.cost < best_residual:
                    best_residual = res.cost
                    best_params = res.x
            except Exception:
                continue
                
        if best_params is not None and best_residual < 1e-6:
            # Extract U, V, W
            U_flat = best_params[:size_U]
            V_flat = best_params[size_U:size_U + size_V]
            W_flat = best_params[size_U + size_V:]
            
            U = U_flat.reshape(n * m, R)
            V = V_flat.reshape(m * p, R)
            W = W_flat.reshape(p * n, R)
            
            # Round to half-integers
            # Multiply by 2, round, divide by 2
            U_2 = np.round(U * 2) / 2
            V_2 = np.round(V * 2) / 2
            W_2 = np.round(W * 2) / 2
            
            # Verify if the rounded solution is still valid
            # Check residuals again
            max_err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        for ip in range(n):
                            for jp in range(m):
                                for kp in range(p):
                                    expected = 1.0 if (i == ip and j == jp and k == kp) else 0.0
                                    val = np.dot(U_2[i * m + j, :], V_2[jp * p + kp, :] * W_2[kp * n + ip, :])
                                    err = abs(val - expected)
                                    if err > max_err:
                                        max_err = err
                                    
            if max_err < 1e-9:
                best = {'U': U_2, 'V': V_2, 'W': W_2}
                best_rank = R
                return best
    except Exception:
        pass
        
    return best

# EVOLVE-BLOCK-END
