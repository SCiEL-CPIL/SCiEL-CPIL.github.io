import numpy as np
import time

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140

    target_ranks = [41, 40, 39, 38, 37, 36]

    num_u = n * m
    num_v = m * p
    num_w = p * n

    def error_func(params, R):
        U = params[:R*num_u].reshape(R, num_u).T
        V = params[R*num_u:R*(num_u+num_v)].reshape(R, num_v).T
        W = params[R*(num_u+num_v):].reshape(R, num_w).T
        
        errors = np.zeros(n * m * p)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    val = np.dot(U[u_idx], V[v_idx] * W[w_idx])
                    errors[idx] = val - 1.0
                    idx += 1
        return errors

    def snap_func(params, R):
        # Returns residuals for snapping to half-integers
        # Residual is sin(pi * 2 * x)
        return np.sin(np.pi * 2.0 * params)

    def combined_error_func(params, R, snap_weight):
        U = params[:R*num_u].reshape(R, num_u).T
        V = params[R*num_u:R*(num_u+num_v)].reshape(R, num_v).T
        W = params[R*(num_u+num_v):].reshape(R, num_w).T
        
        # Tensor errors
        tensor_errors = np.zeros(n * m * p)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    val = np.dot(U[u_idx], V[v_idx] * W[w_idx])
                    tensor_errors[idx] = val - 1.0
                    idx += 1
                    
        # Snapping errors
        snap_errors = np.sin(np.pi * 2.0 * params)
        
        # Combine
        # Scale snap errors by weight
        combined = np.concatenate([tensor_errors, snap_errors * snap_weight])
        return combined

    from scipy.optimize import least_squares

    for R in target_ranks:
        if time.time() > deadline:
            break
            
        total_vars = R * (num_u + num_v + num_w)
        
        max_restarts = 50
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
                
            np.random.seed(1000 + restart)
            x0 = np.random.randn(total_vars) * 0.5
            
            try:
                # Stage 1: Solve tensor equations
                res1 = least_squares(error_func, x0, args=(R,), method='trf', max_nfev=2000)
                
                if res1.cost < 1e-10:
                    # Stage 2: Snap to half-integers
                    # Use a moderate weight to encourage snapping without breaking tensor solution
                    # Since res1 is a solution, we start from res1.x
                    snap_weight = 10.0
                    res2 = least_squares(combined_error_func, res1.x, args=(R, snap_weight), method='trf', max_nfev=2000)
                    
                    # Check if res2 is still a good tensor solution
                    # The combined cost includes tensor errors.
                    # We need to check the tensor part specifically.
                    # Re-evaluate tensor error for res2.x
                    U2 = res2.x[:R*num_u].reshape(R, num_u).T
                    V2 = res2.x[R*num_u:R*(num_u+num_v)].reshape(R, num_v).T
                    W2 = res2.x[R*(num_u+num_v):].reshape(R, num_w).T
                    
                    # Quick check: max tensor error
                    max_err = 0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                u_idx = i * m + j
                                v_idx = j * p + k
                                w_idx = k * n + i
                                val = np.dot(U2[u_idx], V2[v_idx] * W2[w_idx])
                                err = abs(val - 1.0)
                                if err > max_err:
                                    max_err = err
                                    
                    if max_err < 1e-9:
                        # Snap to half-integers
                        U_snap = np.round(U2 * 2) / 2
                        V_snap = np.round(V2 * 2) / 2
                        W_snap = np.round(W2 * 2) / 2
                        
                        # Final verification
                        # Re-calculate error with snapped values
                        max_err_snap = 0
                        for i in range(n):
                            for j in range(m):
                                for k in range(p):
                                    u_idx = i * m + j
                                    v_idx = j * p + k
                                    w_idx = k * n + i
                                    val = np.dot(U_snap[u_idx], V_snap[v_idx] * W_snap[w_idx])
                                    err = abs(val - 1.0)
                                    if err > max_err_snap:
                                        max_err_snap = err
                                        
                        if max_err_snap < 1e-9:
                            return {
                                "U": U_snap.tolist(),
                                "V": V_snap.tolist(),
                                "W": W_snap.tolist()
                            }
            except Exception as e:
                continue

    # Fallback
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
