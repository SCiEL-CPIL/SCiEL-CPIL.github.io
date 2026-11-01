import numpy as np
import time
from scipy.optimize import least_squares

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    # Target ranks to try
    target_ranks = [41] # Try to beat 42
    
    best_result = None
    best_rank = n * m * p # 42
    
    # Precompute indices for the error function
    # i, j, k, i_prime, k_prime
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i (prompt says W[k*n+i, r])
    
    def error_func(params, R):
        # params is flat array of size R * (n*m + m*p + p*n)
        num_u = n * m
        num_v = m * p
        num_w = p * n
        
        U = params[:R*num_u].reshape(R, num_u).T
        V = params[R*num_u:R*(num_u+num_v)].reshape(R, num_v).T
        W = params[R*(num_u+num_v):].reshape(R, num_w).T
        
        errors = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Term A[i,j] B[j,k]
                    # Should contribute to C[i,k]
                    # W index for C[i,k] is k*n + i
                    w_idx = k * n + i
                    
                    # Sum over r
                    # U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    u_idx = i * m + j
                    v_idx = j * p + k
                    
                    # Check if this term contributes to C[i', k']
                    # It should be 1 if i'=i and k'=k, else 0
                    # But the equation is for ALL i', k'
                    # So for fixed i,j,k, we check all i', k'
                    
                    for ip in range(n):
                        for kp in range(p):
                            target = 1.0 if (ip == i and kp == k) else 0.0
                            # Calculate sum
                            # np.dot(U[:, u_idx], V[:, v_idx] * W[:, kp*n+ip])
                            val = np.dot(U[:, u_idx], V[:, v_idx] * W[:, kp*n + ip])
                            errors.append(val - target)
                            
        return np.array(errors)

    found_solution = False
    for R in target_ranks:
        if time.time() > deadline:
            break
            
        num_u = n * m
        num_v = m * p
        num_w = p * n
        total_vars = R * (num_u + num_v + num_w)
        
        # Try a few random initializations
        for restart in range(5):
            if time.time() > deadline:
                break
                
            np.random.seed(42 + restart)
            # Small random init
            x0 = np.random.randn(total_vars) * 0.1
            
            try:
                res = least_squares(error_func, x0, args=(R,), method='lm', max_nfev=1000)
            except Exception:
                continue
                
            if res.cost < 1e-12:
                # Check if we can snap to half-integers
                num_u = n * m
                num_v = m * p
                num_w = p * n
                
                U = res.x[:R*num_u].reshape(R, num_u).T
                V = res.x[R*num_u:R*(num_u+num_v)].reshape(R, num_v).T
                W = res.x[R*(num_u+num_v):].reshape(R, num_w).T
                
                # Snap to nearest half-integer
                U_snap = np.round(U * 2) / 2
                V_snap = np.round(V * 2) / 2
                W_snap = np.round(W * 2) / 2
                
                # Verify snapped solution
                if np.allclose(U_snap, U, atol=1e-9) and np.allclose(V_snap, V, atol=1e-9) and np.allclose(W_snap, W, atol=1e-9):
                     # Re-verify with snapped values
                     err = error_func(np.concatenate([U_snap.flatten(), V_snap.flatten(), W_snap.flatten()]), R)
                     if np.max(np.abs(err)) < 1e-9:
                         # Success
                         return {
                             "U": U_snap.tolist(),
                             "V": V_snap.tolist(),
                             "W": W_snap.tolist()
                         }
                else:
                     # If snap is bad, maybe return float if exact enough?
                     # Prompt says "integer/half-integer entries".
                     # If we can't snap, we might not be able to return this.
                     # But if the float solution is exact, maybe we can find a rational representation?
                     # For now, if snap fails, we might discard or try to use float if allowed?
                     # "exactly decompose" implies exactness.
                     # If float is 0.4999999, it's effectively 0.5.
                     # Let's try to return the float values if they are very close to half-integers.
                     if np.max(np.abs(U - U_snap)) < 1e-8 and np.max(np.abs(V - V_snap)) < 1e-8 and np.max(np.abs(W - W_snap)) < 1e-8:
                          return {
                             "U": U_snap.tolist(),
                             "V": V_snap.tolist(),
                             "W": W_snap.tolist()
                         }
                         
    # Fallback to schoolbook
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
