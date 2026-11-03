import numpy as np
from scipy.optimize import least_squares
import time

DIMENSIONS = [3, 5, 7]

def construct():
    n, m, p = DIMENSIONS
    target_rank = n * m * p
    
    # Target tensor T (flattened)
    # T[i, j, k] = 1 for all i, j, k
    # We represent the decomposition as:
    # T[i, j, k] = sum_c U[i, j, c] * V[j, k, c] * W[k, i, c]
    # Note: The indices for W are k, i (since output is C[k*n + i] in the prompt's convention? 
    # Prompt says: W[k*n+i, r]. So W index is (k, i).
    # U index is (i, j) -> U[i*m + j, r]
    # V index is (j, k) -> V[j*p + k, r]
    
    # Let's define the function to compute the tensor from U, V, W
    def compute_tensor(U, V, W, r):
        # U: (n*m, r)
        # V: (m*p, r)
        # W: (p*n, r)
        # Reshape U to (n, m, r), V to (m, p, r), W to (p, n, r)
        U_3d = U.reshape(n, m, r)
        V_3d = V.reshape(m, p, r)
        W_3d = W.reshape(p, n, r)
        
        # Sum over r: U[i,j,c] * V[j,k,c] * W[k,i,c]
        # Result shape (n, p) ... wait.
        # The tensor T is 3D (n, m, p).
        # The product of U, V, W for a fixed c is a 3D tensor U_c * V_c * W_c.
        # (U_c * V_c * W_c)[i, j, k] = U[i, j, c] * V[j, k, c] * W[k, i, c]
        # We want to sum this over c.
        
        # Efficient computation:
        # For each c, we form the tensor.
        # This is slow if done in pure python loops.
        # We can use einsum or matmul tricks.
        
        # T[i, j, k] = sum_c U[i, j, c] V[j, k, c] W[k, i, c]
        
        # Let's try to compute the residual directly.
        # We want T[i, j, k] - sum_c ... = 0
        # T is all 1s.
        
        # We can compute the "predicted" tensor P[i, j, k]
        # and return P - 1.
        
        # Using einsum:
        # U: (n, m, r)
        # V: (m, p, r)
        # W: (p, n, r)
        
        # We need P[i, j, k] = sum_c U[i, j, c] V[j, k, c] W[k, i, c]
        
        # This is a 3rd order tensor contraction.
        # We can do it in two steps.
        # First, contract U and V over m? No, m is an index in the output.
        
        # Let's use a loop over c if r is small, but r is ~100.
        # 100 iterations of einsum on small arrays is fast.
        
        P = np.zeros((n, m, p))
        for c in range(r):
            # u_c: (n, m)
            # v_c: (m, p)
            # w_c: (p, n)
            
            # term[i, j, k] = u_c[i, j] * v_c[j, k] * w_c[k, i]
            # We can compute this as:
            # temp[i, j, k] = u_c[i, j] * v_c[j, k]
            # Then multiply by w_c[k, i]
            
            # u_c is (n, m). v_c is (m, p).
            # u_c[:, :, None] * v_c[None, :, :] -> (n, m, p)
            # Then multiply by w_c which is (p, n).
            # w_c needs to be reshaped to (n, 1, p) or similar to broadcast.
            # w_c[k, i] -> we need index i, j, k.
            # w_c is indexed by k, i.
            # So w_c.T is (n, p).
            # We need to multiply P[i, j, k] by w_c[k, i].
            # w_c[k, i] can be viewed as a matrix of shape (p, n).
            # To broadcast with (n, m, p), we need shape (n, 1, p).
            # w_c.T has shape (n, p).
            # w_c.T[:, None, :] has shape (n, 1, p).
            
            term = u_c[:, :, None] * v_c[None, :, :]
            term = term * w_c.T[:, None, :]
            P += term
            
        return P - 1.0

    # Optimization function
    def objective(params, r):
        # params: flattened U, V, W
        size_u = n * m * r
        size_v = m * p * r
        size_w = p * n * r
        
        U = params[:size_u].reshape(n * m, r)
        V = params[size_u:size_u + size_v].reshape(m * p, r)
        W = params[size_u + size_v:].reshape(p * n, r)
        
        # Compute residual
        # To speed up, we can compute P - 1
        # We need U_c, V_c, W_c
        # U is (n*m, r) -> (n, m, r)
        U_3d = U.reshape(n, m, r)
        V_3d = V.reshape(m, p, r)
        W_3d = W.reshape(p, n, r)
        
        # We can compute the sum over c more efficiently?
        # P[i, j, k] = sum_c U[i, j, c] V[j, k, c] W[k, i, c]
        
        # Let's use the loop for now, it's simple.
        P = np.zeros((n, m, p))
        for c in range(r):
            u_c = U_3d[:, :, c] # (n, m)
            v_c = V_3d[:, :, c] # (m, p)
            w_c = W_3d[:, :, c] # (p, n)
            
            term = u_c[:, :, None] * v_c[None, :, :]
            term = term * w_c.T[:, None, :]
            P += term
            
        return (P - 1.0).ravel()

    # Try to find lower rank
    best_result = None
    start_time = time.time()
    
    # We try ranks from 104 down to 100
    # If time is running out, we stop.
    
    for r in range(target_rank - 1, target_rank - 5, -1):
        if time.time() - start_time > 100:
            break
            
        # Try a few random initializations
        for attempt in range(10):
            if time.time() - start_time > 110:
                break
                
            np.random.seed(attempt)
            # Initialize with small random values
            params = np.random.randn(n * m * r + m * p * r + p * n * r) * 0.1
            
            try:
                res = least_squares(objective, params, args=(r,), method='lm', max_nfev=1000)
                if res.success or res.cost < 1e-6:
                    # Check if we can round to half-integers
                    size_u = n * m * r
                    size_v = m * p * r
                    size_w = p * n * r
                    
                    U_opt = res.x[:size_u].reshape(n * m, r)
                    V_opt = res.x[size_u:size_u + size_v].reshape(m * p, r)
                    W_opt = res.x[size_u + size_v:].reshape(p * n, r)
                    
                    # Scale to check for integers/half-integers
                    # The solution is not unique (scaling). 
                    # We can try to find a scaling factor for each column c such that entries become integers/half-integers.
                    # Or just check if 2 * U_opt is close to integers.
                    
                    # If the solution is valid, we can return it.
                    # But we need to ensure it's exact.
                    # Let's check the residual with the rounded values.
                    
                    # Try rounding to nearest half-integer
                    U_round = np.round(U_opt * 2) / 2
                    V_round = np.round(V_opt * 2) / 2
                    W_round = np.round(W_opt * 2) / 2
                    
                    # Check exactness
                    U_r = U_round.reshape(n, m, r)
                    V_r = V_round.reshape(m, p, r)
                    W_r = W_round.reshape(p, n, r)
                    
                    P_check = np.zeros((n, m, p))
                    for c in range(r):
                        u_c = U_r[:, :, c]
                        v_c = V_r[:, :, c]
                        w_c = W_r[:, :, c]
                        term = u_c[:, :, None] * v_c[None, :, :]
                        term = term * w_c.T[:, None, :]
                        P_check += term
                    
                    if np.allclose(P_check, 1.0, atol=1e-9):
                        # Found a valid decomposition!
                        best_result = {"U": U_round.tolist(), "V": V_round.tolist(), "W": W_round.tolist()}
                        break
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
