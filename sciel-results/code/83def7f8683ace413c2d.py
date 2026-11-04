# EVOLVE-BLOCK-START
import time
import numpy as np
import scipy.optimize

DIMENSIONS = [3, 3, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    
    def verify(U, V, W):
        rank = len(U[0])
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U_arr[i*m+j] * V_arr[j*p+k] * W_arr[k*n+i])
                    if abs(val - 1.0) > 1e-6:
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

    def find_3x3x3_rank19():
        """Find a rank 19 decomposition for 3x3x3 matrix multiplication."""
        n3, m3, p3 = 3, 3, 3
        rank3 = 19
        
        # Flatten indices
        # U: (9, 19), V: (9, 19), W: (9, 19)
        
        def compute_residual(params):
            # params is a flat array of size 9*19*3 = 513
            U_flat = params[:9*rank3]
            V_flat = params[9*rank3:18*rank3]
            W_flat = params[18*rank3:27*rank3]
            
            U = U_flat.reshape(9, rank3)
            V = V_flat.reshape(9, rank3)
            W = W_flat.reshape(9, rank3)
            
            residual = np.zeros((n3, m3, p3))
            for i in range(n3):
                for j in range(m3):
                    for k in range(p3):
                        u_idx = i * m3 + j
                        v_idx = j * p3 + k
                        w_idx = k * n3 + i
                        residual[i, j, k] = np.dot(U[u_idx], V[v_idx] * W[w_idx])
            return residual - 1.0

        best_err = float('inf')
        best_params = None
        
        for restart in range(10):
            if time.time() > deadline - 20:
                break
                
            np.random.seed(42 + restart)
            # Initialize with small random values
            init_params = np.random.randn(27 * rank3) * 0.1
            
            # Use L-BFGS-B
            try:
                res = scipy.optimize.minimize(
                    lambda x: np.sum(compute_residual(x)**2),
                    init_params,
                    method='L-BFGS-B',
                    options={'maxiter': 5000, 'ftol': 1e-10}
                )
                if res.success or res.fun < best_err:
                    if res.fun < best_err:
                        best_err = res.fun
                        best_params = res.x
                        if best_err < 1e-6:
                            break
            except Exception:
                pass
        
        if best_params is not None and best_err < 1e-4:
            U_flat = best_params[:9*rank3]
            V_flat = best_params[9*rank3:18*rank3]
            W_flat = best_params[18*rank3:27*rank3]
            
            U = U_flat.reshape(9, rank3)
            V = V_flat.reshape(9, rank3)
            W = W_flat.reshape(9, rank3)
            
            # Round to nearest integer or half-integer
            # Strassen uses integers
            for scale in [1, 2]:
                U_round = np.round(U * scale) / scale
                V_round = np.round(V * scale) / scale
                W_round = np.round(W * scale) / scale
                
                res = compute_residual(np.concatenate([U_round.flatten(), V_round.flatten(), W_round.flatten()]))
                if np.allclose(res, 0, atol=1e-6):
                    return U_round, V_round, W_round
            
            # If rounding failed, return the numerical solution (might fail verification if strict)
            # But the task requires integer/half-integer.
            # So we must return None if we can't round.
            return None
        
        return None

    # Try to find rank 38 solution for 3x3x6
    U333, V333, W333 = find_3x3x3_rank19()
    
    if U333 is not None:
        rank3 = 19
        rank6 = 2 * rank3
        
        U6 = np.zeros((9, rank6))
        V6 = np.zeros((18, rank6))
        W6 = np.zeros((18, rank6))
        
        # U is same for both halves
        U6[:, :rank3] = U333
        U6[:, rank3:] = U333
        
        # V: first 3 cols of k (0-2) use first half, last 3 cols (3-5) use second half
        # V333 is (9, 19) indexed by j*3+k
        # V6 is (18, 38) indexed by j*6+k
        for j in range(3):
            for k in range(3):
                # First half
                V6[j*6 + k, :rank3] = V333[j*3 + k, :]
                # Second half
                V6[j*6 + (k+3), rank3:] = V333[j*3 + k, :]
        
        # W: first 3 rows of k (0-2) use first half, last 3 rows (3-5) use second half
        # W333 is (9, 19) indexed by k*3+i
        # W6 is (18, 38) indexed by k*3+i
        for k in range(3):
            for i in range(3):
                # First half
                W6[k*3 + i, :rank3] = W333[k*3 + i, :]
                # Second half
                W6[(k+3)*3 + i, rank3:] = W333[k*3 + i, :]
        
        U_list = U6.tolist()
        V_list = V6.tolist()
        W_list = W6.tolist()
        
        if verify(U_list, V_list, W_list):
            return {"U": U_list, "V": V_list, "W": W_list}
        
    # Fallback
    return make_schoolbook()
# EVOLVE-BLOCK-END
