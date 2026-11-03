"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130.0

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

    # Precompute index arrays for efficient verification
    i_idx = np.repeat(np.arange(n), m * p)
    j_idx = np.tile(np.repeat(np.arange(m), p), n)
    k_idx = np.tile(np.arange(p), n * m)
    u_idx = i_idx * m + j_idx
    v_idx = j_idx * p + k_idx
    w_idx = k_idx * n + i_idx

    def verify(U, V, W, r):
        """Check if U, V, W form a valid tensor decomposition. Returns max error."""
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        
        u_vals = Ua[u_idx, :]  # (n*m*p, r)
        v_vals = Va[v_idx, :]  # (n*m*p, r)
        w_vals = Wa[w_idx, :]  # (n*m*p, r)
        
        prod = u_vals * v_vals * w_vals
        result = prod.sum(axis=1)
        
        return np.max(np.abs(result - 1.0))

    def refine_vectorized(U, V, W, r, max_iter=200):
        """
        Refine using vectorized gradient descent with adaptive learning rate.
        """
        Ua = np.array(U, dtype=float).copy()
        Va = np.array(V, dtype=float).copy()
        Wa = np.array(W, dtype=float).copy()
        
        # Momentum buffers
        mU = np.zeros_like(Ua)
        mV = np.zeros_like(Va)
        mW = np.zeros_like(Wa)
        beta = 0.9
        lr = 0.1
        best_err = np.inf
        
        for it in range(max_iter):
            if time.time() > deadline - 5:
                break
            
            # Compute current values
            u_vals = Ua[u_idx, :]  # (N, r)
            v_vals = Va[v_idx, :]  # (N, r)
            w_vals = Wa[w_idx, :]  # (N, r)
            
            # Reshape to 4D for easier gradient calculation
            u_4d = u_vals.reshape(n, m, p, r)
            v_4d = v_vals.reshape(n, m, p, r)
            w_4d = w_vals.reshape(n, m, p, r)
            
            prod = u_4d * v_4d * w_4d
            result = prod.sum(axis=3) # (n, m, p)
            error = result - 1.0
            err_norm = np.linalg.norm(error)
            
            if err_norm < 1e-8:
                break
            
            if err_norm < best_err:
                best_err = err_norm
            
            # Gradients
            err_4d = error[:, :, :, np.newaxis]
            
            # grad_U[i, j, c] = sum_k 2 * error[i, j, k] * v_4d[i, j, k, c] * w_4d[i, j, k, c]
            grad_U_4d = 2 * err_4d * v_4d * w_4d
            grad_U = np.sum(grad_U_4d, axis=2).reshape(n * m, r)
            
            # grad_V[j, k, c] = sum_i 2 * error[i, j, k] * u_4d[i, j, k, c] * w_4d[i, j, k, c]
            grad_V_4d = 2 * err_4d * u_4d * w_4d
            grad_V_sum = np.sum(grad_V_4d, axis=0)
            grad_V = grad_V_sum.reshape(m * p, r)
            
            # grad_W[k, i, c] = sum_j 2 * error[i, j, k] * u_4d[i, j, k, c] * v_4d[i, j, k, c]
            grad_W_4d = 2 * err_4d * u_4d * v_4d
            grad_W_sum = np.sum(grad_W_4d, axis=1) # Shape (n, p, r)
            grad_W_sum = np.transpose(grad_W_sum, (1, 0, 2))
            grad_W = grad_W_sum.reshape(p * n, r)
            
            # Update with momentum
            mU = beta * mU + (1 - beta) * grad_U
            mV = beta * mV + (1 - beta) * grad_V
            mW = beta * mW + (1 - beta) * grad_W
            
            Ua -= lr * mU
            Va -= lr * mV
            Wa -= lr * mW
            
            # Adaptive learning rate: reduce if error increases
            new_err = verify(Ua.tolist(), Va.tolist(), Wa.tolist(), r)
            if new_err > best_err * 1.5:
                lr *= 0.5
            elif new_err < best_err * 0.5:
                lr *= 1.1
                lr = min(lr, 0.5)
        
        return Ua, Va, Wa

    def linear_code_search(r, max_trials=5):
        """
        Fix U to a structured low-rank basis and solve for V and W.
        The idea: if we fix U to have a specific structure (e.g., random Gaussian
        or Kronecker products), we can set up a linear system for V and W.
        
        However, the tensor equation is:
        sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
        
        This is nonlinear in V and W even if U is fixed. So we can't directly
        solve a linear system. Instead, we use the fixed U to guide the search.
        
        Alternative: Fix U to be a random matrix, then use alternating least squares
        or gradient descent on V and W only, keeping U fixed.
        """
        for trial in range(max_trials):
            if time.time() > deadline - 10:
                return None
            
            np.random.seed(trial * 7919 + r * 131)
            
            # Fix U to a random Gaussian matrix
            U = np.random.randn(n * m, r)
            
            # Initialize V and W randomly
            V = np.random.randn(m * p, r) * 0.5
            W = np.random.randn(p * n, r) * 0.5
            
            # Refine only V and W, keeping U fixed
            for it in range(100):
                if time.time() > deadline - 10:
                    break
                
                u_vals = U[u_idx, :]
                v_vals = V[v_idx, :]
                w_vals = W[w_idx, :]
                
                u_4d = u_vals.reshape(n, m, p, r)
                v_4d = v_vals.reshape(n, m, p, r)
                w_4d = w_vals.reshape(n, m, p, r)
                
                prod = u_4d * v_4d * w_4d
                result = prod.sum(axis=3)
                error = result - 1.0
                err_norm = np.linalg.norm(error)
                
                if err_norm < 1e-8:
                    break
                
                err_4d = error[:, :, :, np.newaxis]
                
                # Gradient for V
                grad_V_4d = 2 * err_4d * u_4d * w_4d
                grad_V_sum = np.sum(grad_V_4d, axis=0)
                grad_V = grad_V_sum.reshape(m * p, r)
                
                # Gradient for W
                grad_W_4d = 2 * err_4d * u_4d * v_4d
                grad_W_sum = np.sum(grad_W_4d, axis=1)
                grad_W_sum = np.transpose(grad_W_sum, (1, 0, 2))
                grad_W = grad_W_sum.reshape(p * n, r)
                
                V -= 0.1 * grad_V
                W -= 0.1 * grad_W
            
            err = verify(U.tolist(), V.tolist(), W.tolist(), r)
            if err < 1e-6:
                # Try rounding
                U_round = np.round(U * 2) / 2
                V_round = np.round(V * 2) / 2
                W_round = np.round(W * 2) / 2
                err_round = verify(U_round.tolist(), V_round.tolist(), W_round.tolist(), r)
                if err_round < 1e-9:
                    return {"U": U_round.tolist(), "V": V_round.tolist(), "W": W_round.tolist()}
        
        return None

    best_result = make_schoolbook()
    best_rank = n * m * p

    # Try structured linear code search for various ranks
    candidate_ranks = list(range(95, 112))
    candidate_ranks = sorted(set(candidate_ranks), key=lambda x: -x)

    found_improvement = False

    for r in candidate_ranks:
        if time.time() > deadline - 15:
            break
        if found_improvement:
            break

        result = linear_code_search(r, max_trials=3)
        if result is not None:
            best_rank = r
            best_result = result
            found_improvement = True
            break

    # Also try random search + refinement as a fallback
    if not found_improvement:
        for r in range(100, 112):
            if time.time() > deadline - 10:
                break
            if found_improvement:
                break
            
            for trial in range(2):
                if time.time() > deadline - 5:
                    break
                
                np.random.seed(trial * 10000 + r * 100)
                U = np.random.randn(n * m, r) * 0.3
                V = np.random.randn(m * p, r) * 0.3
                W = np.random.randn(p * n, r) * 0.3
                
                U_ref, V_ref, W_ref = refine_vectorized(U, V, W, r, max_iter=100)
                
                err = verify(U_ref.tolist(), V_ref.tolist(), W_ref.tolist(), r)
                if err < 1e-6:
                    U_round = np.round(U_ref * 2) / 2
                    V_round = np.round(V_ref * 2) / 2
                    W_round = np.round(W_ref * 2) / 2
                    err_round = verify(U_round.tolist(), V_round.tolist(), W_round.tolist(), r)
                    if err_round < 1e-9:
                        best_rank = r
                        best_result = {
                            "U": U_round.tolist(),
                            "V": V_round.tolist(),
                            "W": W_round.tolist()
                        }
                        found_improvement = True
                        break

    return best_result
# EVOLVE-BLOCK-END
