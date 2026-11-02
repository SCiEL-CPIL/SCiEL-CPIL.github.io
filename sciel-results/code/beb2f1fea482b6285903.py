import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [4, 4, 5]

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 4x4 * 4x5 matrix multiplication
    using continuous optimization (L-BFGS-B) followed by rational reconstruction.
    """
    n, m, p = DIMENSIONS  # n=4, m=4, p=5
    start_time = time.time()
    deadline = 145  # Safety margin for 150s limit

    # Pre-compute index mappings for speed
    # U[i*m + j, r], V[j*p + k, r], W[k*n + i, r]
    # Constraint: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    
    u_indices = []
    v_indices = []
    w_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_indices.append(i * m + j)
                v_indices.append(j * p + k)
                w_indices.append(k * n + i)
                
    u_indices = np.array(u_indices)
    v_indices = np.array(v_indices)
    w_indices = np.array(w_indices)
    
    def compute_residuals(params):
        """
        Compute the vector of residuals for the tensor identity.
        params: flat array of length 3 * (dim * R)
        Returns: array of shape (n*m*p,) containing (sum_r U*V*W - 1) for each (i,j,k)
        """
        # Unpack params
        # Sizes: U is n*m*R, V is m*p*R, W is p*n*R
        size_U = n * m * params_R
        size_V = m * p * params_R
        size_W = p * n * params_R
        
        U_flat = params[:size_U]
        V_flat = params[size_U:size_U + size_V]
        W_flat = params[size_U + size_V:]
        
        # Reshape
        U = U_flat.reshape((n * m, params_R))
        V = V_flat.reshape((m * p, params_R))
        W = W_flat.reshape((p * n, params_R))
        
        # Extract rows
        U_rows = U[u_indices, :]
        V_rows = V[v_indices, :]
        W_rows = W[w_indices, :]
        
        # Product and sum
        prod = U_rows * V_rows * W_rows
        vals = np.sum(prod, axis=1)
        
        return vals - 1.0

    def objective(params):
        """
        Objective function: Sum of squared residuals.
        """
        res = compute_residuals(params)
        return np.dot(res, res)

    def gradient(params):
        """
        Gradient of the objective function.
        Using chain rule and vectorization for speed.
        """
        res = compute_residuals(params)
        
        size_U = n * m * params_R
        size_V = m * p * params_R
        size_W = p * n * params_R
        
        U_flat = params[:size_U]
        V_flat = params[size_U:size_U + size_V]
        W_flat = params[size_U + size_V:]
        
        U = U_flat.reshape((n * m, params_R))
        V = V_flat.reshape((m * p, params_R))
        W = W_flat.reshape((p * n, params_R))
        
        U_rows = U[u_indices, :]
        V_rows = V[v_indices, :]
        W_rows = W[w_indices, :]
        
        # Gradient w.r.t U: 2 * sum_k (res_k * V_k * W_k)
        # We need to accumulate gradients for each element in U, V, W
        # U_grad[i*m+j, r] = 2 * sum_{(i,j,k)} res[i,j,k] * V[j*p+k, r] * W[k*n+i, r]
        
        # Vectorized accumulation
        # res shape: (n*m*p,)
        # V_rows shape: (n*m*p, R)
        # W_rows shape: (n*m*p, R)
        
        # Term for U gradient: res[:, None] * V_rows * W_rows -> shape (n*m*p, R)
        # We need to sum over the (n*m*p) axis but map back to (n*m, R)
        # The index in U is i*m+j. The index in res is (i, j, k).
        # For a fixed (i, j), there are p values of k.
        # So we reshape res to (n, m, p) and sum over p.
        
        res_reshaped = res.reshape((n, m, p))
        
        # V_rows reshaped: (n, m, p, R) ? No, V_rows is indexed by (j, k).
        # Let's stick to the flat index approach for accumulation to avoid complex reshaping errors.
        # Alternative: use einsum or manual loops if R is small? R is ~50-80. n*m*p=80.
        # 80 * 80 * 80 operations per gradient eval is trivial.
        
        # Let's do it with numpy broadcasting carefully.
        # U_grad[r, col] where col is the column in U (i*m+j)
        # Actually, let's just compute the gradients by reconstructing the tensors.
        
        # grad_U[i*m+j, r] = 2 * sum_k res[i,j,k] * V[j*p+k, r] * W[k*n+i, r]
        # grad_V[j*p+k, r] = 2 * sum_i res[i,j,k] * U[i*m+j, r] * W[k*n+i, r]
        # grad_W[k*n+i, r] = 2 * sum_j res[i,j,k] * U[i*m+j, r] * V[j*p+k, r]
        
        # This is easier to do with loops over i, j, k if n,m,p are small, but let's vectorize.
        # Since n=4, m=4, p=5, total 80 terms.
        
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        # To vectorize, we can use the fact that we are summing over k for U.
        # But the indices are mixed.
        # Let's just use a loop over the 80 constraints. It's fast enough.
        # 80 iterations * vector ops of size R.
        
        for idx in range(n * m * p):
            i = idx // (m * p)
            rem = idx % (m * p)
            j = rem // p
            k = rem % p
            
            r_val = res[idx]
            
            u_idx = i * m + j
            v_idx = j * p + k
            w_idx = k * n + i
            
            # grad_U[u_idx, :] += 2 * r_val * V[v_idx, :] * W[w_idx, :]
            grad_U[u_idx, :] += 2.0 * r_val * V[v_idx, :] * W[w_idx, :]
            grad_V[v_idx, :] += 2.0 * r_val * U[u_idx, :] * W[w_idx, :]
            grad_W[w_idx, :] += 2.0 * r_val * U[u_idx, :] * V[v_idx, :]
            
        return np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])

    def try_rank(R, max_iter=500, time_limit=10):
        """
        Attempt to find a decomposition of rank R.
        Returns (U, V, W) if successful, else None.
        """
        nonlocal params_R
        params_R = R
        
        total_size = n * m * R + m * p * R + p * n * R
        
        # Initialize with small random values
        np.random.seed(42) # Fixed seed for reproducibility in this run, or random?
        # Random seed helps avoid getting stuck in the same bad local minimum every time
        # But for deterministic output, we might want fixed. 
        # However, the task allows randomness. Let's use a random seed but check time.
        # Actually, better to try multiple random inits if we have time.
        
        best_params = None
        best_obj = float('inf')
        
        # Number of restarts based on time
        # Each L-BFGS-B call might take 1-5 seconds.
        # Let's try 2-3 restarts.
        
        for restart in range(3):
            if time.time() - start_time > deadline - 5:
                break
                
            x0 = np.random.randn(total_size) * 0.5 # Small init
            
            # Scale initial values? 
            # If we scale too small, gradient might be tiny.
            # If we scale too large, it might diverge.
            # Let's try 1.0
            x0 = np.random.randn(total_size)
            
            # Add a tiny bit of noise to break symmetry?
            # x0 += np.random.randn(total_size) * 0.01
            
            start_opt = time.time()
            
            # Use L-BFGS-B
            # maxiter: limit iterations to prevent infinite loops
            # maxfun: limit function evaluations
            try:
                res_opt = minimize(
                    objective, 
                    x0, 
                    method='L-BFGS-B', 
                    jac=gradient,
                    options={
                        'maxiter': 200, 
                        'maxfun': 500,
                        'gtol': 1e-8
                    }
                )
            except Exception:
                continue
                
            if time.time() - start_time > deadline - 5:
                break
                
            obj_val = res_opt.fun
            if obj_val < best_obj:
                best_obj = obj_val
                best_params = res_opt.x
                
            # If we found a very small residual, we might have a solution
            # But we need to check if it snaps to integers/half-integers.
            
        if best_params is None:
            return None
            
        # Check if the best solution is close to integer/half-integer values
        # And if the residual is close to 0.
        
        # Unpack best_params
        size_U = n * m * R
        size_V = m * p * R
        size_W = p * n * R
        
        U_flat = best_params[:size_U]
        V_flat = best_params[size_U:size_U + size_V]
        W_flat = best_params[size_U + size_V:]
        
        U = U_flat.reshape((n * m, R))
        V = V_flat.reshape((m * p, R))
        W = W_flat.reshape((p * n, R))
        
        # Rational Reconstruction: Snap to nearest half-integer
        # The problem states entries should be integer/half-integer.
        # This implies values are in Z + 0.5 Z.
        # So we multiply by 2, round to nearest integer, divide by 2.
        
        U_int = np.round(U * 2) / 2.0
        V_int = np.round(V * 2) / 2.0
        W_int = np.round(W * 2) / 2.0
        
        # Verify exactness
        # Compute residual with snapped values
        U_rows = U_int[u_indices, :]
        V_rows = V_int[v_indices, :]
        W_rows = W_int[w_indices, :]
        
        prod = U_rows * V_rows * W_rows
        vals = np.sum(prod, axis=1)
        errors = vals - 1.0
        
        # Since entries are half-integers, products are multiples of 0.125?
        # No, half-integers are k/2. Product is (k1/2)(k2/2)(k3/2) = k1k2k3/8.
        # Sum is multiple of 1/8.
        # So error is multiple of 1/8.
        # If error is 0, it's exact.
        
        max_err = np.max(np.abs(errors))
        
        if max_err < 1e-9:
            return U_int, V_int, W_int
            
        return None

    best_solution = None
    best_rank = float('inf')
    
    # Try ranks from low to high? Or high to low?
    # Usually, lower rank is harder to find.
    # Let's start from a reasonable estimate.
    # Schoolbook is 80.
    # Strassen-like for 4x4x4 is 49.
    # For 4x4x5, maybe around 50-60.
    
    # We will try a range of ranks.
    # To maximize chance of finding a low rank, we should try low ranks first,
    # but give them less time if they fail?
    # Or try a few specific promising ranks.
    
    # Let's try ranks: 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80
    # We'll iterate and break if we find one.
    
    # To save time, let's prioritize a few.
    # If we find rank 50, great. If not, 55, etc.
    
    ranks_to_try = list(range(50, 81, 2)) # 50, 52, ..., 80
    
    for target_rank in ranks_to_try:
        if time.time() - start_time > deadline - 10:
            break
            
        # Allocate time for this rank
        # If we are near the end, spend less time.
        time_budget = min(15, deadline - (time.time() - start_time) - 5)
        if time_budget < 2:
            break
            
        # Call try_rank
        # The internal logic of try_rank should respect the global deadline.
        
        result = try_rank(target_rank)
        
        if result is not None:
            U, V, W = result
            best_solution = {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
            best_rank = target_rank
            break # Found a solution, stop
            
    # Fallback: Schoolbook Rank 80
    if best_solution is None:
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
        best_solution = {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
        best_rank = rank

    return best_solution

if __name__ == "__main__":
    # Quick test
    res = construct()
    print("Rank:", len(res['U'][0]) if res['U'] else 0)
