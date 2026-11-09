"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 145.0  # 5 seconds safety margin
    
    n, m, p = DIMENSIONS
    
    # 1. Build the trivial (schoolbook) decomposition as the baseline.
    # Rank R = n*m*p.
    # This is guaranteed to be a valid decomposition.
    rank_schoolbook = n * m * p
    U0 = np.zeros((n * m, rank_schoolbook))
    V0 = np.zeros((m * p, rank_schoolbook))
    W0 = np.zeros((p * n, rank_schoolbook))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i * m + j, col] = 1.0
                V0[j * p + k, col] = 1.0
                W0[k * n + i, col] = 1.0
                col += 1
                
    def verify(U, V, W, tol=1e-9):
        """Check if the decomposition exactly reconstructs the tensor T[i,j,k]=1."""
        # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # We want T[i,j,k] = 1 for all i, j, k.
        # To be robust, we check the Frobenius norm of the difference.
        # However, checking all 60 entries is fast enough.
        # For speed in search loops, we might check a subset or use vectorized ops.
        
        # Efficient check:
        # Reshape U, V, W to 3D?
        # U is (n*m, R). V is (m*p, R). W is (p*n, R).
        # We want to compute T of shape (n, m, p).
        # T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        # This is a bit tricky to vectorize directly without loops over i,j,k or R.
        # Given R is small (<=60) and n,m,p are small, loops are fine.
        # But for the search, we call this many times.
        
        # Let's try a vectorized approach using einsum or similar.
        # U_flat (n*m, R) -> U_3d (n, m, R)
        # V_flat (m*p, R) -> V_3d (m, p, R)
        # W_flat (p*n, R) -> W_3d (p, n, R)
        
        # T[i, j, k] = sum_r U_3d[i, j, r] * V_3d[j, k, r] * W_3d[k, i, r]
        
        # This is a contraction over r, and indices i, j, k are coupled.
        # It's not a simple dot product.
        
        # Let's stick to the loop for verification in the search if it's too slow,
        # but optimize the inner loop.
        
        # Actually, let's just check the max error.
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                u_row = U[i * m + j]
                for k in range(p):
                    v_row = V[j * p + k]
                    w_row = W[k * n + i]
                    val = np.dot(u_row, v_row * w_row)
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
        return max_err < tol

    best_U = U0
    best_V = V0
    best_W = W0
    best_rank = rank_schoolbook
    
    # 2. Search for lower rank decompositions.
    # We try ranks from (n*m*p - 1) down to 1.
    # For each rank R, we try to solve for U, V, W.
    # This is a non-convex problem. We use a local search / optimization approach.
    
    # Strategy:
    # Fix one matrix (e.g., W) to random values, and solve for U, V?
    # Or use gradient descent on the error surface.
    
    # Given the small size, we can try a simple heuristic:
    # Start with a random initialization for U, V, W of rank R.
    # Use scipy.optimize.minimize or similar? No, standard library + numpy.
    # We can implement a simple gradient descent or use a library if available.
    # The prompt says NumPy and SciPy are available.
    
    import scipy.optimize as opt
    
    def objective(params, R):
        """
        Parameters are flattened U, V, W.
        U: n*m*R
        V: m*p*R
        W: p*n*R
        Total size: (n*m + m*p + p*n) * R
        """
        size_u = n * m * R
        size_v = m * p * R
        
        U_flat = params[:size_u]
        V_flat = params[size_u:size_u + size_v]
        W_flat = params[size_u + size_v:]
        
        U_mat = U_flat.reshape((n * m, R))
        V_mat = V_flat.reshape((m * p, R))
        W_mat = W_flat.reshape((p * n, R))
        
        # Calculate error
        # T[i,j,k] should be 1.
        err = 0.0
        for i in range(n):
            for j in range(m):
                u_row = U_mat[i * m + j]
                for k in range(p):
                    v_row = V_mat[j * p + k]
                    w_row = W_mat[k * n + i]
                    val = np.dot(u_row, v_row * w_row)
                    err += (val - 1.0) ** 2
        return err

    # We iterate R from best_rank - 1 downwards.
    # For each R, we try a few random initializations.
    
    # To save time, we only try a few ranks and few restarts.
    # n=2, m=5, p=6.
    # Ranges: 59 down to maybe 20?
    # Time budget is ~140s.
    
    # Let's estimate cost.
    # One objective call is ~60 * R * (few ops).
    # Minimize might take 100-1000 iterations.
    # So one optimization run might take 1-10s depending on R and convergence.
    # We can afford ~10-50 runs.
    
    # We'll try R from 59 down to 1.
    # But we stop if we find a valid one.
    
    # We use a random seed for reproducibility if needed, but random is fine.
    
    # Optimization method: BFGS or L-BFGS-B.
    # Since we want exact integer/half-integer entries?
    # The prompt says "integer/half-integer entries".
    # The optimization might find real numbers.
    # We need to check if the found solution is close to integer/half-integer.
    # If so, we can snap to the nearest valid value and re-verify.
    # If not, we discard it.
    
    # This constraint makes it harder.
    # However, for small ranks, exact solutions are often integer or simple fractions.
    
    # Let's try the optimization.
    
    # We'll use a time check in the loop.
    
    for R in range(rank_schoolbook - 1, 0, -1):
        if time.time() > deadline - 10:
            break
            
        # Try a few random initializations for this rank R.
        num_restarts = 2
        found = False
        
        for restart in range(num_restarts):
            if time.time() > deadline - 5:
                break
                
            # Random initialization
            # Scale: values around 1.
            U_init = np.random.randn(n * m, R)
            V_init = np.random.randn(m * p, R)
            W_init = np.random.randn(p * n, R)
            
            # Normalize? Maybe not.
            
            params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            # Run optimization
            # Use L-BFGS-B for speed.
            try:
                result = opt.minimize(objective, params_init, args=(R,), method='L-BFGS-B', 
                                      options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-9})
            except Exception:
                continue
                
            if result.fun < 1e-6: # Good enough for a candidate
                # Extract U, V, W
                size_u = n * m * R
                size_v = m * p * R
                U_opt = result.x[:size_u].reshape((n * m, R))
                V_opt = result.x[size_u:size_u + size_v].reshape((m * p, R))
                W_opt = result.x[size_u + size_v:].reshape((p * n, R))
                
                # Check if entries are close to integer or half-integer
                # We can try to snap them.
                # But first, verify the raw solution.
                if verify(U_opt, V_opt, W_opt):
                    # It's a valid real-valued solution.
                    # Now, try to find an exact rational/integer solution.
                    # For now, if it's very close to integers, we can use it.
                    # The prompt says "integer/half-integer entries".
                    # If the optimizer finds 0.999999, we can set to 1.
                    
                    # Let's try to snap to nearest 0.5
                    U_snapped = np.round(U_opt * 2) / 2
                    V_snapped = np.round(V_opt * 2) / 2
                    W_snapped = np.round(W_opt * 2) / 2
                    
                    if verify(U_snapped, V_snapped, W_snapped):
                        # Found a valid solution with rank R!
                        best_U = U_snapped
                        best_V = V_snapped
                        best_W = W_snapped
                        best_rank = R
                        found = True
                        break # Break restart loop
                    
                    # If snapping to 0.5 didn't work, maybe it's integer?
                    U_int = np.round(U_opt)
                    V_int = np.round(V_opt)
                    W_int = np.round(W_opt)
                    
                    if verify(U_int, V_int, W_int):
                        best_U = U_int
                        best_V = V_int
                        best_W = W_int
                        best_rank = R
                        found = True
                        break
                    
                    # If neither, we might still use the real-valued one if the evaluator accepts it?
                    # "integer/half-integer entries" suggests strictness.
                    # Let's stick to snapped versions for safety.
                    
        if found:
            # If we found a lower rank, we can continue searching for even lower ranks?
            # Yes, but we should be careful with time.
            # Let's continue the loop, it will check time.
            pass

    # Convert best solution to lists of lists as required by the interface
    # "factor matrices of shapes ... with integer/half-integer entries"
    # The return type is {'U', 'V', 'W'} with lists.
    
    def to_list(mat):
        return mat.tolist()
        
    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
# EVOLVE-BLOCK-END
