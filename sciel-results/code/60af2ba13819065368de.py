"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
import time
import random
from scipy.optimize import minimize

def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    schoolbook_rank = nm * p
    
    # Generate schoolbook decomposition as fallback
    U_school = np.zeros((nm, schoolbook_rank))
    V_school = np.zeros((mp, schoolbook_rank))
    W_school = np.zeros((pn, schoolbook_rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m + j, col] = 1
                V_school[j*p + k, col] = 1
                W_school[k*n + i, col] = 1
                col += 1
    
    best_decomp = (U_school, V_school, W_school)
    best_rank = schoolbook_rank
    
    # Define the target tensor indices
    # Target is 1 for valid triples (i, j, k)
    # Indexing:
    # U index: i * m + j
    # V index: j * p + k
    # W index: k * n + i
    target_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                target_indices.append((u_idx, v_idx, w_idx))
                
    def error_function(params, r):
        """
        Compute the squared error of the tensor reconstruction.
        params: flattened array of size (nm + mp + pn) * r
        """
        U_flat = params[:nm * r].reshape(nm, r)
        V_flat = params[nm * r : nm * r + mp * r].reshape(mp, r)
        W_flat = params[nm * r + mp * r :].reshape(pn, r)
        
        error = 0.0
        # Check valid indices (should be 1)
        for u_idx, v_idx, w_idx in target_indices:
            val = np.dot(U_flat[u_idx], V_flat[v_idx])
            # Wait, the formula is sum_c U[u,c] V[v,c] W[w,c]
            # This is not a simple dot product of U and V.
            # It is sum_c (U[u,c] * V[v,c]) * W[w,c]
            # Let's compute this efficiently.
            # We can't easily vectorize the triple product for all targets at once without expanding.
            # Given small size, loop is fine.
            s = 0.0
            for c in range(r):
                s += U_flat[u_idx, c] * V_flat[v_idx, c] * W_flat[w_idx, c]
            error += (s - 1.0) ** 2
            
        # Check invalid indices (should be 0)
        # We only need to check a subset of invalid indices to save time, 
        # or all of them. Total indices = nm * mp * pn = 15 * 30 * 18 = 8100.
        # Checking all 8100 - 90 = 8010 indices is fast.
        # We can pre-compute the list of invalid indices.
        valid_set = set(target_indices)
        # To avoid generating all 8100 indices every time, we can rely on the fact
        # that if the valid ones are 1, the invalid ones are likely 0 if the 
        # decomposition is structured, but we must enforce it.
        # For speed in optimization, we might skip invalid checks in the inner loop
        # and only verify at the end.
        return error

    # Pre-compute invalid indices for final verification
    all_indices = set()
    for u in range(nm):
        for v in range(mp):
            for w in range(pn):
                all_indices.add((u, v, w))
    invalid_indices = list(all_indices - set(target_indices))
    
    def full_error(params, r):
        U_flat = params[:nm * r].reshape(nm, r)
        V_flat = params[nm * r : nm * r + mp * r].reshape(mp, r)
        W_flat = params[nm * r + mp * r :].reshape(pn, r)
        
        error = 0.0
        for u_idx, v_idx, w_idx in target_indices:
            s = 0.0
            for c in range(r):
                s += U_flat[u_idx, c] * V_flat[v_idx, c] * W_flat[w_idx, c]
            error += (s - 1.0) ** 2
            
        for u_idx, v_idx, w_idx in invalid_indices:
            s = 0.0
            for c in range(r):
                s += U_flat[u_idx, c] * V_flat[v_idx, c] * W_flat[w_idx, c]
            error += (s - 0.0) ** 2
        return error

    # Try to find a lower rank
    # We'll try ranks from schoolbook_rank - 1 down to schoolbook_rank - 5
    # This is a heuristic to see if a simple reduction is possible.
    
    start_time = time.time()
    max_time = 110.0 # Leave 10s buffer
    
    for r in range(schoolbook_rank - 1, max(10, schoolbook_rank - 10), -1):
        if time.time() - start_time > max_time:
            break
            
        # Initialize with random sparse values
        # Use a mix of 0, 1, -1
        U_init = np.zeros((nm, r))
        V_init = np.zeros((mp, r))
        W_init = np.zeros((pn, r))
        
        # Randomly assign some 1s
        for c in range(r):
            if random.random() < 0.5:
                idx = random.randint(0, nm-1)
                U_init[idx, c] = 1
            if random.random() < 0.5:
                idx = random.randint(0, mp-1)
                V_init[idx, c] = 1
            if random.random() < 0.5:
                idx = random.randint(0, pn-1)
                W_init[idx, c] = 1
                
        params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        # Optimize
        # We use a simple Nelder-Mead or L-BFGS-B. L-BFGS-B is better for box constraints but we don't have them.
        # We just minimize the error.
        try:
            res = minimize(full_error, params_init, args=(r,), method='L-BFGS-B', options={'maxiter': 50, 'ftol': 1e-10})
            if res.fun < 1e-6:
                # Valid solution found
                U_opt = res.x[:nm * r].reshape(nm, r)
                V_opt = res.x[nm * r : nm * r + mp * r].reshape(mp, r)
                W_opt = res.x[nm * r + mp * r :].reshape(pn, r)
                
                # Verify strictly
                if full_error(res.x, r) < 1e-5:
                    best_decomp = (U_opt, V_opt, W_opt)
                    best_rank = r
                    # We can break early if we find a very low rank, but we want minimal.
                    # Continue to try lower ranks?
                    # The loop goes downwards, so we are already trying lower ranks.
                    # We can break if we find rank 1? Unlikely.
                    pass
        except Exception:
            pass

    def to_list(mat):
        return mat.tolist()
    
    U, V, W = best_decomp
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
# EVOLVE-BLOCK-END
