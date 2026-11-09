"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 135  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    # Schoolbook as fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][column] = 1
                V_school[j * p + k][column] = 1
                W_school[k * n + i][column] = 1
                column += 1
    
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_school

    # Pre-compute the list of valid (a, b, c) indices for the tensor
    # a = i*m + j, b = j*p + k, c = k*n + i
    triples = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                triples.append((a, b, c))
    
    num_triples = len(triples)
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n

    a_idx = np.array([t[0] for t in triples])
    b_idx = np.array([t[1] for t in triples])
    c_idx = np.array([t[2] for t in triples])

    def calculate_error(U, V, W):
        """Calculate max absolute error from 1.0"""
        T_vals = np.zeros(num_triples)
        rank = U.shape[1]
        for s in range(rank):
            T_vals += U[a_idx, s] * V[b_idx, s] * W[c_idx, s]
        return np.max(np.abs(T_vals - 1.0))

    def local_search_half_int(U, V, W, max_iters=5000):
        """
        Perform local search on half-integer grid to minimize error to 0.
        Returns (U, V, W) if error is 0, else None.
        """
        U = np.round(U * 2) / 2.0
        V = np.round(V * 2) / 2.0
        W = np.round(W * 2) / 2.0
        
        current_err = calculate_error(U, V, W)
        if current_err < 1e-9:
            return U, V, W
            
        # Local search: try changing one entry at a time
        # We will iterate over all entries and try small perturbations
        # To be fast, we only check if error decreases significantly
        
        # Flatten indices for iteration
        U_flat_idx = []
        for i in range(dim_U):
            for s in range(U.shape[1]):
                U_flat_idx.append((i, s))
        V_flat_idx = []
        for i in range(dim_V):
            for s in range(V.shape[1]):
                V_flat_idx.append((i, s))
        W_flat_idx = []
        for i in range(dim_W):
            for s in range(W.shape[1]):
                W_flat_idx.append((i, s))

        # Randomize order to avoid bias
        np.random.shuffle(U_flat_idx)
        np.random.shuffle(V_flat_idx)
        np.random.shuffle(W_flat_idx)

        # Candidate values: current, current +/- 0.5, current +/- 1.0, 0, 1, -1
        # We'll just try current +/- 0.5 and 0, 1, -1
        
        for _ in range(max_iters):
            if time.time() > deadline - 5:
                return None
                
            improved = False
            
            # Try U
            for idx in U_flat_idx:
                if time.time() > deadline - 5:
                    return None
                i, s = idx
                orig_val = U[i, s]
                # Try nearby half-integers
                candidates = [orig_val - 0.5, orig_val + 0.5, 0.0, 0.5, -0.5, 1.0, -1.0, 1.5, -1.5]
                best_val = orig_val
                best_err = current_err
                for cand in candidates:
                    if cand == orig_val:
                        continue
                    U[i, s] = cand
                    err = calculate_error(U, V, W)
                    if err < best_err - 1e-10:
                        best_err = err
                        best_val = cand
                    else:
                        U[i, s] = orig_val # revert if not better immediately? 
                        # Actually, we should revert if it's not the best we found so far in this step
                        # But to keep it simple: if cand is better than current, keep it.
                        # If multiple are better, pick the best.
                        # This implementation reverts immediately if not better than best_err found so far in this loop?
                        # No, it reverts to orig_val if cand is not better than best_err.
                        # If cand IS better, we set best_val and best_err.
                        # We need to restore U[i,s] to orig_val before trying next cand?
                        # Yes.
                    U[i, s] = orig_val # Always revert after testing a candidate
                    
                if best_val != orig_val:
                    U[i, s] = best_val
                    current_err = best_err
                    improved = True
                    if current_err < 1e-9:
                        return U, V, W
                        
            # Try V
            for idx in V_flat_idx:
                if time.time() > deadline - 5:
                    return None
                i, s = idx
                orig_val = V[i, s]
                candidates = [orig_val - 0.5, orig_val + 0.5, 0.0, 0.5, -0.5, 1.0, -1.0, 1.5, -1.5]
                best_val = orig_val
                best_err = current_err
                for cand in candidates:
                    if cand == orig_val:
                        continue
                    V[i, s] = cand
                    err = calculate_error(U, V, W)
                    if err < best_err - 1e-10:
                        best_err = err
                        best_val = cand
                    V[i, s] = orig_val
                    
                if best_val != orig_val:
                    V[i, s] = best_val
                    current_err = best_err
                    improved = True
                    if current_err < 1e-9:
                        return U, V, W
                        
            # Try W
            for idx in W_flat_idx:
                if time.time() > deadline - 5:
                    return None
                i, s = idx
                orig_val = W[i, s]
                candidates = [orig_val - 0.5, orig_val + 0.5, 0.0, 0.5, -0.5, 1.0, -1.0, 1.5, -1.5]
                best_val = orig_val
                best_err = current_err
                for cand in candidates:
                    if cand == orig_val:
                        continue
                    W[i, s] = cand
                    err = calculate_error(U, V, W)
                    if err < best_err - 1e-10:
                        best_err = err
                        best_val = cand
                    W[i, s] = orig_val
                    
                if best_val != orig_val:
                    W[i, s] = best_val
                    current_err = best_err
                    improved = True
                    if current_err < 1e-9:
                        return U, V, W

            if not improved:
                # Try a random restart of local search? Or just break.
                # If no improvement in a full pass, we might be in a local min.
                # Break to save time, or do a random jump.
                # Let's do a random jump: pick a random entry and set to random half-integer.
                if np.random.rand() < 0.1:
                    # Random jump
                    mat = np.random.choice([U, V, W])
                    i = np.random.randint(mat.shape[0])
                    s = np.random.randint(mat.shape[1])
                    mat[i, s] = np.random.randint(-3, 4) / 2.0
                else:
                    break

        if current_err < 1e-9:
            return U, V, W
        return None

    def try_als_decomposition(rank, max_iter=50, restarts=10):
        """
        Attempts to find a rank-r decomposition using ALS.
        Returns (U, V, W) if found and verified, else None.
        """
        if time.time() > deadline - 5:
            return None
            
        for restart in range(restarts):
            if time.time() > deadline - 5:
                return None
            
            # Initialize with random small values
            np.random.seed(12345 + rank + restart)
            U = np.random.randn(dim_U, rank) * 0.1
            V = np.random.randn(dim_V, rank) * 0.1
            W = np.random.randn(dim_W, rank) * 0.1
            
            for it in range(max_iter):
                if time.time() > deadline - 5:
                    return None
                
                # Update U
                A_U = np.zeros((num_triples, dim_U * rank))
                for s in range(rank):
                    vals = V[b_idx, s] * W[c_idx, s]
                    cols = a_idx * rank + s
                    A_U[np.arange(num_triples), cols] = vals
                
                rhs = np.ones(num_triples)
                try:
                    U_flat, _, _, _ = np.linalg.lstsq(A_U, rhs, rcond=None)
                    U_new = U_flat.reshape(dim_U, rank)
                except:
                    continue
                
                # Update V
                A_V = np.zeros((num_triples, dim_V * rank))
                for s in range(rank):
                    vals = U_new[a_idx, s] * W[c_idx, s]
                    cols = b_idx * rank + s
                    A_V[np.arange(num_triples), cols] = vals
                
                try:
                    V_flat, _, _, _ = np.linalg.lstsq(A_V, rhs, rcond=None)
                    V_new = V_flat.reshape(dim_V, rank)
                except:
                    continue
                
                # Update W
                A_W = np.zeros((num_triples, dim_W * rank))
                for s in range(rank):
                    vals = U_new[a_idx, s] * V_new[b_idx, s]
                    cols = c_idx * rank + s
                    A_W[np.arange(num_triples), cols] = vals
                
                try:
                    W_flat, _, _, _ = np.linalg.lstsq(A_W, rhs, rcond=None)
                    W_new = W_flat.reshape(dim_W, rank)
                except:
                    continue
                
                U, V, W = U_new, V_new, W_new
                
                # Check for convergence / exactness
                if it % 5 == 0 or it == max_iter - 1:
                    err = calculate_error(U, V, W)
                    if err < 1e-9:
                        # Found a solution!
                        # Try to round to half-integers
                        U_rounded = np.round(U * 2) / 2.0
                        V_rounded = np.round(V * 2) / 2.0
                        W_rounded = np.round(W * 2) / 2.0
                        
                        # Verify rounded solution
                        err_rounded = calculate_error(U_rounded, V_rounded, W_rounded)
                        if err_rounded < 1e-9:
                            return U_rounded, V_rounded, W_rounded
                        
                        # If rounding failed, try local search on the float solution
                        # This might find a nearby integer solution
                        result = local_search_half_int(U, V, W, max_iters=1000)
                        if result:
                            return result

        return None

    # Try ranks from min_rank to max_rank_to_try
    # Prioritize higher ranks that are more likely to be solvable? 
    # Or just go up from lower bound.
    # Let's go up from 15.
    min_rank = 15
    max_rank_to_try = 30 # Include schoolbook
    
    for target_rank in range(min_rank, max_rank_to_try + 1):
        if time.time() > deadline - 10:
            break
        
        # Try ALS
        result = try_als_decomposition(target_rank, max_iter=30, restarts=5)
        
        if result is not None:
            U, V, W = result
            # Final verification
            if calculate_error(U, V, W) < 1e-9:
                best_rank = target_rank
                best_result = {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
                break  # Found minimal rank in this range, stop
        
        # If ALS didn't find it, try a quick random search for this rank
        # This helps if ALS gets stuck in local minima.
        if time.time() < deadline - 15:
            np.random.seed(42 + target_rank)
            for attempt in range(5):
                if time.time() > deadline - 10:
                    break
                # Use smaller range for random init to help convergence
                U_rand = np.random.randint(-2, 3, size=(n*m, target_rank)) / 2.0
                V_rand = np.random.randint(-2, 3, size=(m*p, target_rank)) / 2.0
                W_rand = np.random.randint(-2, 3, size=(p*n, target_rank)) / 2.0
                
                # Try local search on this random start
                result_ls = local_search_half_int(U_rand, V_rand, W_rand, max_iters=500)
                if result_ls:
                    U, V, W = result_ls
                    if calculate_error(U, V, W) < 1e-9:
                        best_rank = target_rank
                        best_result = {
                            "U": U.tolist(),
                            "V": V.tolist(),
                            "W": W.tolist()
                        }
                        break
            if best_rank == target_rank:
                break

    return best_result

# EVOLVE-BLOCK-END
