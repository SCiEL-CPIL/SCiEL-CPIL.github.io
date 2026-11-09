import numpy as np
import time

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Shapes
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    # Build the target tensor T as a 3D array or as a list of non-zero entries
    # T[u, v, w] = 1 if u=im+j, v=jp+k, w=kn+i
    # We represent T as a list of tuples (u, v, w) where T[u,v,w]=1
    target_entries = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                target_entries.append((u, v, w))
                
    # Try to find a low rank decomposition
    # We will try ranks from 16 up to 30
    # For each rank, we try to find U, V, W such that sum_c U_c V_c W_c = T
    
    # Strategy: 
    # 1. Start with a random W of shape (dim_w, rank)
    # 2. Solve for U and V using least squares
    # 3. Iterate (ALS)
    # 4. Round to integers/half-integers and check validity
    
    best_rank = n * m * p
    best_U = None
    best_V = None
    best_W = None
    
    # Initialize schoolbook solution
    rank_sb = best_rank
    U_sb = np.zeros((dim_u, rank_sb))
    V_sb = np.zeros((dim_v, rank_sb))
    W_sb = np.zeros((dim_w, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                U_sb[u, col] = 1
                V_sb[v, col] = 1
                W_sb[w, col] = 1
                col += 1
    
    # Function to check if a decomposition is valid
    def is_valid(U, V, W):
        # Compute the tensor sum
        # T_calc[u, v, w] = sum_c U[u, c] V[v, c] W[w, c]
        # We only care about the entries in target_entries
        # We can compute the full tensor or just check the constraints
        # Since dim_u, dim_v, dim_w are small (6, 15, 10), we can compute the full tensor
        T_calc = np.einsum('uc,vc,wc->uvw', U, V, W)
        for u, v, w in target_entries:
            if abs(T_calc[u, v, w] - 1.0) > 1e-9:
                return False
        # Check that other entries are 0? 
        # The tensor T is 0 elsewhere.
        # So we must ensure T_calc[u, v, w] == 0 for (u,v,w) not in target_entries.
        # This is a strong constraint.
        # Let's check a few random non-target entries or all of them.
        # Total entries 6*15*10 = 900. Target entries 30.
        # We can check all.
        for u in range(dim_u):
            for v in range(dim_v):
                for w in range(dim_w):
                    if (u, v, w) not in target_entries:
                        if abs(T_calc[u, v, w]) > 1e-9:
                            return False
        return True

    # Try to find a lower rank solution using ALS
    # We will try ranks from 16 to 29
    # To save time, we only try a few iterations per rank
    start_time = time.time()
    deadline = start_time + 100  # 100 seconds limit
    
    for rank in range(16, best_rank):
        if time.time() > deadline:
            break
            
        # Try a few random initializations
        for attempt in range(5):
            if time.time() > deadline:
                break
                
            # Random W
            W_curr = np.random.randn(dim_w, rank)
            U_curr = np.zeros((dim_u, rank))
            V_curr = np.zeros((dim_v, rank))
            
            # ALS iterations
            for _ in range(20):
                # Solve for U
                # T_calc = sum_c U_c V_c W_c
                # For fixed V, W, T_calc[u, v, w] = sum_c U[u, c] (V[v, c] W[w, c])
                # We want T_calc[u, v, w] = T[u, v, w]
                # This is a linear system for U[u, :].
                # For each u, we have equations for all v, w.
                # Let M_u be a matrix where M_u[(v, w), c] = V[v, c] W[w, c]
                # Then M_u @ U[u, :] = T[u, :, :] (flattened)
                
                # Construct M for all u at once?
                # M shape: (dim_u, dim_v * dim_w, rank)
                # M[u, v*w, c] = V[v, c] W[w, c]
                
                # This might be memory heavy but dims are small.
                # dim_v * dim_w = 150. rank <= 30. dim_u = 6.
                # 6 * 150 * 30 = 27000 floats. Fine.
                
                M = np.einsum('vc,wc->vwc', V_curr, W_curr).reshape(dim_v * dim_w, rank).T
                # M shape: (rank, dim_v * dim_w)
                # We need M_u for each u? No, M is the same for all u?
                # Wait, U[u, c] multiplies V[v, c] W[w, c].
                # So for a fixed u, the coefficient matrix for U[u, :] is M.
                # The target vector is T[u, :, :] flattened.
                
                # So for all u, we solve M.T @ U[u, :] = T_flat[u, :]
                # Where M.T has shape (dim_v * dim_w, rank)
                # And U is (dim_u, rank)
                
                # Let's construct the full system.
                # We want to minimize sum_u || M.T @ U[u, :] - T_flat[u, :] ||^2
                # This is equivalent to solving a least squares problem for each row of U.
                
                # We can use np.linalg.lstsq
                # But we can also use the normal equations: (M M.T) U.T = M T_flat.T
                # Let's use lstsq for simplicity and stability.
                
                T_flat = np.zeros((dim_u, dim_v * dim_w))
                for u, v, w in target_entries:
                    T_flat[u, v * dim_w + w] = 1.0
                    # Note: T is 0 elsewhere.
                    
                # Solve for U
                # M.T shape (dim_v*dim_w, rank)
                # We want U (dim_u, rank)
                # For each u: M.T @ U[u, :] = T_flat[u, :]
                # This is (dim_v*dim_w x rank) @ (rank x 1) = (dim_v*dim_w x 1)
                # We can solve all u at once if we stack them.
                # Let A = M.T (dim_v*dim_w, rank)
                # Let B = T_flat (dim_u, dim_v*dim_w)
                # We want U (dim_u, rank) such that U @ A.T = B?
                # No. M.T @ U[u, :] = T_flat[u, :]
                # => (M.T).T @ U[u, :].T = T_flat[u, :].T ? No.
                # M.T is (L, r). U[u, :] is (r,). Result is (L,).
                # So A @ x = b.
                # We have A = M.T (L, r).
                # We want x = U[u, :] (r,).
                # b = T_flat[u, :] (L,).
                
                # We can solve A^T A x = A^T b for each u.
                # Or use lstsq.
                
                # Let's use a vectorized approach.
                # A = M.T (L, r)
                # B = T_flat (u_count, L)
                # We want X (u_count, r) such that X @ A.T = B?
                # No. A @ X.T = B.T?
                # A (L, r) @ X.T (r, u_count) = (L, u_count)
                # B.T (L, u_count)
                # So we solve A @ X.T = B.T
                # X.T = lstsq(A, B.T)
                # X = (lstsq(A, B.T)).T
                
                A_mat = M.T # (L, r)
                B_mat = T_flat.T # (L, u_count)
                sol, _, _, _ = np.linalg.lstsq(A_mat, B_mat, rcond=None)
                U_curr = sol.T # (u_count, r)
                
                # Solve for V
                # T_calc[u, v, w] = sum_c U[u, c] V[v, c] W[w, c]
                # For fixed U, W, solve for V.
                # For each v: M_V @ V[v, :] = T_V_flat[v, :]
                # M_V[(u, w), c] = U[u, c] W[w, c]
                M_V = np.einsum('uc,wc->uwc', U_curr, W_curr).reshape(dim_u * dim_w, rank).T
                # M_V shape: (rank, dim_u * dim_w)
                # A_V = M_V.T (dim_u * dim_w, rank)
                # B_V: Target for V.
                # T_V_flat[v, (u, w)] = T[u, v, w]
                T_V_flat = np.zeros((dim_v, dim_u * dim_w))
                for u, v, w in target_entries:
                    T_V_flat[v, u * dim_w + w] = 1.0
                A_V = M_V.T
                B_V = T_V_flat.T
                sol, _, _, _ = np.linalg.lstsq(A_V, B_V, rcond=None)
                V_curr = sol.T
                
                # Solve for W
                M_W = np.einsum('uc,vc->ucv', U_curr, V_curr).reshape(dim_u * dim_v, rank).T
                T_W_flat = np.zeros((dim_w, dim_u * dim_v))
                for u, v, w in target_entries:
                    T_W_flat[w, u * dim_v + v] = 1.0
                A_W = M_W.T
                B_W = T_W_flat.T
                sol, _, _, _ = np.linalg.lstsq(A_W, B_W, rcond=None)
                W_curr = sol.T
            
            # Check validity
            if is_valid(U_curr, V_curr, W_curr):
                # Try to round
                U_rounded = np.round(U_curr * 2) / 2
                V_rounded = np.round(V_curr * 2) / 2
                W_rounded = np.round(W_curr * 2) / 2
                if is_valid(U_rounded, V_rounded, W_rounded):
                    best_rank = rank
                    best_U = U_rounded
                    best_V = V_rounded
                    best_W = W_rounded
                    break # Found a valid solution for this rank, try lower? No, we are going up.
                    # We want the MINIMAL rank. So we should break out of the attempt loop
                    # and continue to the next (lower) rank? 
                    # Wait, the loop is `for rank in range(16, best_rank)`.
                    # So if we find rank 16, we should stop and return it.
                    # If we find rank 17, we should try 16 next? 
                    # No, we iterate 16, 17, 18...
                    # If we find 17, we update best_rank=17 and continue to 18? 
                    # No, we want the minimum. So if we find 16, we return.
                    # If we find 17, we keep looking for 16? 
                    # The loop goes 16, 17, 18.
                    # If we find 17, we should NOT break the outer loop.
                    # But we should break the attempt loop.
                    # And we should update best_rank.
                    # But if we find 16, we should return immediately.
                    if rank == 16:
                        # Convert to lists
                        return {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}
                else:
                    # Check if the non-rounded solution is valid (floating point)
                    # The task requires integer/half-integer entries.
                    # So we must return valid integer/half-integer.
                    pass

    # If no better solution found, return schoolbook
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
# EVOLVE-BLOCK-END
