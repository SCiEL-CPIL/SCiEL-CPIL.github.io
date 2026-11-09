import numpy as np
import time

DIMENSIONS = [3, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    rank_sb = n * m * p
    
    # Schoolbook construction
    U_sb = np.zeros((n * m, rank_sb))
    V_sb = np.zeros((m * p, rank_sb))
    W_sb = np.zeros((p * n, rank_sb))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
                
    # Precompute index mappings for fast evaluation
    # We need to evaluate: sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    # Let's create arrays of indices
    # i is 0..n-1, j is 0..m-1, k is 0..p-1
    
    # To speed up the error calculation, we can reshape
    # But the indices are coupled.
    # U_idx = i*m + j
    # V_idx = j*p + k
    # W_idx = k*n + i
    
    # We can create a list of tuples (u_idx, v_idx, w_idx) for all i,j,k
    indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                indices.append((i * m + j, j * p + k, k * n + i))
                
    indices = np.array(indices)
    u_idx = indices[:, 0]
    v_idx = indices[:, 1]
    w_idx = indices[:, 2]
    
    def compute_error(U, V, W):
        # U shape (nm, R), V (mp, R), W (pn, R)
        # We want sum_r U[u_idx, r] V[v_idx, r] W[w_idx, r]
        # This is a batched matrix multiplication or einsum
        # E = einsum('ir, jr, kr -> i', U[u_idx], V[v_idx], W[w_idx])
        # But u_idx, v_idx, w_idx are arrays of length nmp
        # So U[u_idx] has shape (nmp, R)
        # np.einsum('ir,ir->i', U[u_idx], V[v_idx]) gives (nmp,)
        # Then multiply by W[w_idx]
        
        # More efficient:
        # Temp = U[u_idx] * V[v_idx] -> (nmp, R)
        # Sum over r: Temp.sum(axis=1) -> (nmp,)
        # Then multiply by W[w_idx]? No, W is also in the sum.
        # It is sum_r (U[u_idx, r] * V[v_idx, r] * W[w_idx, r])
        
        # Using einsum:
        # 'ar, br, cr -> abc' is not right because a,b,c are coupled.
        # We have arrays of indices.
        # np.einsum('ir,jr,kr->ij', ...) is for matrix mult.
        # Here we have 3 tensors coupled by r and by the specific i,j,k triple.
        
        # Let's just do:
        # Term = U[u_idx] * V[v_idx] # (nmp, R)
        # Term = Term * W[w_idx]    # (nmp, R)
        # Sum = Term.sum(axis=1)    # (nmp,)
        
        return np.sum((np.sum(U[u_idx] * V[v_idx] * W[w_idx], axis=1) - 1.0)**2)

    def snap_to_half_integers(U, V, W):
        # Snap to nearest 0.5
        U_s = np.round(U * 2) / 2
        V_s = np.round(V * 2) / 2
        W_s = np.round(W * 2) / 2
        return U_s, V_s, W_s

    def is_exact(U, V, W, tol=1e-9):
        err = compute_error(U, V, W)
        return err < tol

    start_time = time.time()
    deadline = start_time + 110
    
    best_rank = rank_sb
    best_U, best_V, best_W = U_sb.copy(), V_sb.copy(), W_sb.copy()
    
    # Try to find a better rank
    # We try ranks from rank_sb - 1 down to max(10, rank_sb - 30)
    # The reduction might be small.
    
    for R in range(rank_sb - 1, max(10, rank_sb - 40), -1):
        if time.time() > deadline:
            break
            
        found = False
        # Try a few random restarts
        for attempt in range(10):
            if time.time() > deadline:
                break
                
            # Initialize with random values
            # Scale is important. 
            # If we use Strassen-like, values are small integers.
            # Let's start with small random integers or reals.
            U = np.random.randn(n * m, R) * 0.5
            V = np.random.randn(m * p, R) * 0.5
            W = np.random.randn(p * n, R) * 0.5
            
            # Gradient Descent
            # We need gradients.
            # E = sum_i (sum_r U_{u_i, r} V_{v_i, r} W_{w_i, r} - 1)^2
            # dE/dU_{u, r} = 2 * sum_{i: u_i=u} (val_i - 1) * V_{v_i, r} * W_{w_i, r}
            
            # We can compute gradients numerically or analytically.
            # Analytical is faster.
            
            # Precompute for gradients
            # For each equation i (index in 0..nmp-1):
            # val_i = sum_r U[u_i, r] V[v_i, r] W[w_i, r]
            # err_i = val_i - 1
            # dU[u_i, r] += 2 * err_i * V[v_i, r] * W[w_i, r]
            
            # We can vectorize this.
            
            lr = 0.01
            for step in range(100):
                if time.time() > deadline:
                    break
                    
                # Compute values
                vals = np.sum(U[u_idx] * V[v_idx] * W[w_idx], axis=1)
                errs = vals - 1.0
                err_val = np.sum(errs**2)
                
                if err_val < 1e-8:
                    break
                    
                # Compute gradients
                # dU[u_idx, r] = 2 * errs * V[v_idx, r] * W[w_idx, r]
                # We need to scatter-add these to U
                # Since u_idx has duplicates, we need to sum them.
                
                # dU_full = np.zeros((n*m, R))
                # np.add.at(dU_full, (u_idx, np.arange(R)[None, :]), 2 * errs[:, None] * V[v_idx] * W[w_idx])
                # This is slow.
                
                # Alternative: Use sparse matrix trick or just accept the slowness for small sizes.
                # n*m = 12, R ~ 80. u_idx length 84.
                # 84 * 80 = 6720 operations. Very fast.
                
                dU = np.zeros((n * m, R))
                dV = np.zeros((m * p, R))
                dW = np.zeros((p * n, R))
                
                # Vectorized gradient calculation
                # term = errs[:, None] * V[v_idx] * W[w_idx]  # (84, R)
                # We want to sum term where u_idx is the same.
                
                # np.add.at is the way.
                term_U = 2 * errs[:, None] * V[v_idx] * W[w_idx]
                term_V = 2 * errs[:, None] * U[u_idx] * W[w_idx]
                term_W = 2 * errs[:, None] * U[u_idx] * V[v_idx]
                
                # Add to gradients
                # We need to handle the r dimension.
                # np.add.at(dU, (u_idx, np.arange(R)), term_U)
                
                # This is tricky with 2D indices.
                # Let's loop over R? No, R is up to 80. 80 loops is fine.
                # Or just use a sparse approach.
                
                # Simpler: Just do a loop over the 84 equations.
                # 84 iterations is nothing.
                
                # But inside, we update 12, 28, 21 elements.
                # Total 84 * (12+28+21) ~ 4000 ops.
                
                # Let's use the loop for clarity and correctness.
                for i in range(len(u_idx)):
                    ui = u_idx[i]
                    vi = v_idx[i]
                    wi = w_idx[i]
                    e = 2 * errs[i]
                    
                    # dU[ui, :] += e * V[vi, :] * W[wi, :]
                    dU[ui, :] += e * V[vi, :] * W[wi, :]
                    dV[vi, :] += e * U[ui, :] * W[wi, :]
                    dW[wi, :] += e * U[ui, :] * V[vi, :]
                    
                # Update
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
                
                # Snap to half-integers periodically to guide the search?
                # No, keep it continuous for now.
                
            # Check if converged
            if err_val < 1e-6:
                # Snap to half-integers
                U_s, V_s, W_s = snap_to_half_integers(U, V, W)
                if is_exact(U_s, V_s, W_s):
                    best_rank = R
                    best_U, best_V, best_W = U_s, V_s, W_s
                    found = True
                    break
        
        if found:
            # If we found a solution for R, we can try to go lower?
            # The loop structure already goes lower.
            # But if we found one, we can break early if we are satisfied?
            # No, we want the MINIMAL rank. So we continue to lower R.
            # However, if we found R, we know R is possible.
            # We should continue to R-1.
            
            # Optimization: If we found a solution, we can use it as initialization for R-1?
            # Hard to do directly.
            pass

    # If no better solution found, return schoolbook
    if best_rank == rank_sb:
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    else:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
