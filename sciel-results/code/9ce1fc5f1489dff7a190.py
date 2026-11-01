import numpy as np
import time

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    # Build the target tensor constraints
    # T[i*m+j, j*p+k, k*n+i] = 1
    constraints = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                constraints.append((u_idx, v_idx, w_idx))
    
    u_idxs = np.array([c[0] for c in constraints])
    v_idxs = np.array([c[1] for c in constraints])
    w_idxs = np.array([c[2] for c in constraints])
    num_constraints = len(constraints)
    
    def compute_error(U, V, W):
        """Compute sum of squared errors for all constraints."""
        vals = np.sum(U[u_idxs, :] * V[v_idxs, :] * W[w_idxs, :], axis=1)
        return np.sum((vals - 1.0) ** 2)
    
    def verify_exact(U, V, W, tol=1e-10):
        """Check if the decomposition is exact (within tolerance)."""
        return compute_error(U, V, W) < tol
    
    def round_to_halfint(X):
        """Round to nearest half-integer."""
        return np.round(X * 2) / 2
    
    def polish(U, V, W, max_iters=200, tol=1e-14):
        """High-precision Newton-style polish to drive error to zero."""
        for _ in range(max_iters):
            if time.time() > deadline - 5:
                break
            vals = np.sum(U[u_idxs, :] * V[v_idxs, :] * W[w_idxs, :], axis=1)
            diffs = vals - 1.0
            if np.max(np.abs(diffs)) < tol:
                break
            
            r = U.shape[1]
            # Build linear system: for each constraint, we have U[u]*V[v]*W[w]
            # Use Gauss-Newton style: treat one factor at a time
            # Actually, use a simple iterative refinement
            # Update W: for each constraint c, W[w_c] should satisfy sum_k U[u_c,k]*V[v_c,k]*W[w_c,k] = 1
            # This is a linear system if we fix U and V
            # Let's do coordinate descent polish
            for coord in range(r):
                # Update W[:, coord]
                # For each constraint c: sum_k U[u_c,k]*V[v_c,k]*W[w_c,k] = 1
                # Fix all other coordinates, solve for W[:, coord]
                # This is a linear system: A * x = b where A[w_c, w_c'] = U[u_c]*V[v_c] for the fixed part
                # Actually simpler: just do a few GD steps with small lr
                pass
            
            # Simple: just do more GD steps with very small lr
            lr = 0.001
            vals = np.sum(U[u_idxs, :] * V[v_idxs, :] * W[w_idxs, :], axis=1)
            diffs = 2 * (vals - 1.0)
            
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            
            np.add.at(dU, u_idxs, diffs[:, None] * V[v_idxs, :] * W[w_idxs, :])
            np.add.at(dV, v_idxs, diffs[:, None] * U[u_idxs, :] * W[w_idxs, :])
            np.add.at(dW, w_idxs, diffs[:, None] * U[u_idxs, :] * V[v_idxs, :])
            
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
            
            err = compute_error(U, V, W)
            if err < tol:
                break
        
        return U, V, W
    
    # Schoolbook rank-24 solution
    rank_seed = n * m * p
    U_seed = np.zeros((n * m, rank_seed), dtype=np.float64)
    V_seed = np.zeros((m * p, rank_seed), dtype=np.float64)
    W_seed = np.zeros((p * n, rank_seed), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_seed[i * m + j, col] = 1
                V_seed[j * p + k, col] = 1
                W_seed[k * n + i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_seed.copy(), V_seed.copy(), W_seed.copy()
    best_rank = rank_seed
    
    def merge_columns(U, V, W, c1, c2, alpha=0.5):
        """Merge two columns into one via convex combination."""
        U_m = U[:, c1] * alpha + U[:, c2] * (1 - alpha)
        V_m = V[:, c1] * alpha + V[:, c2] * (1 - alpha)
        W_m = W[:, c1] * alpha + W[:, c2] * (1 - alpha)
        
        idxs = [c for c in range(U.shape[1]) if c != c1 and c != c2]
        U_new = np.column_stack([U[:, idxs], U_m]) if idxs else U_m[:, None]
        V_new = np.column_stack([V[:, idxs], V_m]) if idxs else V_m[:, None]
        W_new = np.column_stack([W[:, idxs], W_m]) if idxs else W_m[:, None]
        
        return U_new, V_new, W_new
    
    def try_rank_from_seed(r, num_trials=50, max_steps=800, seed_base=0):
        """Try to find a rank-r decomposition by merging columns from rank-24 seed."""
        if time.time() > deadline - 15:
            return None
        
        for trial in range(num_trials):
            if time.time() > deadline - 15:
                return None
            
            np.random.seed(seed_base + trial)
            
            U = U_seed.copy()
            V = V_seed.copy()
            W = W_seed.copy()
            current_rank = rank_seed
            
            # Greedily merge columns to reduce rank from 24 to r
            while current_rank > r:
                if time.time() > deadline - 15:
                    return None
                
                # Try all pairs, pick the one that minimizes error after merge
                best_pair = None
                best_err = float('inf')
                best_alpha = 0.5
                
                # Sample some pairs if too many
                num_pairs = current_rank * (current_rank - 1) // 2
                if num_pairs > 100:
                    pairs = np.random.choice(current_rank * (current_rank - 1) // 2, size=100, replace=False)
                    pair_list = []
                    for pi in pairs:
                        # Convert linear index to (c1, c2)
                        c2 = int((-1 + np.sqrt(1 + 8 * pi)) / 2)
                        c1 = pi - c2 * (c2 + 1) // 2
                        pair_list.append((int(c1), int(c2)))
                else:
                    pair_list = [(c1, c2) for c1 in range(current_rank) for c2 in range(c1 + 1, current_rank)]
                
                for c1, c2 in pair_list:
                    for alpha in [0.3, 0.5, 0.7]:
                        U_m, V_m, W_m = merge_columns(U, V, W, c1, c2, alpha)
                        err = compute_error(U_m, V_m, W_m)
                        if err < best_err:
                            best_err = err
                            best_pair = (c1, c2)
                            best_alpha = alpha
                
                if best_pair is None:
                    return None
                
                U, V, W = merge_columns(U, V, W, best_pair[0], best_pair[1], best_alpha)
                current_rank -= 1
                
                # If error is getting large, break
                if best_err > 10.0:
                    break
            
            if current_rank > r:
                continue
            
            # Now polish with gradient descent
            lr = 0.01
            momentum = 0.9
            vU = np.zeros_like(U)
            vV = np.zeros_like(V)
            vW = np.zeros_like(W)
            
            prev_err = compute_error(U, V, W)
            stuck_count = 0
            
            for step in range(max_steps):
                if time.time() > deadline - 15:
                    return None
                
                vals = np.sum(U[u_idxs, :] * V[v_idxs, :] * W[w_idxs, :], axis=1)
                diffs = 2 * (vals - 1.0)
                
                dU = np.zeros_like(U)
                dV = np.zeros_like(V)
                dW = np.zeros_like(W)
                
                np.add.at(dU, u_idxs, diffs[:, None] * V[v_idxs, :] * W[w_idxs, :])
                np.add.at(dV, v_idxs, diffs[:, None] * U[u_idxs, :] * W[w_idxs, :])
                np.add.at(dW, w_idxs, diffs[:, None] * U[u_idxs, :] * V[v_idxs, :])
                
                vU = momentum * vU - lr * dU
                vV = momentum * vV - lr * dV
                vW = momentum * vW - lr * dW
                
                U += vU
                V += vV
                W += vW
                
                err = compute_error(U, V, W)
                
                if err < 1e-14:
                    U_r = round_to_halfint(U)
                    V_r = round_to_halfint(V)
                    W_r = round_to_halfint(W)
                    if verify_exact(U_r, V_r, W_r):
                        return U_r, V_r, W_r
                    else:
                        # Try polish then round
                        U_p, V_p, W_p = polish(U, V, W)
                        U_r = round_to_halfint(U_p)
                        V_r = round_to_halfint(V_p)
                        W_r = round_to_halfint(W_p)
                        if verify_exact(U_r, V_r, W_r):
                            return U_r, V_r, W_r
                        break
                
                if err > prev_err * 1.01:
                    lr *= 0.95
                    stuck_count += 1
                    if stuck_count > 30:
                        break
                else:
                    stuck_count = 0
                    if step > 50 and err < 1.0:
                        lr = min(lr * 1.02, 0.05)
                
                prev_err = err
            
            # Final attempt: polish and check
            U_p, V_p, W_p = polish(U, V, W)
            U_r = round_to_halfint(U_p)
            V_r = round_to_halfint(V_p)
            W_r = round_to_halfint(W_p)
            if verify_exact(U_r, V_r, W_r):
                return U_r, V_r, W_r
        
        return None
    
    def try_rank_random(r, num_restarts=30, max_steps=600, seed_base=0):
        """Try random initialization with gradient descent."""
        if time.time() > deadline - 15:
            return None
        
        for restart in range(num_restarts):
            if time.time() > deadline - 15:
                return None
            
            np.random.seed(seed_base + restart)
            
            # Mix of initialization strategies
            strategy = restart % 3
            if strategy == 0:
                scale = 0.2
                U = np.random.randn(n * m, r) * scale
                V = np.random.randn(m * p, r) * scale
                W = np.random.randn(p * n, r) * scale
            elif strategy == 1:
                # Sparse-ish initialization
                U = np.zeros((n * m, r))
                V = np.zeros((m * p, r))
                W = np.zeros((p * n, r))
                for c in range(r):
                    U[np.random.randint(n * m), c] = 1
                    V[np.random.randint(m * p), c] = 1
                    W[np.random.randint(p * n), c] = 1
            else:
                # Use subset of seed columns with noise
                cols = np.random.choice(rank_seed, size=r, replace=False)
                U = U_seed[:, cols].copy() + np.random.randn(n * m, r) * 0.1
                V = V_seed[:, cols].copy() + np.random.randn(m * p, r) * 0.1
                W = W_seed[:, cols].copy() + np.random.randn(p * n, r) * 0.1
            
            lr = 0.015
            momentum = 0.9
            vU = np.zeros_like(U)
            vV = np.zeros_like(V)
            vW = np.zeros_like(W)
            
            prev_err = compute_error(U, V, W)
            stuck_count = 0
            
            for step in range(max_steps):
                if time.time() > deadline - 15:
                    return None
                
                vals = np.sum(U[u_idxs, :] * V[v_idxs, :] * W[w_idxs, :], axis=1)
                diffs = 2 * (vals - 1.0)
                
                dU = np.zeros_like(U)
                dV = np.zeros_like(V)
                dW = np.zeros_like(W)
                
                np.add.at(dU, u_idxs, diffs[:, None] * V[v_idxs, :] * W[w_idxs, :])
                np.add.at(dV, v_idxs, diffs[:, None] * U[u_idxs, :] * W[w_idxs, :])
                np.add.at(dW, w_idxs, diffs[:, None] * U[u_idxs, :] * V[v_idxs, :])
                
                vU = momentum * vU - lr * dU
                vV = momentum * vV - lr * dV
                vW = momentum * vW - lr * dW
                
                U += vU
                V += vV
                W += vW
                
                err = compute_error(U, V, W)
                
                if err < 1e-14:
                    U_r = round_to_halfint(U)
                    V_r = round_to_halfint(V)
                    W_r = round_to_halfint(W)
                    if verify_exact(U_r, V_r, W_r):
                        return U_r, V_r, W_r
                    else:
                        U_p, V_p, W_p = polish(U, V, W)
                        U_r = round_to_halfint(U_p)
                        V_r = round_to_halfint(V_p)
                        W_r = round_to_halfint(W_p)
                        if verify_exact(U_r, V_r, W_r):
                            return U_r, V_r, W_r
                        break
                
                if err > prev_err * 1.01:
                    lr *= 0.95
                    stuck_count += 1
                    if stuck_count > 30:
                        break
                else:
                    stuck_count = 0
                    if step > 50 and err < 1.0:
                        lr = min(lr * 1.02, 0.05)
                
                prev_err = err
            
            U_p, V_p, W_p = polish(U, V, W)
            U_r = round_to_halfint(U_p)
            V_r = round_to_halfint(V_p)
            W_r = round_to_halfint(W_p)
            if verify_exact(U_r, V_r, W_r):
                return U_r, V_r, W_r
        
        return None
    
    # Try ranks from 18 to 23, using both strategies
    for r in [18, 19, 20, 21, 22, 23]:
        if time.time() > deadline - 25:
            break
        
        # Try seed-merge strategy
        sol = try_rank_from_seed(r, num_trials=20, max_steps=500, seed_base=r * 1000)
        if sol is not None:
            U_r, V_r, W_r = sol
            if verify_exact(U_r, V_r, W_r):
                best_U, best_V, best_W = U_r, V_r, W_r
                best_rank = r
                break
        
        # Try random strategy
        if time.time() > deadline - 25:
            break
        sol = try_rank_random(r, num_restarts=20, max_steps=400, seed_base=r * 2000 + 500)
        if sol is not None:
            U_r, V_r, W_r = sol
            if verify_exact(U_r, V_r, W_r):
                best_U, best_V, best_W = U_r, V_r, W_r
                best_rank = r
                break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
