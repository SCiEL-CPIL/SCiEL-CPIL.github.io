import numpy as np
import time

DIMENSIONS = [4, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 135  # Leave 15s safety margin

    # Build the list of non-zero entries
    nonzero_entries = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                nonzero_entries.append((a, b, c))
    
    num_nonzero = len(nonzero_entries)  # 120

    # Schoolbook as fallback
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    fallback = schoolbook()
    best_rank = n * m * p
    best_result = fallback

    # Try to find lower rank solutions using ALS-style approach
    def try_rank_als(r, max_iters=200, seed=42):
        nonlocal best_rank, best_result
        if time.time() > deadline:
            return False
        
        np.random.seed(seed)
        
        # Initialize U, V, W with random values
        U = np.random.randn(n*m, r) * 0.5
        V = np.random.randn(m*p, r) * 0.5
        W = np.random.randn(p*n, r) * 0.5
        
        # Precompute index arrays
        a_idx = np.array([e[0] for e in nonzero_entries])
        b_idx = np.array([e[1] for e in nonzero_entries])
        c_idx = np.array([e[2] for e in nonzero_entries])
        
        for it in range(max_iters):
            if time.time() > deadline:
                return False
            
            # Update W: for each c, we need sum_k U[a,k]*V[b,k]*W[c,k] = 1
            # Group equations by c
            for c in range(p*n):
                mask = (c_idx == c)
                if not np.any(mask):
                    continue
                a_sub = a_idx[mask]
                b_sub = b_idx[mask]
                # For these equations: sum_k U[a,k]*V[b,k]*W[c,k] = 1
                # Let X = U[a,:] * V[b,:] (element-wise product), shape (num_eqs, r)
                X = U[a_sub, :] * V[b_sub, :]
                # We need X @ W[c,:] = 1 (vector of ones)
                # This is a linear system: W[c,:] = X^+ @ ones
                if X.shape[0] >= r:
                    # Use least squares
                    W_c, _, _, _ = np.linalg.lstsq(X, np.ones(X.shape[0]), rcond=None)
                    W[c, :] = W_c
                else:
                    # Underdetermined, use pseudoinverse
                    W[c, :] = np.linalg.pinv(X) @ np.ones(X.shape[0])
            
            # Update V: for each b
            for b in range(m*p):
                mask = (b_idx == b)
                if not np.any(mask):
                    continue
                a_sub = a_idx[mask]
                c_sub = c_idx[mask]
                X = U[a_sub, :] * W[c_sub, :]
                if X.shape[0] >= r:
                    V_b, _, _, _ = np.linalg.lstsq(X, np.ones(X.shape[0]), rcond=None)
                    V[b, :] = V_b
                else:
                    V[b, :] = np.linalg.pinv(X) @ np.ones(X.shape[0])
            
            # Update U: for each a
            for a in range(n*m):
                mask = (a_idx == a)
                if not np.any(mask):
                    continue
                b_sub = b_idx[mask]
                c_sub = c_idx[mask]
                X = V[b_sub, :] * W[c_sub, :]
                if X.shape[0] >= r:
                    U_a, _, _, _ = np.linalg.lstsq(X, np.ones(X.shape[0]), rcond=None)
                    U[a, :] = U_a
                else:
                    U[a, :] = np.linalg.pinv(X) @ np.ones(X.shape[0])
            
            # Check convergence
            if it % 5 == 0:
                # Compute residual
                residuals = []
                for a, b, c in nonzero_entries:
                    val = np.sum(U[a, :] * V[b, :] * W[c, :])
                    residuals.append(val - 1)
                residuals = np.array(residuals)
                err = np.max(np.abs(residuals))
                if err < 1e-10:
                    # Verify all entries
                    valid = True
                    for a, b, c in nonzero_entries:
                        if abs(np.sum(U[a, :] * V[b, :] * W[c, :]) - 1) > 1e-10:
                            valid = False
                            break
                    if valid:
                        if r < best_rank:
                            best_rank = r
                            best_result = {"U": U.copy(), "V": V.copy(), "W": W.copy()}
                        return True
        
        return False

    # Try ranks from lower bound upward
    # Lower bound: max(n*m, m*p, p*n) = 30, but likely higher
    # For M_{4,5,6}, let's try a range
    for r in range(30, 121):
        if time.time() > deadline:
            break
        success = try_rank_als(r, max_iters=50, seed=r)
        if success:
            break

    if best_rank < n * m * p:
        return best_result
    else:
        return fallback

