import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n, m, p = 3, 3, 4
    
    # Build target tensor T
    # T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i
    # a in 0..8, b in 0..11, c in 0..11
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    def verify_decomposition(U, V, W):
        """Check if U, V, W exactly decompose T."""
        recon = np.einsum('at,bt,ct->abc', U, V, W)
        # Check if max absolute error is small
        err = np.abs(recon - T).max()
        return err < 1e-9

    def snap_to_half_integers(U, V, W):
        """
        Try to snap U, V, W to nearest half-integers and verify.
        Returns snapped matrices if valid, else None.
        """
        U_s = np.round(U * 2) / 2.0
        V_s = np.round(V * 2) / 2.0
        W_s = np.round(W * 2) / 2.0
        
        if verify_decomposition(U_s, V_s, W_s):
            return U_s, V_s, W_s
        
        # Try integers
        U_i = np.round(U)
        V_i = np.round(V)
        W_i = np.round(W)
        if verify_decomposition(U_i, V_i, W_i):
            return U_i, V_i, W_i
            
        # Try quarters? The prompt says integer/half-integer.
        # If half-integer fails, maybe it's not a "nice" solution or we need better precision.
        # But if the exact solution is half-integers, rounding should work if error < 0.25.
        # Our tolerance 1e-9 is much smaller, so if it's close, it should snap.
        return None

    def try_rank_als(r, max_restarts=5, max_iters=100, tol=1e-8, seed=42):
        """
        Attempt to find a rank-r decomposition using ALS with multiple restarts.
        """
        if time.time() > deadline:
            return None
            
        np.random.seed(seed)
        
        # Pre-compute dimensions
        dim_a = n * m
        dim_b = m * p
        dim_c = p * n
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
                
            # Random initialization
            U = np.random.randn(dim_a, r) * 0.1
            V = np.random.randn(dim_b, r) * 0.1
            W = np.random.randn(dim_c, r) * 0.1
            
            for it in range(max_iters):
                if time.time() > deadline:
                    return None
                
                # Update U
                # For each a, solve U[a, :] such that sum_t U[a,t] V[b,t] W[c,t] = T[a,b,c]
                # This is a linear least squares problem: A_mat @ u = rhs
                # A_mat has shape (dim_b * dim_c, r)
                # A_mat[idx, t] = V[b, t] * W[c, t]
                
                # We can build A_mat once per iteration? No, V and W change.
                # But for fixed V, W, A_mat is constant for all a.
                # Let's build A_mat for U-update.
                # V shape (dim_b, r), W shape (dim_c, r)
                # We need outer product of each row of V and W.
                # A_mat_t = np.outer(V[:, t], W[:, t]) reshaped.
                
                # Efficient construction of A_mat (dim_b*dim_c, r)
                # A_mat[:, t] = (V[:, t] * W[:, t][:, None]).ravel() ? No.
                # V[:, t] is vector of length dim_b.
                # W[:, t] is vector of length dim_c.
                # Outer product is dim_b x dim_c.
                # Flatten to dim_b*dim_c.
                
                A_mat = np.zeros((dim_b * dim_c, r))
                for t in range(r):
                    # V[:, t] is (dim_b,)
                    # W[:, t] is (dim_c,)
                    # np.outer(V[:, t], W[:, t]) is (dim_b, dim_c)
                    A_mat[:, t] = np.outer(V[:, t], W[:, t]).ravel()
                
                # Solve for each row of U
                # T_a is T[a, :, :] reshaped to (dim_b*dim_c,)
                for a in range(dim_a):
                    rhs = T[a, :, :].ravel()
                    # np.linalg.lstsq(A_mat, rhs, rcond=None)
                    # Since A_mat is small (144 x r), this is fast.
                    U[a, :], residuals, rank_a, sv = np.linalg.lstsq(A_mat, rhs, rcond=None)
                
                # Update V
                # For each b, solve V[b, :]
                # A_mat_v[:, t] = (U[:, t] * W[:, t][:, None]).ravel() ? 
                # U shape (dim_a, r), W shape (dim_c, r)
                # Outer product of U[:, t] and W[:, t] -> (dim_a, dim_c)
                A_mat_v = np.zeros((dim_a * dim_c, r))
                for t in range(r):
                    A_mat_v[:, t] = np.outer(U[:, t], W[:, t]).ravel()
                
                for b in range(dim_b):
                    rhs = T[:, b, :].reshape(-1)
                    V[b, :], residuals, rank_b, sv = np.linalg.lstsq(A_mat_v, rhs, rcond=None)
                    
                # Update W
                # For each c, solve W[c, :]
                # A_mat_w[:, t] = (U[:, t] * V[:, t][:, None]).ravel()
                # U shape (dim_a, r), V shape (dim_b, r)
                A_mat_w = np.zeros((dim_a * dim_b, r))
                for t in range(r):
                    A_mat_w[:, t] = np.outer(U[:, t], V[:, t]).ravel()
                
                for c in range(dim_c):
                    rhs = T[:, :, c].reshape(-1)
                    W[c, :], residuals, rank_c, sv = np.linalg.lstsq(A_mat_w, rhs, rcond=None)
                
                # Check convergence
                recon = np.einsum('at,bt,ct->abc', U, V, W)
                err = np.abs(recon - T).max()
                if err < tol:
                    break
            
            # After ALS, check if we have a good solution
            recon = np.einsum('at,bt,ct->abc', U, V, W)
            err = np.abs(recon - T).max()
            if err < 1e-6:
                # Try to snap
                snapped = snap_to_half_integers(U, V, W)
                if snapped is not None:
                    return snapped
        
        return None

    # Try ranks from low to high
    # Start from 25, go up to 35
    # If we find one, return it.
    # Schoolbook is 36.
    
    # To save time, we can skip some ranks if we have time pressure, 
    # but 36 is the baseline.
    # Let's try 30, 32, 34, 35 first? No, lower is better.
    # Let's try 25 to 35.
    
    for r in range(25, 36):
        if time.time() > deadline - 5:
            break
            
        # Use different seeds for different ranks to diversify
        seed = 42 + r
        result = try_rank_als(r, max_restarts=10, max_iters=50, seed=seed)
        
        if result is not None:
            U, V, W = result
            U_list = [[float(x) for x in row] for row in U]
            V_list = [[float(x) for x in row] for row in V]
            W_list = [[float(x) for x in row] for row in W]
            return {"U": U_list, "V": V_list, "W": W_list}
            
    # Fallback to schoolbook rank 36
    r = n * m * p
    U = np.zeros((n*m, r))
    V = np.zeros((m*p, r))
    W = np.zeros((p*n, r))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
