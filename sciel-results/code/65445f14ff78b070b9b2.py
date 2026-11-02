import numpy as np
import time

DIMENSIONS = [2, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook
    rank_school = n * m * p
    
    # Build the target tensor non-zeros
    # T[u, v, w] = 1 if u=(i,j), v=(j,k), w=(k,i)
    # u in [0, n*m), v in [0, m*p), w in [0, p*n)
    non_zeros = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                non_zeros.append((u, v, w))
    
    K = len(non_zeros)
    U_dims = n * m
    V_dims = m * p
    W_dims = p * n
    
    # Precompute index arrays for vectorized operations
    nz_u = np.array([x[0] for x in non_zeros])
    nz_v = np.array([x[1] for x in non_zeros])
    nz_w = np.array([x[2] for x in non_zeros])
    
    def verify_decomp(U, V, W):
        """
        Checks if U, V, W exactly decompose the tensor.
        Returns True if valid.
        """
        r = len(U[0])
        U_arr = np.array(U)
        V_arr = np.array(V)
        W_arr = np.array(W)
        
        # Check all non-zero entries (vectorized)
        prods = np.sum(U_arr[nz_u] * V_arr[nz_v] * W_arr[nz_w], axis=1)
        if np.any(np.abs(prods - 1.0) > 1e-9):
            return False
        
        # Check all zero entries using sparse approach
        # Build full tensor and check zeros
        # For efficiency, only check a sample or use structure
        # Actually, let's check all - it's small enough
        full_u = np.arange(U_dims)
        full_v = np.arange(V_dims)
        full_w = np.arange(W_dims)
        
        # Create meshgrid
        U_grid, V_grid, W_grid = np.meshgrid(full_u, full_v, full_w, indexing='ij')
        U_flat = U_grid.flatten()
        V_flat = V_grid.flatten()
        W_flat = W_grid.flatten()
        
        # Compute all products
        all_prods = np.sum(U_arr[U_flat] * V_arr[V_flat] * W_arr[W_flat], axis=1)
        
        # Zero entries should be ~0
        nz_set = set(non_zeros)
        zero_mask = np.array([((u, v, w) not in nz_set) for u, v, w in zip(U_flat, V_flat, W_flat)])
        
        if np.any(np.abs(all_prods[zero_mask]) > 1e-9):
            return False
            
        return True

    def round_to_half(X):
        # Round to nearest 0.5
        return np.round(X * 2) / 2
    
    def compute_error(U, V, W):
        """Compute the error of the decomposition."""
        prods = np.sum(U[nz_u] * V[nz_v] * W[nz_w], axis=1)
        return np.sum((prods - 1.0)**2)
    
    def al_solve(U, V, W, rank, iters=50, tol=1e-12):
        """
        Alternating Least Squares for CP decomposition.
        Minimizes sum (T - sum U_c V_c W_c)^2
        """
        # Precompute indices grouped by w, u, v
        indices_by_w = {i: [] for i in range(W_dims)}
        indices_by_u = {i: [] for i in range(U_dims)}
        indices_by_v = {i: [] for i in range(V_dims)}
        
        for idx, (u, v, w) in enumerate(non_zeros):
            indices_by_w[w].append(idx)
            indices_by_u[u].append(idx)
            indices_by_v[v].append(idx)
            
        prev_err = np.inf
        for it in range(iters):
            # Update W
            for w in range(W_dims):
                if not indices_by_w[w]:
                    continue
                idxs = indices_by_w[w]
                A = np.empty((len(idxs), rank))
                for r_idx, idx in enumerate(idxs):
                    u, v, _ = non_zeros[idx]
                    A[r_idx, :] = U[u, :] * V[v, :]
                b = np.ones(len(idxs))
                sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                W[w, :] = sol
                
            # Update U
            for u in range(U_dims):
                if not indices_by_u[u]:
                    continue
                idxs = indices_by_u[u]
                A = np.empty((len(idxs), rank))
                for r_idx, idx in enumerate(idxs):
                    _, v, w = non_zeros[idx]
                    A[r_idx, :] = V[v, :] * W[w, :]
                b = np.ones(len(idxs))
                sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                U[u, :] = sol
                
            # Update V
            for v in range(V_dims):
                if not indices_by_v[v]:
                    continue
                idxs = indices_by_v[v]
                A = np.empty((len(idxs), rank))
                for r_idx, idx in enumerate(idxs):
                    u, _, w = non_zeros[idx]
                    A[r_idx, :] = U[u, :] * W[w, :]
                b = np.ones(len(idxs))
                sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                V[v, :] = sol
            
            # Check convergence
            if it > 0:
                err = compute_error(U, V, W)
                if err < tol:
                    break
                if err < prev_err * 0.999:
                    prev_err = err
                    
        return U, V, W

    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin before 120s limit
    
    # Try ranks from low to high
    # Start from 20, go up to schoolbook-1
    min_rank_to_try = 20
    max_rank_to_try = rank_school - 1
    
    for r in range(min_rank_to_try, max_rank_to_try + 1):
        if time.time() > deadline:
            break
            
        # Try a few restarts for this rank
        found = False
        # More restarts for lower ranks
        if r < 30:
            restarts = 8
        elif r < 40:
            restarts = 5
        elif r < 50:
            restarts = 3
        else:
            restarts = 2
        
        for _ in range(restarts):
            if time.time() > deadline:
                break
                
            np.random.seed(r * 100 + _)
            U = np.random.randn(U_dims, r)
            V = np.random.randn(V_dims, r)
            W = np.random.randn(W_dims, r)
            
            # Use more iterations for lower ranks
            iters = 60 if r < 30 else 40
            
            U, V, W = al_solve(U, V, W, r, iters=iters)
            
            # Check error before rounding
            err = compute_error(U, V, W)
            
            if err < 1e-4:
                Ur = round_to_half(U)
                Vr = round_to_half(V)
                Wr = round_to_half(W)
                
                # Convert to lists for verification
                U_list = Ur.tolist()
                V_list = Vr.tolist()
                W_list = Wr.tolist()
                
                if verify_decomp(U_list, V_list, W_list):
                    return {"U": U_list, "V": V_list, "W": W_list}
                
        # If not found, continue to next rank

    # Fallback to schoolbook
    rank = rank_school
    U = np.zeros((U_dims, rank))
    V = np.zeros((V_dims, rank))
    W = np.zeros((W_dims, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                U[u, col] = 1
                V[v, col] = 1
                W[w, col] = 1
                col += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
