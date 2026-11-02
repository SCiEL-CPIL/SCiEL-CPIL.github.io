"""Combined program: ALS search for rank 29, fallback to schoolbook rank 30."""
import numpy as np
import time

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Target Tensor T
    # T[u, v, w] = 1 if u, v, w correspond to a valid (i, j, k)
    # u = i*m + j
    # v = j*p + k
    # w = k*n + i
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0

    def validate(U, V, W, tol=1e-9):
        """Check if U, V, W decompose T exactly."""
        # U: (nm, R), V: (mp, R), W: (pn, R)
        # Compute tensor product sum
        # Efficient check:
        # For each c, we have rank-1 tensor U[:,c] x V[:,c] x W[:,c]
        # We want sum_c ... == T
        # We can compute the reconstructed tensor and compare.
        R = U.shape[1]
        T_rec = np.zeros_like(T)
        for c in range(R):
            # Outer product of 3 vectors
            # (nm, 1, 1) * (1, mp, 1) * (1, 1, pn)
            T_rec += np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])
        
        # Check difference
        if np.allclose(T_rec, T, atol=tol):
            return True
        return False

    def try_rank(rank, max_iters=500, seed=0, time_limit=110):
        """
        Try to find a decomposition of given rank using ALS.
        Returns (U, V, W) if successful, else None.
        """
        rng = np.random.RandomState(seed)
        
        # Initialize
        U = rng.randn(nm, rank)
        V = rng.randn(mp, rank)
        W = rng.flatten() # No, W is (pn, rank)
        W = rng.randn(pn, rank)
        
        # Precompute indices for T
        # We only need to update U, V, W based on T
        # For U update: for each u, solve least squares
        # For V update: for each v, solve least squares
        # For W update: for each w, solve least squares
        
        best_err = float('inf')
        
        start_time = time.time()
        
        for it in range(max_iters):
            if time.time() > time_limit:
                break
            
            # Update U
            for u in range(nm):
                # We want to minimize || T[u, :, :] - sum_c U[u,c] (V[:,c] * W[:,c]) ||^2
                # Let A be a matrix of size (mp*pn, rank)
                # A[idx, c] = V[v, c] * W[w, c] where idx = v*pn + w
                # b = T[u, :, :].flatten()
                
                # Construct A
                # V is (mp, rank), W is (pn, rank)
                # We need the element-wise product of V and W broadcasted?
                # No, for a fixed u, the term is sum_c U[u,c] * V[v,c] * W[w,c]
                # So for each (v,w), the coefficient of U[u,c] is V[v,c] * W[w,c].
                
                # A shape: (mp * pn, rank)
                # Reshape V to (mp, 1, rank) and W to (1, pn, rank)?
                # Product: (mp, pn, rank)
                # Flatten to (mp*pn, rank)
                
                # Optimization: 
                # V[:, c] is a vector of length mp.
                # W[:, c] is a vector of length pn.
                # The outer product is (mp, pn).
                # We need this for each c.
                
                # A_c = np.outer(V[:, c], W[:, c])
                # A = np.stack([np.outer(V[:, c], W[:, c]).flatten() for c in range(rank)])
                # A = A.T # (mp*pn, rank)
                
                # Better:
                # A = np.einsum('vc,wc->vw c', V, W).reshape(mp * pn, rank)
                
                A = np.einsum('vc,wc->vwc', V, W).reshape(mp * pn, rank)
                b = T[u, :, :].flatten()
                
                # Solve A x = b
                # x is U[u, :]
                try:
                    sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    U[u, :] = sol
                except:
                    pass

            # Update V
            for v in range(mp):
                # Minimize || T[:, v, :] - sum_c V[v,c] (U[:,c] * W[:,c]) ||^2
                # A shape: (nm * pn, rank)
                # A[idx, c] = U[u, c] * W[w, c] where idx = u*pn + w
                
                A = np.einsum('uc,wc->uwc', U, W).reshape(nm * pn, rank)
                b = T[:, v, :].flatten()
                
                try:
                    sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    V[v, :] = sol
                except:
                    pass

            # Update W
            for w in range(pn):
                # Minimize || T[:, :, w] - sum_c W[w,c] (U[:,c] * V[:,c]) ||^2
                # A shape: (nm * mp, rank)
                # A[idx, c] = U[u, c] * V[v, c] where idx = u*mp + v
                
                A = np.einsum('uc,vc->uvc', U, V).reshape(nm * mp, rank)
                b = T[:, :, w].flatten()
                
                try:
                    sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    W[w, :] = sol
                except:
                    pass

            # Calculate error
            T_rec = np.zeros_like(T)
            for c in range(rank):
                T_rec += np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])
            
            err = np.linalg.norm(T_rec - T)
            
            if err < best_err:
                best_err = err
                # Store best
                U_best = U.copy()
                V_best = V.copy()
                W_best = W.copy()
                
            if err < 1e-10:
                # Found a solution. Try to round to half-integers.
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                
                if validate(U_r, V_r, W_r, tol=1e-9):
                    return U_r, V_r, W_r
                else:
                    # Try integer rounding
                    U_i = np.round(U)
                    V_i = np.round(V)
                    W_i = np.round(W)
                    if validate(U_i, V_i, W_i, tol=1e-9):
                        return U_i, V_i, W_i
                    else:
                        # If rounding fails, return the float solution? 
                        # The problem requires integer/half-integer.
                        # If we can't round, this solution is invalid for the task.
                        # But maybe it's close. Let's try a few more iterations or restart.
                        # For now, if we can't round, we treat it as failure for this seed.
                        pass
                        # Actually, if err is 0, but rounding fails, it's a valid real solution 
                        # but not a valid task solution.
                        # We should probably continue searching or return None.
                        # However, if we found a real solution of rank R, it proves R is achievable.
                        # But we need the specific format.
                        # Let's assume if we find a real solution, we can often perturb it 
                        # or find a nearby rational one. 
                        # For now, return None to trigger restart.
                        pass

        # If loop finishes without finding a valid rounded solution, return None
        # But if best_err is very small, maybe we can return the rounded version 
        # even if validate fails slightly? No, validate must pass.
        
        return None

    # Try rank 29
    # Use multiple seeds
    deadline = time.time() + 115 # Leave 5s for fallback
    
    for seed in range(20):
        if time.time() > deadline:
            break
        res = try_rank(29, max_iters=100, seed=seed, time_limit=deadline)
        if res is not None:
            U, V, W = res
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }

    # Fallback: Schoolbook rank 30
    rank = n * m * p
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, r] = 1.0
                V[j*p+k, r] = 1.0
                W[k*n+i, r] = 1.0
                r += 1
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
