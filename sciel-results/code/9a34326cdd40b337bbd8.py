"""Rewritten solver for matrix multiplication M_{3,3,7} rank-minimal decomposition."""
import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    deadline = start + 145.0  # safety margin under 150s wall-clock

    n, m, p = DIMENSIONS
    dim_u = n * m  # 9
    dim_v = m * p  # 21
    dim_w = p * n  # 21
    
    # Precompute target tensor T[a,b,c]
    # T[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i
    T = np.zeros((dim_u, dim_v, dim_w), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    def verify(U, V, W, tol=1e-9):
        """Check tensor decomposition validity for M_{n,m,p}."""
        # Check if entries are half-integers
        U_2 = U * 2
        V_2 = V * 2
        W_2 = W * 2
        if not np.allclose(U_2, np.round(U_2), atol=tol): return False
        if not np.allclose(V_2, np.round(V_2), atol=tol): return False
        if not np.allclose(W_2, np.round(W_2), atol=tol): return False
        
        # Round to half-integers
        U_r = np.round(U_2) / 2.0
        V_r = np.round(V_2) / 2.0
        W_r = np.round(W_2) / 2.0
        
        # Reconstruct tensor
        # T_rec[a,b,c] = sum_r U_r[a,r] V_r[b,r] W_r[c,r]
        # Efficient check:
        for a in range(dim_u):
            # M_a = V_r @ diag(U_r[a]) @ W_r.T
            # We want M_a[b,c] == T[a,b,c]
            # M_a = V_r * U_r[a] (broadcasting) then dot with W_r?
            # V_r is (dim_v, R). U_r[a] is (R,).
            # prod = V_r * U_r[a] -> (dim_v, R)
            # M_a = prod @ W_r.T -> (dim_v, dim_w)
            M_a = (V_r * U_r[a]) @ W_r.T
            if not np.allclose(M_a, T[a], atol=1e-9):
                return False
        return True

    def objective_and_grads(x, U_shape, V_shape, W_shape, T_target):
        """
        x is flattened [U, V, W]
        Returns loss and gradient.
        """
        U_size = U_shape[0] * U_shape[1]
        V_size = V_shape[0] * V_shape[1]
        W_size = W_shape[0] * W_shape[1]
        
        U = x[:U_size].reshape(U_shape)
        V = x[U_size:U_size+V_size].reshape(V_shape)
        W = x[U_size+V_size:U_size+V_size+W_size].reshape(W_shape)
        
        R = U.shape[1]
        
        # Compute Error tensor E
        # E[a,b,c] = sum_r U[a,r] V[b,r] W[c,r] - T[a,b,c]
        # We can compute this by iterating a, or using einsum
        # T_rec = np.einsum('ar,br,cr->abc', U, V, W)
        # This is O(R * dim_u * dim_v * dim_w) = O(R * 9 * 21 * 21) ~ 4000 * R.
        # For R=40, ~160k ops. Very fast.
        T_rec = np.einsum('ar,br,cr->abc', U, V, W)
        E = T_rec - T_target
        
        loss = np.sum(E * E)
        
        # Gradients
        # dL/dU[a,r] = 2 * sum_{b,c} E[a,b,c] V[b,r] W[c,r]
        # This is 2 * (E * (V[:,:,None] * W[None,:,:])) sum over b,c?
        # Let's use einsum for gradients too.
        # grad_U[a,r] = 2 * einsum('abc,br,cr->ar', E, V, W)
        grad_U = 2.0 * np.einsum('abc,br,cr->ar', E, V, W)
        grad_V = 2.0 * np.einsum('abc,ar,cr->br', E, U, W)
        grad_W = 2.0 * np.einsum('abc,ar,br->cr', E, U, V)
        
        grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
        
        return loss, grad

    def try_rank(R, seed, max_iters=500):
        """Attempt to find a rank-R decomposition."""
        if time.time() > deadline - 10:
            return None
            
        rng = np.random.RandomState(seed)
        
        # Initialize
        # Scale: Target entries are 1. U,V,W entries should be around 1/sqrt(R) or similar?
        # If U,V,W are ~1, T_rec ~ R. We want T_rec ~ 1. So scale by 1/sqrt(R)?
        # Actually, if we use random init, the scale doesn't matter much for L-BFGS, 
        # but it helps convergence.
        scale = 1.0 / np.sqrt(R)
        
        U = rng.randn(dim_u, R) * scale
        V = rng.randn(dim_v, R) * scale
        W = rng.randn(dim_w, R) * scale
        
        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        U_shape = (dim_u, R)
        V_shape = (dim_v, R)
        W_shape = (dim_w, R)
        
        # Use L-BFGS-B
        try:
            res = minimize(
                objective_and_grads, 
                x0, 
                args=(U_shape, V_shape, W_shape, T),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': max_iters, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False}
            )
        except Exception:
            return None
            
        if res.success or res.fun < 1e-6:
            x_opt = res.x
            U_opt = x_opt[:U_size].reshape(U_shape)
            V_opt = x_opt[U_size:U_size+V_size].reshape(V_shape)
            W_opt = x_opt[U_size+V_size:U_size+V_size+W_size].reshape(W_shape)
            
            # Polish: Round to half-integers and verify
            # Sometimes the optimizer gets close but not exact.
            # We try rounding to nearest half-integer.
            U_r = np.round(U_opt * 2) / 2.0
            V_r = np.round(V_opt * 2) / 2.0
            W_r = np.round(W_opt * 2) / 2.0
            
            if verify(U_r, V_r, W_r):
                return U_r, V_r, W_r
            
            # If direct rounding fails, try a local polish with tighter constraints?
            # Or just accept if loss is extremely small (might be floating point noise in verify?)
            # Verify uses 1e-9.
            loss = res.fun
            if loss < 1e-8:
                # Try to snap to integers/half-integers more aggressively?
                # If loss is this low, it's likely a valid solution with slight FP error.
                # Let's try rounding again, maybe with different tol?
                # Or just return the rounded version if it passes a looser check?
                # The task requires EXACT decomposition.
                # If verify fails, we can't return it.
                
                # Try a few more steps of gradient descent on the rounded values?
                # Or just return None.
                
                # Let's try to see if it's close to a half-integer solution.
                # Calculate distance to nearest half-integer
                dist = np.max(np.abs(U_opt * 2 - np.round(U_opt * 2))) + \
                       np.max(np.abs(V_opt * 2 - np.round(V_opt * 2))) + \
                       np.max(np.abs(W_opt * 2 - np.round(W_opt * 2)))
                if dist < 1e-5:
                     # It's effectively a half-integer solution
                     if verify(U_r, V_r, W_r, tol=1e-7):
                          return U_r, V_r, W_r
        return None

    best_result = None
    best_rank = 63 # Trivial
    
    # Trivial decomposition for fallback
    U_t = np.zeros((dim_u, 63), dtype=float)
    V_t = np.zeros((dim_v, 63), dtype=float)
    W_t = np.zeros((dim_w, 63), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_t[i * m + j, col] = 1.0
                V_t[j * p + k, col] = 1.0
                W_t[k * n + i, col] = 1.0
                col += 1

    # Try ranks
    # Known bounds: R(3,3,3)=23. R(3,3,7) is likely around 30-40.
    # We scan from low to high. If we find one, we keep it and try lower? 
    # No, if we find rank 40, we should try 39, 38... until we fail or time out.
    # But optimization is expensive.
    # Strategy: Try a set of candidate ranks with multiple seeds.
    # Prioritize lower ranks.
    
    candidate_ranks = [25, 30, 35, 40, 45, 50, 55, 60]
    seeds = [42, 123, 7, 99, 1234, 5678, 9012, 3456, 777, 888]
    
    # To save time, we can stop trying a rank if we find a solution, 
    # and move to a lower rank.
    # But we don't know if a lower rank exists.
    # We will iterate ranks in increasing order.
    # If we find a solution for R, we set best_result and try R-1?
    # That might be too slow if R-1 doesn't exist.
    # Better: Try all candidate ranks, keep the best (lowest) found.
    
    found_ranks = set()
    
    for R in candidate_ranks:
        if time.time() > deadline - 20:
            break
            
        # If we already found a lower rank, skip higher ones? 
        # No, we are iterating low to high. 
        # If we find R=30, we should break and return 30? 
        # No, maybe R=25 exists. We should have tried 25 first.
        # So if we iterate 25, 30, 35..., and find 30, we should continue to see if 25 was missed?
        # No, we tried 25 first. If 25 failed, we try 30.
        # If 30 succeeds, we have a rank 30 solution.
        # Do we try 29? Probably not worth it if 25-29 are unlikely.
        # Let's just take the first success in the increasing order.
        
        success = False
        for seed in seeds:
            if time.time() > deadline - 10:
                break
            res = try_rank(R, seed, max_iters=200)
            if res is not None:
                U_r, V_r, W_r = res
                best_result = (U_r, V_r, W_r, R)
                success = True
                break
        
        if success:
            # We found a solution for this R. 
            # Since we iterate R in increasing order, this is the minimum found so far.
            # We can break here to return the best (lowest) rank found.
            break

    if best_result is not None:
        U, V, W, _ = best_result
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback
    return {"U": U_t.tolist(), "V": V_t.tolist(), "W": W_t.tolist()}
# EVOLVE-BLOCK-END
