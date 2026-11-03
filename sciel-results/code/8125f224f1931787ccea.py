import numpy as np
import time
import random

DIMENSIONS = [3, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    # Shapes: U (n*m, R), V (m*p, R), W (p*n, R)
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    # Build target tensor T of shape (dim_u, dim_v, dim_w)
    # T[u, v, w] = 1 if the indices match the multiplication rule
    # u = i*m + j
    # v = j*p + k
    # w = k*n + i
    # Condition: j_u == j_v and k_v == k_w and i_u == i_w
    
    T = np.zeros((dim_u, dim_v, dim_w), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0
                
    # Seed decomposition (Rank 48)
    rank_seed = n * m * p
    U_seed = np.zeros((dim_u, rank_seed))
    V_seed = np.zeros((dim_v, rank_seed))
    W_seed = np.zeros((dim_w, rank_seed))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                U_seed[u, col] = 1
                V_seed[v, col] = 1
                W_seed[w, col] = 1
                col += 1
                
    best_result = {"U": U_seed.tolist(), "V": V_seed.tolist(), "W": W_seed.tolist()}
    best_rank = rank_seed
    
    start_time = time.time()
    deadline = start_time + 115.0 # 115s to be safe
    
    # Try to find a decomposition with rank < 48
    # We will try ranks from 47 down to 30
    # Using Alternating Least Squares (ALS)
    
    def try_rank(R, max_iter=200, restarts=50):
        if time.time() > deadline:
            return None
            
        for r in range(restarts):
            if time.time() > deadline:
                break
                
            # Initialize
            U = np.random.randn(dim_u, R) * 0.1
            V = np.random.randn(dim_v, R) * 0.1
            W = np.random.randn(dim_w, R) * 0.1
            
            # Pre-compute unfoldings
            T_u = T.reshape(dim_u, dim_v * dim_w)
            T_v = T.transpose(1, 0, 2).reshape(dim_v, dim_u * dim_w)
            T_w = T.transpose(2, 0, 1).reshape(dim_w, dim_u * dim_v)
            
            for it in range(max_iter):
                # Update U
                # T_u = U * (V khronos W)
                # (V khronos W) has shape (dim_v * dim_w, R)
                VW = (V * W.T).T # This is wrong. Khatri-Rao product
                # Khatri-Rao of V (dim_v x R) and W (dim_w x R)
                # Result is (dim_v * dim_w) x R
                # Column r is V[:, r] * W[:, r] (element-wise)
                # V[:, r] has length dim_v. W[:, r] has length dim_w.
                # The product vector has length dim_v * dim_w.
                # We can compute this as:
                # M = np.zeros((dim_v * dim_w, R))
                # for r in range(R):
                #     M[:, r] = np.kron(V[:, r], W[:, r]) # No, element-wise product mapped to indices
                # Actually, T_u[u, (v, w)] = sum_r U[u, r] V[v, r] W[w, r]
                # Let M = V khronos W.
                # M is a matrix of size (dim_v * dim_w) x R.
                # M[v * dim_w + w, r] = V[v, r] * W[w, r]
                
                # Efficient construction of M:
                # M = np.einsum('vr,wr->vwr', V, W).reshape(dim_v * dim_w, R)
                M = np.einsum('vr,wr->vwr', V, W).reshape(dim_v * dim_w, R)
                # Solve U = T_u M^+
                # Use least squares
                try:
                    U, residuals, rank, s = np.linalg.lstsq(M, T_u.T, rcond=None)
                except:
                    U = np.random.randn(dim_u, R) * 0.1
                    
                # Update V
                # T_v = V * (W khronos U)
                # M = W khronos U
                # M[k * dim_u + i, r] = W[k, r] * U[i, r] ?
                # T_v[v, (u, w)] = sum_r V[v, r] W[w, r] U[u, r]
                # Indices: v is row. (u, w) are cols.
                # M[u * dim_w + w, r] = U[u, r] * W[w, r]
                M = np.einsum('ur,wr->uwr', U, W).reshape(dim_u * dim_w, R)
                try:
                    V, residuals, rank, s = np.linalg.lstsq(M, T_v.T, rcond=None)
                except:
                    V = np.random.randn(dim_v, R) * 0.1
                    
                # Update W
                # T_w = W * (U khronos V)
                # M[u * dim_v + v, r] = U[u, r] * V[v, r]
                M = np.einsum('ur,vr->uvr', U, V).reshape(dim_u * dim_v, R)
                try:
                    W, residuals, rank, s = np.linalg.lstsq(M, T_w.T, rcond=None)
                except:
                    W = np.random.randn(dim_w, R) * 0.1
                    
            # Check error
            # Compute reconstructed tensor
            # T_recon = np.einsum('ur,vr,wr->uvw', U, V, W)
            # err = np.linalg.norm(T - T_recon)
            
            # Faster check:
            # Calculate error on a subset or full? Full is small.
            T_recon = np.einsum('ur,vr,wr->uvw', U, V, W)
            err = np.linalg.norm(T - T_recon)
            
            if err < 1e-8:
                # Try to round to half-integers
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                
                T_recon_r = np.einsum('ur,vr,wr->uvw', U_r, V_r, W_r)
                err_r = np.linalg.norm(T - T_recon_r)
                
                if err_r < 1e-9:
                    return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist(), "rank": R}
                
                # If rounding failed, maybe the solution is real but valid?
                # Task says "integer/half-integer entries".
                # If we can't round, we can't use this solution.
                # But maybe a different initialization yields integer/half-integer?
                # Or maybe we should just return the real one if the evaluator accepts it?
                # "integer/half-integer entries" is a constraint.
                # So we must return integer/half-integer.
                
        return None

    # Search for lower ranks
    for R in range(rank_seed - 1, 25, -1):
        if time.time() > deadline:
            break
        result = try_rank(R)
        if result:
            best_result = {k: v for k, v in result.items() if k != "rank"}
            best_rank = R
            # If we found a very good one, we might stop early?
            # But we want minimal rank. Continue searching lower?
            # Given time, maybe stop if R is significantly lower?
            # Let's continue to see if we can go lower, but prioritize finding ANY.
            # If we find R=47, we are happy.
            # If we find R=40, we are happier.
            # But finding lower is harder.
            # Let's break after first success to save time for verification?
            # No, let's try a few more iterations to see if we can go lower.
            # Actually, if we find one, we can try to refine it or search lower.
            # But for now, let's just return the best found.
            break # Found one, let's return it. Searching for even lower might take too long.

    return best_result

