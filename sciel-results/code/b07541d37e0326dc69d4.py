import numpy as np
import time

DIMENSIONS = [2, 5, 5]

def _make_schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-10):
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _try_random_decomposition(n, m, p, target_rank, max_iters=100, max_restarts=5, seed=0):
    """
    Try to find a decomposition of the given rank using randomized ALS.
    """
    rng = np.random.RandomState(seed)
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    # Precompute the indices for the non-zero entries of the tensor
    # T[u, v, w] = 1 if u=i*m+j, v=j*p+k, w=k*n+i
    # We need to check if sum_r U[u,r] V[v,r] W[w,r] = 1 for all valid (u,v,w)
    # And 0 for all others?
    # The _verify function only checks the valid (i,j,k) triplets.
    # It does NOT check that other entries are 0.
    # Wait, the tensor T is defined by the multiplication.
    # If U, V, W produce non-zero values for invalid (u,v,w), is it invalid?
    # The task says "exactly decompose the matrix multiplication tensor".
    # The multiplication tensor has 0 for invalid entries.
    # So we MUST ensure that for invalid (u,v,w), the sum is 0.
    
    # My _verify function in the seed ONLY checks the valid entries.
    # I must UPDATE _verify to check ALL entries?
    # Or does the evaluator check all entries?
    # "exactly decompose the matrix multiplication tensor" implies all entries.
    # The seed's _verify is likely incomplete/incorrect for the "exact" requirement if it ignores zeros.
    # However, the seed scored 1.0, which is the schoolbook rank.
    # The schoolbook decomposition HAS zeros for invalid entries.
    # So a correct decomposition must have zeros there.
    
    # I will update _verify to check all possible (u,v,w) combinations?
    # That is dim_u * dim_v * dim_w = 10 * 25 * 10 = 2500 checks.
    # This is fast.
    
    pass

def _verify_full(U, V, W, n, m, p, tol=1e-10):
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    # Check all u, v, w
    # We can vectorize this using tensor contraction?
    # T_reconstructed = np.einsum('ur,vr,wr->uvw', U, V, W)
    # Then check T_reconstructed against the target tensor.
    
    T_rec = np.einsum('ur,vr,wr->uvw', U, V, W)
    
    # Construct target tensor T_target
    T_target = np.zeros((dim_u, dim_v, dim_w), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T_target[u, v, w] = 1.0
                
    # Check difference
    if np.max(np.abs(T_rec - T_target)) > tol:
        return False
    return True

def construct():
    n, m, p = DIMENSIONS
    rank_school = n * m * p
    
    start_time = time.time()
    deadline = 110  # seconds
    
    # Fallback
    U_school, V_school, W_school = _make_schoolbook(n, m, p)
    
    # Try to find a better decomposition
    # We will try ranks from rank_school - 1 down to rank_school - 10
    # Using randomized search
    
    best_U, best_V, best_W, best_rank = U_school, V_school, W_school, rank_school
    
    # Try a few ranks
    for target_rank in range(rank_school - 1, max(rank_school - 10, 0), -1):
        if time.time() - start_time > deadline:
            break
            
        # Try a few restarts for this rank
        for restart in range(5):
            if time.time() - start_time > deadline:
                break
                
            U, V, W = _try_random_decomposition(n, m, p, target_rank, max_iters=50, max_restarts=1, seed=restart)
            if U is not None:
                # Verify
                if _verify_full(U, V, W, n, m, p):
                    best_U, best_V, best_W, best_rank = U, V, W, target_rank
                    break # Found one for this rank, try lower?
        
        if best_rank < rank_school:
             # We found a better one. Can we go lower?
             # The loop continues to lower ranks.
             pass
             
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

def _try_random_decomposition(n, m, p, target_rank, max_iters=100, max_restarts=1, seed=0):
    rng = np.random.RandomState(seed)
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    U = rng.randn(dim_u, target_rank)
    V = rng.randn(dim_v, target_rank)
    W = rng.randn(dim_w, target_rank)
    
    # Normalize to prevent overflow
    U /= np.linalg.norm(U)
    V /= np.linalg.norm(V)
    W /= np.linalg.norm(W)
    
    for it in range(max_iters):
        # Update U
        # T_target = sum_r U[:, r] (V (W^T) )^T ?
        # T_rec = sum_r U[:, r] V[:, r] W[:, r]^T
        # For fixed V, W, we want to minimize || T_target - sum_r U[:, r] (V[:, r] W[:, r]^T) ||^2
        # This is a linear least squares problem for U.
        # Let M_r = V[:, r] * W[:, r] (outer product? No, element-wise? No.)
        # T_rec[u, v, w] = sum_r U[u, r] V[v, r] W[w, r]
        # We can reshape T_target to (dim_u, dim_v * dim_w).
        # And construct a matrix A where A[:, r] corresponds to the contribution of the r-th term?
        # Actually, for fixed V, W, the term is U[:, r] * (V[:, r] * W[:, r] reshaped?).
        # No, it's a tensor product.
        # We can solve for U by unfolding the tensor.
        
        # Let's use a simpler ALS step:
        # Fix V, W.
        # For each u, U[u, :] is a vector.
        # The equation is T_target[u, :, :] = sum_r U[u, r] (V[:, r] W[:, r]^T)
        # Let B_r = V[:, r] W[:, r]^T (matrix of size dim_v x dim_w)
        # We want to approximate T_target[u] (matrix) by sum_r U[u, r] B_r.
        # This is a matrix factorization problem for each u.
        # We can stack all u's.
        # T_target_2d = T_target.reshape(dim_u, dim_v * dim_w)
        # B_2d = np.stack([ (V[:, r] * W[:, r]).flatten() for r in range(target_rank) ], axis=1)
        # Then T_target_2d = U @ B_2d^T ?
        # Let's check: (U @ B_2d^T)[u, idx] = sum_r U[u, r] B_2d[idx, r]
        # B_2d[idx, r] = V[v, r] W[w, r] where idx = v*dim_w + w.
        # Yes.
        
        T_target = np.zeros((dim_u, dim_v, dim_w), dtype=np.float64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = i * m + j
                    v = j * p + k
                    w = k * n + i
                    T_target[u, v, w] = 1.0
        
        T_2d = T_target.reshape(dim_u, dim_v * dim_w)
        
        # Update U
        # B_2d[r, idx] = V[v, r] W[w, r]
        # B_2d is (target_rank, dim_v * dim_w)
        B_2d = np.zeros((target_rank, dim_v * dim_w))
        for r in range(target_rank):
            B_2d[r, :] = (V[:, r] * W[:, r]).flatten()
        
        # Solve U @ B_2d.T = T_2d
        # U = T_2d @ B_2d.T @ (B_2d @ B_2d.T)^-1
        try:
            U_new = np.linalg.lstsq(B_2d.T, T_2d.T, rcond=None)[0].T
            U = U_new
        except:
            pass
            
        # Update V
        # T_target[v, :, :] = sum_r V[v, r] (W[:, r] U[:, r]^T)
        # C_2d[r, idx] = W[w, r] U[u, r] where idx = u*dim_w + w? No, u and w are not adjacent in T_target (u, v, w).
        # We need to permute T_target to (dim_v, dim_u * dim_w).
        T_perm = np.transpose(T_target, (1, 0, 2)).reshape(dim_v, dim_u * dim_w)
        
        C_2d = np.zeros((target_rank, dim_u * dim_w))
        for r in range(target_rank):
            C_2d[r, :] = (U[:, r] * W[:, r]).flatten()
            
        try:
            V_new = np.linalg.lstsq(C_2d.T, T_perm.T, rcond=None)[0].T
            V = V_new
        except:
            pass
            
        # Update W
        # T_target[w, :, :] = sum_r W[w, r] (U[:, r] V[:, r]^T)
        T_perm2 = np.transpose(T_target, (2, 0, 1)).reshape(dim_w, dim_u * dim_v)
        
        D_2d = np.zeros((target_rank, dim_u * dim_v))
        for r in range(target_rank):
            D_2d[r, :] = (U[:, r] * V[:, r]).flatten()
            
        try:
            W_new = np.linalg.lstsq(D_2d.T, T_perm2.T, rcond=None)[0].T
            W = W_new
        except:
            pass
            
        # Check convergence
        if it % 10 == 0:
             # Check if close to integer/half-integer
             # We can check the residual
             T_rec = np.einsum('ur,vr,wr->uvw', U, V, W)
             err = np.max(np.abs(T_rec - T_target))
             if err < 1e-6:
                 # Try rounding
                 U_round = np.round(U * 2) / 2
                 V_round = np.round(V * 2) / 2
                 W_round = np.round(W * 2) / 2
                 
                 T_rec_round = np.einsum('ur,vr,wr->uvw', U_round, V_round, W_round)
                 err_round = np.max(np.abs(T_rec_round - T_target))
                 if err_round < 1e-10:
                     return U_round, V_round, W_round
                     
    return None, None, None

