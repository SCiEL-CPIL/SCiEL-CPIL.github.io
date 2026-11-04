"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time
import sys

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    deadline = time.time() + 140.0  # 10s safety margin before 150s kill
    
    # Build the matrix multiplication tensor T[i, j, k] = 1 if we multiply A[i,j]*B[j,k]
    # The tensor for matrix mult C = A * B where C[i,k] = sum_j A[i,j] B[j,k]
    # T is a 3-tensor of shape (n, m, p) where T[i,j,k] = 1 for all valid i,j,k
    # Actually, the standard tensor decomposition: C[i,k] = sum_r U[i,r] V[k,r] W[r,j]
    # Wait, let me re-read the contract.
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # The tensor is T[a, b, c] where a indexes A (n*m), b indexes B (m*p), c indexes C (p*n)
    # T[a, b, c] = 1 if the multiplication contributes, 0 otherwise.
    # For A[i,j] (a = i*m+j) * B[j,k] (b = j*p+k) -> C[k,i] (c = k*n+i)
    # So T[a, b, c] = 1 if j matches in a and b, and c = k*n + i where i from a and k from b.
    
    # Let's build the tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # We want to find U (n*m x r), V (m*p x r), W (p*n x r) such that:
    # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # This is a tensor decomposition problem.
    
    # Start with the schoolbook decomposition as initial guess
    def make_schoolbook():
        r = n * m * p
        U = np.zeros((n*m, r))
        V = np.zeros((m*p, r))
        W = np.zeros((p*n, r))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1.0
                    V[j*p + k, col] = 1.0
                    W[k*n + i, col] = 1.0
                    col += 1
        return U, V, W
    
    U0, V0, W0 = make_schoolbook()
    
    # Try to reduce rank by optimizing a smaller decomposition
    # We'll try ranks from 60 down to some minimum
    # The known lower bound for 4x5x6 matrix mult is not well known, but schoolbook is 120.
    # Let's try to find a decomposition with rank < 120.
    
    def residual(U, V, W, r):
        # Compute sum_r U[:,r] V[:,r] W[:,r] and compare to T
        # This is expensive, so we'll use a smarter approach
        # T_decomposed[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # We can compute this as: for each r, outer(U[:,r], V[:,r]) then contract with W
        # Or: T_decomp = sum_r np.einsum('ar,br,cr->abc', U, V, W)
        # But that's O(n*m * m*p * p*n * r) which is huge.
        # Better: T_decomp = np.einsum('ar,br,cr->abc', U, V, W)
        # Actually, let's use a different approach.
        # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        # This can be computed as: for fixed a,b, T[a,b,:] = sum_r U[a,r] V[b,r] W[:,r]
        # = (U[a,:] * V[b,:]) @ W.T
        # So T_decomp[a,b,:] = (U[a,:] * V[b,:]) @ W.T
        # And we want this to equal T[a,b,:]
        
        # Let's compute the residual efficiently
        # T_decomp = np.zeros((n*m, m*p, p*n))
        # for a in range(n*m):
        #     for b in range(m*p):
        #         T_decomp[a,b,:] = (U[a,:] * V[b,:]) @ W.T
        # That's O(n*m * m*p * p*n * r) = O((n*m)*(m*p)*(p*n)*r)
        # For n=4,m=5,p=6: 20*30*24*r = 14400*r
        # For r=120, that's 1.7 million operations per residual evaluation.
        # That's manageable.
        
        # Actually, we can vectorize this:
        # T_decomp[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        # = np.einsum('ar,br,cr->abc', U, V, W)
        # But einsum might be slow for large arrays.
        # Let's try: T_decomp = np.einsum('ar,br,cr->abc', U, V, W)
        
        # Alternative: T_decomp = np.tensordot(U, V, axes=([1],[1])) then multiply by W
        # No, that's not right.
        
        # Let's just use the loop approach but vectorize over c:
        # For each a,b: T_decomp[a,b,:] = (U[a,:] * V[b,:]) @ W.T
        # We can do this as:
        # UV = U[:, :, None] * V[None, :, :]  # shape (n*m, m*p, r)
        # T_decomp = np.einsum('abr,cr->abc', UV, W)
        # That's O(n*m * m*p * r * p*n) which is the same.
        
        # Let's try the einsum approach:
        T_decomp = np.einsum('ar,br,cr->abc', U, V, W)
        return np.sum((T_decomp - T) ** 2)
    
    # We'll use a local search approach: start with schoolbook, try to zero out columns
    # and re-optimize the remaining entries.
    
    # Actually, let's try a different approach: use the known structure.
    # The matrix multiplication tensor has a lot of symmetry.
    # Let's try to find a decomposition with rank around 80-100.
    
    # For now, let's just return the schoolbook decomposition if we can't find better quickly.
    # But we should try to improve.
    
    # Let's try a simple approach: start with schoolbook, randomly set some columns to zero,
    # then use gradient descent to minimize the residual.
    
    def try_reduce_rank(U_init, V_init, W_init, target_rank, max_iters=100, lr=0.01):
        """Try to find a decomposition of rank target_rank by optimizing."""
        r = target_rank
        # Take the first r columns of the initial decomposition
        U = U_init[:, :r].copy()
        V = V_init[:, :r].copy()
        W = W_init[:, :r].copy()
        
        # Use gradient descent
        # The loss is L = sum_{a,b,c} (sum_r U[a,r]V[b,r]W[c,r] - T[a,b,c])^2
        # Gradients:
        # dL/dU[a,r] = 2 * sum_{b,c} (T_decomp[a,b,c] - T[a,b,c]) * V[b,r] * W[c,r]
        # = 2 * V[:,r] @ (T_decomp[a,:,:] - T[a,:,:]) @ W[:,r].T
        # Wait, let's be more careful.
        # T_decomp[a,b,c] = sum_r' U[a,r'] V[b,r'] W[c,r']
        # dT_decomp[a,b,c]/dU[a,r] = V[b,r] W[c,r]
        # dL/dU[a,r] = 2 * sum_{b,c} (T_decomp[a,b,c] - T[a,b,c]) V[b,r] W[c,r]
        # = 2 * V[:,r] @ ((T_decomp[a,:,:] - T[a,:,:]) * W[:,r][None,:]).T
        # Hmm, this is getting complicated. Let's just compute the residual and use numerical gradients
        # or a simpler approach.
        
        # Actually, let's use a different strategy: alternating least squares.
        # Fix V and W, solve for U:
        # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        # For fixed a: T[a,b,c] = sum_r U[a,r] (V[b,r] W[c,r])
        # This is a linear system in U[a,:].
        # Let M_a be a matrix where M_a[(b,c), r] = V[b,r] W[c,r]
        # Then T[a,b,c] = sum_r M_a[(b,c), r] U[a,r]
        # So U[a,:] = (M_a^T M_a)^{-1} M_a^T T[a,:,:].ravel()
        
        # This is ALS. Let's implement it.
        
        def als_step(U, V, W, r, max_inner=10):
            for _ in range(max_inner):
                if time.time() > deadline:
                    break
                # Update U
                for a in range(n*m):
                    M_a = np.outer(V.ravel(), W.ravel())  # shape (m*p*p*n, r)
                    # Actually, M_a[(b,c), r] = V[b,r] * W[c,r]
                    # So M_a = np.einsum('br,cr->bcr', V, W) reshaped to (m*p*p*n, r)
                    M_a = np.einsum('br,cr->bcr', V, W).reshape(m*p*p*n, r)
                    t_a = T[a].reshape(-1)
                    # Solve: M_a @ U[a,:] = t_a
                    # Least squares: U[a,:] = (M_a^T M_a)^{-1} M_a^T t_a
                    try:
                        U[a, :] = np.linalg.lstsq(M_a, t_a, rcond=None)[0]
                    except:
                        pass
                # Update V
                for b in range(m*p):
                    M_b = np.einsum('ar,cr->acr', U, W).reshape(n*m*p*n, r)
                    t_b = T[:, b, :].reshape(-1)
                    try:
                        V[b, :] = np.linalg.lstsq(M_b, t_b, rcond=None)[0]
                    except:
                        pass
                # Update W
                for c in range(p*n):
                    M_c = np.einsum('ar,br->abr', U, V).reshape(n*m*m*p, r)
                    t_c = T[:, :, c].reshape(-1)
                    try:
                        W[c, :] = np.linalg.lstsq(M_c, t_c, rcond=None)[0]
                    except:
                        pass
            return U, V, W
        
        U, V, W = als_step(U, V, W, r, max_inner=5)
        return U, V, W, residual(U, V, W, r)
    
    # Try to reduce rank
    best_U, best_V, best_W = U0, V0, W0
    best_rank = rank_schoolbook
    best_res = 0.0
    
    # Try a few target ranks
    for target_rank in [100, 90, 80, 70, 60]:
        if time.time() > deadline - 30:
            break
        U_t, V_t, W_t, res = try_reduce_rank(U0, V0, W0, target_rank, max_iters=50)
        if res < 1e-6:  # Found a valid decomposition
            best_U, best_V, best_W = U_t, V_t, W_t
            best_rank = target_rank
            best_res = res
            break
    
    # Convert to lists
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
