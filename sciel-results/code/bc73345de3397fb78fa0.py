import numpy as np
import time
from scipy.optimize import minimize
from itertools import product

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Shapes: U (n*m, rank), V (m*p, rank), W (p*n, rank)
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    # Build the target tensor M of shape (dim_u, dim_v, dim_w)
    # M[i*m+j, j*p+k, k*n+i] = 1 if consistent, else 0
    # Actually, the tensor is sparse.
    # T[a, b, c] = 1 iff a=i*m+j, b=j*p+k, c=k*n+i
    # We can build it by looping i,j,k
    target = np.zeros((dim_u, dim_v, dim_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                target[u_idx, v_idx, w_idx] = 1.0
                
    # Function to compute the tensor from U, V, W
    def compute_tensor(U, V, W):
        # T[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
        T = np.einsum('ar,br,cr->abc', U, V, W)
        return T

    def loss_and_grad(params, rank):
        u_flat = params[:dim_u * rank]
        v_flat = params[dim_u * rank : dim_u * rank + dim_v * rank]
        w_flat = params[dim_u * rank + dim_v * rank : ]
        
        U = u_flat.reshape(dim_u, rank)
        V = v_flat.reshape(dim_v, rank)
        W = w_flat.reshape(dim_w, rank)
        
        T = np.einsum('ar,br,cr->abc', U, V, W)
        diff = T - target
        loss = np.sum(diff * diff)
        
        # Gradient
        # grad_u[a, r] = 2 * sum_{b,c} diff[a, b, c] V[b, r] W[c, r]
        grad_u = 2.0 * np.einsum('abc,br,cr->ar', diff, V, W)
        grad_v = 2.0 * np.einsum('abc,ar,cr->br', diff, U, W)
        grad_w = 2.0 * np.einsum('abc,ar,br->cr', diff, U, V)
        
        grad = np.concatenate([grad_u.flatten(), grad_v.flatten(), grad_w.flatten()])
        return loss, grad

    # Try to find a decomposition with rank < 50
    # Start with rank 49
    # Use multiple random restarts
    start_time = time.time()
    deadline = 145.0 # 5 seconds margin
    
    best_result = None
    
    # Fallback: Schoolbook
    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((dim_u, rank))
        V = np.zeros((dim_v, rank))
        W = np.zeros((dim_w, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W, rank

    # Try rank 49 down to maybe 40
    # We will use a more robust search strategy:
    # 1. Random restarts with L-BFGS-B
    # 2. If that fails, try a grid search on a smaller subset or different initialization
    # 3. The key is that the solution must be half-integers.
    
    # Known lower bound for 2x5x5 is likely around 30-40?
    # Let's try to find the minimal rank.
    
    # To speed up, we can try to fix some variables or use a better initialization.
    # However, the current approach is generic.
    
    # Let's try a few specific ranks.
    # The schoolbook rank is 50.
    # Can we do better?
    
    # We will iterate from 49 down to 30.
    # For each rank, we do a few restarts.
    
    for target_rank in range(49, 30, -1):
        if time.time() > deadline:
            break
            
        found = False
        # Increase restarts for smaller ranks if time permits?
        # No, keep it fixed to ensure we finish.
        # Maybe try 20 restarts for the first few ranks?
        
        num_restarts = 20 if target_rank > 40 else 50
        
        for restart in range(num_restarts):
            if time.time() > deadline:
                break
                
            # Initialize randomly
            # Use a mix of random and structured initialization
            if restart % 2 == 0:
                # Random Gaussian
                np.random.seed(42 + restart)
                u0 = np.random.randn(dim_u * target_rank)
                v0 = np.random.randn(dim_v * target_rank)
                w0 = np.random.randn(dim_w * target_rank)
            else:
                # Random sparse initialization (many zeros)
                np.random.seed(42 + restart)
                u0 = np.random.randn(dim_u * target_rank)
                v0 = np.random.randn(dim_v * target_rank)
                w0 = np.random.randn(dim_w * target_rank)
                # Set 50% to zero
                mask_u = np.random.rand(dim_u, target_rank) > 0.5
                mask_v = np.random.rand(dim_v, target_rank) > 0.5
                mask_w = np.random.rand(dim_w, target_rank) > 0.5
                u0 = u0.reshape(dim_u, target_rank)
                v0 = v0.reshape(dim_v, target_rank)
                w0 = w0.reshape(dim_w, target_rank)
                u0[mask_u] = 0
                v0[mask_v] = 0
                w0[mask_w] = 0
                u0 = u0.flatten()
                v0 = v0.flatten()
                w0 = w0.flatten()
            
            params = np.concatenate([u0, v0, w0])
            
            # Use L-BFGS-B
            def fun(params):
                loss, grad = loss_and_grad(params, target_rank)
                return loss, grad
                
            try:
                # Use a tighter tolerance and more iterations if needed
                res = minimize(fun, params, method='L-BFGS-B', jac=True, 
                               options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8})
                
                # Check convergence
                if res.fun < 1e-8:
                    u_final = res.x[:dim_u * target_rank].reshape(dim_u, target_rank)
                    v_final = res.x[dim_u * target_rank : dim_u * target_rank + dim_v * target_rank].reshape(dim_v, target_rank)
                    w_final = res.x[dim_u * target_rank + dim_v * target_rank : ].reshape(dim_w, target_rank)
                    
                    # Snap to nearest 0.5
                    u_snap = np.round(u_final * 2.0) / 2.0
                    v_snap = np.round(v_final * 2.0) / 2.0
                    w_snap = np.round(w_final * 2.0) / 2.0
                    
                    # Verify snapped solution
                    T_snap = np.einsum('ar,br,cr->abc', u_snap, v_snap, w_snap)
                    if np.allclose(T_snap, target, atol=1e-9):
                        found = True
                        best_result = (u_snap, v_snap, w_snap, target_rank)
                        break
            except Exception:
                continue

        if found:
            break

    if best_result is not None:
        U, V, W, rank = best_result
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        U, V, W, rank = get_schoolbook()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
