"""Find a rank-minimal exact tensor decomposition for 2x5 * 5x5 matrix multiplication."""
import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    dim_u = n * m  # 10
    dim_v = m * p  # 25
    dim_w = p * n  # 10
    
    # Build the target tensor
    # T[a, b, c] = 1 where a=i*m+j, b=j*p+k, c=k*n+i
    target = np.zeros((dim_u, dim_v, dim_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                target[u_idx, v_idx, w_idx] = 1.0
    
    def get_schoolbook():
        """Return schoolbook decomposition as dict of lists."""
        rank = n * m * p  # 50
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
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    def verify_decomposition(U, V, W, atol=1e-9):
        """Verify that U, V, W exactly decompose the target tensor."""
        T = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(T, target, atol=atol)
    
    def snap_and_verify(U, V, W, snap_to_half=True):
        """Snap to nearest integer or half-integer and verify."""
        if snap_to_half:
            U_snap = np.round(U * 2.0) / 2.0
            V_snap = np.round(V * 2.0) / 2.0
            W_snap = np.round(W * 2.0) / 2.0
        else:
            U_snap = np.round(U)
            V_snap = np.round(V)
            W_snap = np.round(W)
        
        if verify_decomposition(U_snap, V_snap, W_snap):
            # Clean up tiny values
            U_clean = np.where(np.abs(U_snap) < 1e-6, 0, U_snap)
            V_clean = np.where(np.abs(V_snap) < 1e-6, 0, V_snap)
            W_clean = np.where(np.abs(W_snap) < 1e-6, 0, W_snap)
            if verify_decomposition(U_clean, V_clean, W_clean):
                return (U_clean, V_clean, W_clean)
            return (U_snap, V_snap, W_snap)
        return None
    
    def loss_and_grad(params, rank):
        """Compute loss and gradient for given parameters and rank."""
        u_flat = params[:dim_u * rank]
        v_flat = params[dim_u * rank : dim_u * rank + dim_v * rank]
        w_flat = params[dim_u * rank + dim_v * rank : ]
        
        U = u_flat.reshape(dim_u, rank)
        V = v_flat.reshape(dim_v, rank)
        W = w_flat.reshape(dim_w, rank)
        
        T = np.einsum('ar,br,cr->abc', U, V, W)
        diff = T - target
        loss = np.sum(diff * diff)
        
        grad_u = 2.0 * np.einsum('abc,br,cr->ar', diff, V, W)
        grad_v = 2.0 * np.einsum('abc,ar,cr->br', diff, U, W)
        grad_w = 2.0 * np.einsum('abc,ar,br->cr', diff, U, V)
        
        grad = np.concatenate([grad_u.flatten(), grad_v.flatten(), grad_w.flatten()])
        return loss, grad
    
    def try_rank_with_init(rank, U_init, V_init, W_init, max_iter=50, ftol=1e-12):
        """Try to optimize from given initial point."""
        params = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        def fun(params):
            return loss_and_grad(params, rank)
        
        try:
            res = minimize(fun, params, method='L-BFGS-B', jac=True,
                           options={'maxiter': max_iter, 'ftol': ftol, 'gtol': 1e-10, 'maxls': 30})
            
            if res.fun < 1e-8:
                u_final = res.x[:dim_u * rank].reshape(dim_u, rank)
                v_final = res.x[dim_u * rank : dim_u * rank + dim_v * rank].reshape(dim_v, rank)
                w_final = res.x[dim_u * rank + dim_v * rank : ].reshape(dim_w, rank)
                
                # Try snapping to half-integers
                result = snap_and_verify(u_final, v_final, w_final, snap_to_half=True)
                if result is not None:
                    return result
                
                # Try snapping to integers
                result = snap_and_verify(u_final, v_final, w_final, snap_to_half=False)
                if result is not None:
                    return result
                
                # Try without snapping (in case exact solution is close to float)
                if verify_decomposition(u_final, v_final, w_final, atol=1e-10):
                    return (u_final, v_final, w_final)
                
        except Exception:
            pass
        return None
    
    def structured_init(rank, seed=0):
        """Create structured initializations exploiting tensor symmetry."""
        np.random.seed(seed)
        
        U = np.zeros((dim_u, rank))
        V = np.zeros((dim_v, rank))
        W = np.zeros((dim_w, rank))
        
        # Strategy 1: Assign random columns to cover the tensor structure
        # The tensor has entries at (i*m+j, j*p+k, k*n+i)
        # We want each column r to contribute to multiple such entries
        
        # For each column, pick a random (i,j,k) pattern and add contributions
        for r in range(rank):
            # Random base
            U[:, r] = np.random.randn(dim_u) * 0.3
            V[:, r] = np.random.randn(dim_v) * 0.3
            W[:, r] = np.random.randn(dim_w) * 0.3
            
            # Add structured components
            # Pick 2-3 random (i,j,k) triples to emphasize
            for _ in range(2):
                i = np.random.randint(n)
                j = np.random.randint(m)
                k = np.random.randint(p)
                U[i * m + j, r] += 1.0
                V[j * p + k, r] += 1.0
                W[k * n + i, r] += 1.0
        
        return U, V, W
    
    def random_init(rank, seed=0):
        """Create random initialization."""
        np.random.seed(seed)
        U = np.random.randn(dim_u, rank) * 0.5
        V = np.random.randn(dim_v, rank) * 0.5
        W = np.random.randn(dim_w, rank) * 0.5
        return U, V, W
    
    def try_rank(rank, max_restarts=15, max_iter=40):
        """Try to find a decomposition of given rank."""
        if time.time() > deadline:
            return None
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                return None
            
            seed = 42 + restart * 17 + rank * 31
            
            # Alternate between structured and random initializations
            if restart % 2 == 0:
                U_init, V_init, W_init = structured_init(rank, seed)
            else:
                U_init, V_init, W_init = random_init(rank, seed)
            
            result = try_rank_with_init(rank, U_init, V_init, W_init, max_iter=max_iter)
            if result is not None:
                return result
        
        return None
    
    # Try ranks from lower to upper bound
    # Theoretical lower bound is max(n*m, m*p, p*n) = 25
    # But matrix multiplication tensor rank is known to be less than n*m*p
    # For 2x5x5, let's try ranks from 30 upward
    
    best_result = None
    
    # Try promising ranks
    ranks_to_try = [30, 32, 35, 38, 40, 42, 45, 47, 48, 49]
    
    for target_rank in ranks_to_try:
        if time.time() > deadline:
            break
        
        # Adjust restarts based on time remaining
        time_remaining = deadline - time.time()
        if time_remaining < 10:
            break
        
        # More restarts for lower ranks (harder to find)
        max_restarts = min(20, max(5, int(time_remaining / 5)))
        
        result = try_rank(target_rank, max_restarts=max_restarts, max_iter=35)
        if result is not None:
            U, V, W = result
            best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            break
    
    if best_result is not None:
        return best_result
    else:
        return get_schoolbook()
# EVOLVE-BLOCK-END
