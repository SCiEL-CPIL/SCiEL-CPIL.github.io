import numpy as np
import time

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    dim_u = n * m  # 10
    dim_v = m * p  # 25
    dim_w = p * n  # 10
    
    # Build the target tensor: T[a,b,c] = 1 if a = i*m+j, b = j*p+k, c = k*n+i
    target = np.zeros((dim_u, dim_v, dim_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                target[u_idx, v_idx, w_idx] = 1.0
    
    start_time = time.time()
    deadline = 135.0  # 15s safety margin before 150s kill
    
    def get_schoolbook():
        """Return schoolbook decomposition (rank = n*m*p = 50)."""
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
        return U, V, W
    
    def verify(U, V, W):
        """Check if U, V, W exactly decompose the target tensor."""
        T = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(T, target, atol=1e-9)
    
    def snap_to_half_int(arr):
        """Snap array to nearest half-integer."""
        return np.round(arr * 2.0) / 2.0
    
    def loss_func(params, rank):
        """Compute loss for continuous optimization."""
        u_flat = params[:dim_u * rank]
        v_flat = params[dim_u * rank : dim_u * rank + dim_v * rank]
        w_flat = params[dim_u * rank + dim_v * rank : ]
        
        U = u_flat.reshape(dim_u, rank)
        V = v_flat.reshape(dim_v, rank)
        W = w_flat.reshape(dim_w, rank)
        
        T = np.einsum('ar,br,cr->abc', U, V, W)
        diff = T - target
        return np.sum(diff * diff)
    
    def local_search_half_int(U, V, W, max_iter=200):
        """Local search: try changing one element at a time to nearby half-integers."""
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        best_loss = np.sum((np.einsum('ar,br,cr->abc', best_U, best_V, best_W) - target)**2)
        
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            
            improved = False
            
            # Try changing elements in U
            for a in range(dim_u):
                for r in range(U.shape[1]):
                    current = best_U[a, r]
                    for delta in [-1.0, -0.5, 0.5, 1.0]:
                        new_val = current + delta
                        if new_val != current:
                            test_U = best_U.copy()
                            test_U[a, r] = new_val
                            T = np.einsum('ar,br,cr->abc', test_U, best_V, best_W)
                            loss = np.sum((T - target)**2)
                            if loss < best_loss - 1e-12:
                                best_U = test_U
                                best_loss = loss
                                improved = True
            
            # Try changing elements in V
            for b in range(dim_v):
                for r in range(V.shape[1]):
                    current = best_V[b, r]
                    for delta in [-1.0, -0.5, 0.5, 1.0]:
                        new_val = current + delta
                        if new_val != current:
                            test_V = best_V.copy()
                            test_V[b, r] = new_val
                            T = np.einsum('ar,br,cr->abc', best_U, test_V, best_W)
                            loss = np.sum((T - target)**2)
                            if loss < best_loss - 1e-12:
                                best_V = test_V
                                best_loss = loss
                                improved = True
            
            # Try changing elements in W
            for c in range(dim_w):
                for r in range(W.shape[1]):
                    current = best_W[c, r]
                    for delta in [-1.0, -0.5, 0.5, 1.0]:
                        new_val = current + delta
                        if new_val != current:
                            test_W = best_W.copy()
                            test_W[c, r] = new_val
                            T = np.einsum('ar,br,cr->abc', best_U, best_V, test_W)
                            loss = np.sum((T - target)**2)
                            if loss < best_loss - 1e-12:
                                best_W = test_W
                                best_loss = loss
                                improved = True
            
            if not improved:
                break
        
        return best_U, best_V, best_W, best_loss
    
    best_result = None
    
    # Try ranks from 49 down to 40
    for target_rank in range(49, 39, -1):
        if time.time() > deadline:
            break
        
        found = False
        n_restarts = 20
        
        for restart in range(n_restarts):
            if time.time() > deadline:
                break
            
            np.random.seed(10000 + restart * 7)
            
            # Initialize with small random values
            u0 = np.random.randn(dim_u * target_rank) * 0.3
            v0 = np.random.randn(dim_v * target_rank) * 0.3
            w0 = np.random.randn(dim_w * target_rank) * 0.3
            
            params = np.concatenate([u0, v0, w0])
            
            # Use scipy.optimize if available, otherwise use simple gradient descent
            try:
                from scipy.optimize import minimize
                
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
                    
                    # Gradients
                    grad_u = 2.0 * np.einsum('abc,br,cr->ar', diff, V, W)
                    grad_v = 2.0 * np.einsum('abc,ar,cr->br', diff, U, W)
                    grad_w = 2.0 * np.einsum('abc,ar,br->cr', diff, U, V)
                    
                    grad = np.concatenate([grad_u.flatten(), grad_v.flatten(), grad_w.flatten()])
                    return loss, grad
                
                res = minimize(loss_and_grad, params, args=(target_rank,), method='L-BFGS-B', 
                              jac=True, options={'maxiter': 80, 'ftol': 1e-10, 'gtol': 1e-6})
                
                if res.fun < 1e-4:  # Good continuous solution
                    u_final = res.x[:dim_u * target_rank].reshape(dim_u, target_rank)
                    v_final = res.x[dim_u * target_rank : dim_u * target_rank + dim_v * target_rank].reshape(dim_v, target_rank)
                    w_final = res.x[dim_u * target_rank + dim_v * target_rank : ].reshape(dim_w, target_rank)
                    
                    # Phase 2: Snap to half-integers
                    u_snap = snap_to_half_int(u_final)
                    v_snap = snap_to_half_int(v_final)
                    w_snap = snap_to_half_int(w_final)
                    
                    if verify(u_snap, v_snap, w_snap):
                        found = True
                        best_result = (u_snap, v_snap, w_snap, target_rank)
                        break
                    
                    # Phase 3: Local search from snapped solution
                    u_ls, v_ls, w_ls, ls_loss = local_search_half_int(u_snap, v_snap, w_snap, max_iter=100)
                    
                    if verify(u_ls, v_ls, w_ls):
                        found = True
                        best_result = (u_ls, v_ls, w_ls, target_rank)
                        break
                        
            except Exception:
                continue
        
        if found:
            break
    
    # If not found, try schoolbook as fallback
    if best_result is not None:
        U, V, W, rank = best_result
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        U, V, W = get_schoolbook()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
