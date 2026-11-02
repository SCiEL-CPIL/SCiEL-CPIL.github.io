import numpy as np
import time

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 2x2 * 2x6 matrix multiplication.
    Attempts rank 20 via algebraic search, falls back to rank 21 Strassen if not found.
    """
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s limit
    
    n, m, p = 2, 2, 6
    nm, mp, pn = n * m, m * p, p * n  # 4, 12, 12
    
    # Build the target tensor T with shape (4, 12, 12)
    # T[a, b, c] = 1 if there exist i,j,k such that:
    #   a = i*2+j (i in {0,1}, j in {0,1})
    #   b = j*6+k (j in {0,1}, k in {0,1,2,3,4,5})
    #   c = k*2+i (k in {0,1,2,3,4,5}, i in {0,1})
    # and the product A[i,j]*B[j,k] contributes to C[i,k]
    
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j       # index in U (A)
                b = j * p + k       # index in V (B)
                c = k * n + i       # index in W (C), trace convention W[k*n+i]
                T[a, b, c] = 1.0
    
    def try_rank(R, max_iters=5000, tol=1e-10):
        """Try to find rank-R decomposition using Newton's method."""
        # Initialize with random values
        np.random.seed(42 + R)
        U = np.random.randn(nm, R) * 0.5
        V = np.random.randn(mp, R) * 0.5
        W = np.random.randn(pn, R) * 0.5
        
        # Compute residual: for each (a,b,c), residual = sum_r U[a,r]*V[b,r]*W[c,r] - T[a,b,c]
        # We want to minimize sum of squared residuals
        # Gradient descent / Newton-like approach
        
        # Use a simple iterative refinement
        # For fixed U and V, optimal W is determined linearly
        # For fixed U and W, optimal V is determined linearly
        # For fixed V and W, optimal U is determined linearly
        
        # Alternating least squares style
        best_residual = np.inf
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iters):
            if time.time() > deadline - 5.0:
                break
            
            # Update U: for each a, U[a,r] should minimize sum_{b,c} (sum_r' U[a,r']*V[b,r']*W[c,r'] - T[a,b,c])^2
            # This is a linear system in U[a,:]
            # Let M_a be a matrix where M_a[b,c] = sum_r V[b,r]*W[c,r] (but this couples r)
            # Actually, for fixed V and W, the optimal U[a,:] satisfies:
            # sum_r U[a,r] * (sum_{b,c} V[b,r]*W[c,r]*T[a,b,c] + ...) 
            # This is getting complex. Let me use a different approach.
            
            # Simpler: use gradient of the objective F = 0.5 * ||sum_r U_r ⊗ V_r ⊗ W_r - T||^2
            # dF/dU[a,r] = sum_{b,c} (sum_r' U[a,r']*V[b,r']*W[c,r'] - T[a,b,c]) * V[b,r] * W[c,r]
            # Similarly for V and W
            
            # Compute the current tensor estimate
            # This is expensive: sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r]
            # Use einsum
            est = np.einsum('ar,br,cr->abc', U, V, W)
            resid = est - T
            
            # Compute gradients
            grad_U = np.einsum('abc,br,cr->ar', resid, V, W)
            grad_V = np.einsum('abc,ar,cr->br', resid, U, W)
            grad_W = np.einsum('abc,ar,br->cr', resid, U, V)
            
            # Check convergence
            res_norm = np.linalg.norm(resid)
            if res_norm < best_residual:
                best_residual = res_norm
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if res_norm < tol:
                return best_U, best_V, best_W, best_residual
            
            # Gradient step with adaptive learning rate
            # Use a simple backtracking line search
            lr = 0.01
            for _ in range(10):
                new_U = U - lr * grad_U
                new_V = V - lr * grad_V
                new_W = W - lr * grad_W
                new_est = np.einsum('ar,br,cr->abc', new_U, new_V, new_W)
                new_resid = new_est - T
                new_norm = np.linalg.norm(new_resid)
                if new_norm < res_norm:
                    break
                lr *= 0.5
            
            U = new_U
            V = new_V
            W = new_W
        
        return best_U, best_V, best_W, best_residual
    
    # Try rank 20
    found = False
    for attempt in range(5):
        if time.time() > deadline - 20.0:
            break
        U20, V20, W20, res = try_rank(20, max_iters=3000, tol=1e-12)
        if res < 1e-10:
            # Verify exactness
            est = np.einsum('ar,br,cr->abc', U20, V20, W20)
            if np.allclose(est, T, atol=1e-10):
                found = True
                break
    
    if found and time.time() < deadline - 5.0:
        # Round to nearest half-integer if close
        U20 = np.round(U20 * 2) / 2
        V20 = np.round(V20 * 2) / 2
        W20 = np.round(W20 * 2) / 2
        
        # Verify after rounding
        est = np.einsum('ar,br,cr->abc', U20, V20, W20)
        if np.allclose(est, T, atol=1e-10):
            return {"U": U20.tolist(), "V": V20.tolist(), "W": W20.tolist()}
    
    # Fall back to rank 21 Strassen
    rank = 21
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    strassen_steps = [
        {'u': [1, 0, 0, 1], 'v': [1, 0, 0, 1], 'w': [1, 0, 0, 1]},
        {'u': [0, 0, 1, 1], 'v': [1, 0, 0, 0], 'w': [0, 0, 1, -1]},
        {'u': [1, 0, 0, 0], 'v': [0, 1, 0, -1], 'w': [0, 1, 0, 1]},
        {'u': [0, 0, 0, 1], 'v': [-1, 0, 1, 0], 'w': [1, 0, 1, 0]},
        {'u': [1, 1, 0, 0], 'v': [0, 0, 0, 1], 'w': [-1, 1, 0, 0]},
        {'u': [-1, 0, 1, 0], 'v': [1, 1, 0, 0], 'w': [0, 0, 0, 1]},
        {'u': [0, 1, 0, -1], 'v': [0, 0, 1, 1], 'w': [1, 0, 0, 0]},
    ]
    
    v_indices = []
    w_indices = []
    for g in range(3):
        k0 = 2 * g
        k1 = 2 * g + 1
        v_indices.append([0*6+k0, 0*6+k1, 1*6+k0, 1*6+k1])
        w_indices.append([k0*2+0, k1*2+0, k0*2+1, k1*2+1])
    
    for g in range(3):
        for s in range(7):
            r = 7 * g + s
            step = strassen_steps[s]
            for local_idx in range(4):
                if step['u'][local_idx] != 0:
                    U[local_idx, r] = step['u'][local_idx]
                if step['v'][local_idx] != 0:
                    V[v_indices[g][local_idx], r] = step['v'][local_idx]
                if step['w'][local_idx] != 0:
                    W[w_indices[g][local_idx], r] = step['w'][local_idx]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
