import numpy as np
import time
import random

DIMENSIONS = [3, 3, 6]

def _verify(U, V, W, n=3, m=3, p=6):
    """Verify that U, V, W form a valid decomposition."""
    rank = U.shape[1]
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    T = np.zeros((p * n, n * m, m * p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[k * n + i, i * m + j, j * p + k] = 1.0
    
    T_decomp = np.zeros_like(T)
    for r in range(rank):
        T_decomp += np.einsum('a,b,c->abc', W[:, r], U[:, r], V[:, r])
    
    return np.allclose(T_decomp, T, atol=1e-6)

def _try_rank(rank, n=3, m=3, p=6, max_iters=5000, seed=42):
    """Try to find a decomposition of given rank using random search with local refinement."""
    rng = np.random.RandomState(seed)
    
    U = rng.randn(n * m, rank)
    V = rng.randn(m * p, rank)
    W = rng.randn(p * n, rank)
    
    T = np.zeros((p * n, n * m, m * p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[k * n + i, i * m + j, j * p + k] = 1.0
    
    best_err = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iters):
        T_decomp = np.einsum('ar,br,cr->abc', W, U, V)
        err = np.linalg.norm(T_decomp - T)
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if err < 1e-6:
            return best_U, best_V, best_W
        
        scale = max(0.1, 0.5 * best_err)
        U += scale * rng.randn(n * m, rank) * 0.1
        V += scale * rng.randn(m * p, rank) * 0.1
        W += scale * rng.randn(p * n, rank) * 0.1
    
    if best_err < 1e-6:
        return best_U, best_V, best_W
    return None

def _schoolbook(n=3, m=3, p=6):
    """Return the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _structured_search(n=3, m=3, p=6, rank_target=36, max_iters=3000, seed=42):
    """
    Structured decomposition exploiting that p=6=2*3.
    We look for a decomposition where the W factor has a block structure
    that allows us to reduce the effective rank.
    
    Key insight: C[i,k] = sum_j A[i,j]*B[j,k]
    If we can find rank-r tensors that sum to give the multiplication,
    we can potentially do better than n*m*p.
    
    This function tries a parameterized family of decompositions.
    """
    rng = np.random.RandomState(seed)
    
    # Start with schoolbook and try to compress
    U0, V0, W0 = _schoolbook(n, m, p)
    
    # Try to find a lower-rank approximation by optimizing
    # We'll use a gradient-free approach with structured initialization
    
    T = np.zeros((p * n, n * m, m * p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[k * n + i, i * m + j, j * p + k] = 1.0
    
    best_err = float('inf')
    best_U, best_V, best_W = None, None, None
    
    # Try multiple random initializations
    for attempt in range(20):
        U = rng.randn(n * m, rank_target) * 0.5
        V = rng.randn(m * p, rank_target) * 0.5
        W = rng.randn(p * n, rank_target) * 0.5
        
        # Local optimization using coordinate descent
        for it in range(max_iters // 20):
            # Compute error
            T_decomp = np.einsum('ar,br,cr->abc', W, U, V)
            err = np.linalg.norm(T_decomp - T)
            
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if err < 1e-6:
                break
            
            # Update each factor using least squares
            # For fixed V, W: minimize ||sum_r U[:,r] outer V[:,r] outer W[:,r] - T||
            # This is equivalent to finding U that minimizes a quadratic
            
            # Residual
            R = T - T_decomp
            
            # Update U: for each r, U[:,r] should approximate
            # sum over a,c: R[a,b,c] * V[c,r] * W[a,r]
            # This is a linear system
            for r in range(rank_target):
                # Compute the "target" for U[:,r]
                # We want: sum_r U[b,r]*V[c,r]*W[a,r] ≈ T[a,b,c]
                # For fixed r, the contribution is U[b,r] * (V[c,r]*W[a,r])
                # So U[:,r] should be the least-squares solution to:
                # M_r * U[:,r] = target_r
                # where M_r[a,b] = V[c,r]*W[a,r] summed over c... 
                # This is complex. Let's use a simpler approach.
                
                pass
            
            # Simpler: random perturbations with annealing
            scale = max(0.05, 0.3 * err)
            U += scale * rng.randn(n * m, rank_target) * 0.05
            V += scale * rng.randn(m * p, rank_target) * 0.05
            W += scale * rng.randn(p * n, rank_target) * 0.05
        
        if best_err < 1e-6:
            break
    
    if best_err < 1e-6:
        return best_U, best_V, best_W
    return None

def _gradient_descent(rank, n=3, m=3, p=6, max_iters=10000, seed=42, lr=0.01):
    """
    Gradient descent to find a rank-r decomposition.
    """
    rng = np.random.RandomState(seed)
    
    U = rng.randn(n * m, rank) * 0.1
    V = rng.randn(m * p, rank) * 0.1
    W = rng.randn(p * n, rank) * 0.1
    
    T = np.zeros((p * n, n * m, m * p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[k * n + i, i * m + j, j * p + k] = 1.0
    
    best_err = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iters):
        # Compute T_decomp and gradients
        # T_decomp[a,b,c] = sum_r U[b,r]*V[c,r]*W[a,r]
        
        # Gradient w.r.t. U[b,r] = sum_{a,c} (T_decomp[a,b,c] - T[a,b,c]) * V[c,r] * W[a,r]
        # Gradient w.r.t. V[c,r] = sum_{a,b} (T_decomp[a,b,c] - T[a,b,c]) * U[b,r] * W[a,r]
        # Gradient w.r.t. W[a,r] = sum_{b,c} (T_decomp[a,b,c] - T[a,b,c]) * U[b,r] * V[c,r]
        
        T_decomp = np.einsum('ar,br,cr->abc', W, U, V)
        R = T_decomp - T  # Residual
        
        # Compute gradients
        grad_U = np.einsum('abc,cr,ar->br', R, V, W)
        grad_V = np.einsum('abc,br,ar->cr', R, U, W)
        grad_W = np.einsum('abc,br,cr->ar', R, U, V)
        
        err = np.linalg.norm(R)
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if err < 1e-6:
            return best_U, best_V, best_W
        
        # Adaptive learning rate
        grad_norm = np.sqrt(np.sum(grad_U**2) + np.sum(grad_V**2) + np.sum(grad_W**2))
        if grad_norm > 0:
            step_lr = lr * err / grad_norm
        else:
            step_lr = lr
        
        U -= step_lr * grad_U
        V -= step_lr * grad_V
        W -= step_lr * grad_W
        
        # Occasionally add noise to escape local minima
        if it % 500 == 0 and it > 0:
            noise_scale = max(0.01, 0.1 * err)
            U += noise_scale * rng.randn(n * m, rank)
            V += noise_scale * rng.randn(m * p, rank)
            W += noise_scale * rng.randn(p * n, rank)
    
    if best_err < 1e-6:
        return best_U, best_V, best_W
    return None

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Start with schoolbook as fallback
    U_fb, V_fb, W_fb = _schoolbook(n, m, p)
    best_result = (U_fb, V_fb, W_fb, schoolbook_rank)
    
    # Try gradient descent at various ranks
    # Lower bound is at least max(n*m, m*p, p*n) = 18
    # But realistically, for 3x3 x 3x6, we expect rank around 30-40
    
    ranks_to_try = list(range(24, schoolbook_rank))
    
    for rank in ranks_to_try:
        if time.time() > deadline:
            break
        
        found = None
        
        # Try gradient descent with multiple seeds
        for seed in [42, 123, 456, 789, 1000, 2000, 3000, 4000]:
            if time.time() > deadline:
                break
            
            result = _gradient_descent(rank, n, m, p, max_iters=5000, seed=seed, lr=0.005)
            if result is not None:
                U, V, W = result
                if _verify(U, V, W, n, m, p):
                    found = (U, V, W, rank)
                    break
        
        if found is not None:
            best_result = found
            continue
        else:
            # If we can't find this rank, try one lower before giving up
            # but don't go below 24
            if rank <= 25:
                break
    
    U, V, W, rank = best_result
    
    # Convert to lists
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
