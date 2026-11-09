import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    dim_u = n * m  # 6
    dim_v = m * p  # 15
    dim_w = p * n  # 10
    
    # Build target tensor T
    T = np.zeros((dim_u, dim_v, dim_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
                
    start_time = time.time()
    deadline = 140.0  # 10s margin before 150s kill
    
    # Lower bound is max(dim_u, dim_v, dim_w) = 15
    # Try ranks from 15 upward, but be smart about it
    
    # First, check if rank 15 is possible (very unlikely but worth checking)
    # Then try 16, 17, ... up to some reasonable limit
    
    # Actually, let's be more strategic. The minimal rank for 2x3 x 3x5 is unknown,
    # but let's try a few promising ranks.
    
    # Try ranks in order, but skip some that are very unlikely
    # Focus on ranks that are multiples or near multiples of small numbers
    
    candidate_ranks = []
    for R in range(15, 31):
        candidate_ranks.append(R)
    
    # Try each rank
    for R in candidate_ranks:
        if time.time() > deadline - 5:
            break
            
        found = _find_decomposition(T, dim_u, dim_v, dim_w, R, n, m, p, deadline)
        if found is not None:
            return found
            
    # Fallback to schoolbook
    return _schoolbook(n, m, p)

def _find_decomposition(T, dim_u, dim_v, dim_w, R, n, m, p, deadline):
    """Try to find a rank-R decomposition using structured initialization"""
    
    # Strategy: use the structure of the tensor to initialize
    # The tensor has non-zero entries at specific positions
    # We can try to find a decomposition that exploits this
    
    max_restarts = 5
    
    for restart in range(max_restarts):
        if time.time() > deadline - 3:
            return None
            
        # Initialize using a structured approach
        U, V, W = _structured_init(dim_u, dim_v, dim_w, R, n, m, p, restart)
        
        # Run ALS with half-integer rounding
        success = _als_refine(T, U, V, W, dim_u, dim_v, dim_w, R, deadline)
        if success:
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
            
    return None

def _structured_init(dim_u, dim_v, dim_w, R, n, m, p, restart):
    """Create a structured initialization based on the tensor structure"""
    
    np.random.seed(42 + restart * 1000)
    
    U = np.zeros((dim_u, R))
    V = np.zeros((dim_v, R))
    W = np.zeros((dim_w, R))
    
    # Use the schoolbook structure as a starting point
    # Each (i,j,k) triple gives a rank-1 component
    # Try to combine some of these
    
    # First, create a base decomposition with n*m*p terms
    # Then try to merge some terms
    
    # For now, use random half-integer initialization but with structure
    for r in range(R):
        # Assign random half-integers
        U[:, r] = np.random.choice([-1, -0.5, 0, 0.5, 1], size=dim_u)
        V[:, r] = np.random.choice([-1, -0.5, 0, 0.5, 1], size=dim_v)
        W[:, r] = np.random.choice([-1, -0.5, 0, 0.5, 1], size=dim_w)
        
        # Ensure not all zero
        if np.all(U[:, r] == 0):
            U[np.random.randint(dim_u), r] = 1
        if np.all(V[:, r] == 0):
            V[np.random.randint(dim_v), r] = 1
        if np.all(W[:, r] == 0):
            W[np.random.randint(dim_w), r] = 1
            
    return U, V, W

def _als_refine(T, U, V, W, dim_u, dim_v, dim_w, R, deadline):
    """Run ALS refinement with half-integer rounding"""
    
    max_iters = 50
    
    for it in range(max_iters):
        if time.time() > deadline - 2:
            break
            
        # Check if we've converged
        T_decomp = _compute_tensor(U, V, W)
        if np.allclose(T_decomp, T, atol=1e-9):
            return True
            
        # Update U
        U_new = np.zeros_like(U)
        for a in range(dim_u):
            target = T[a, :, :].flatten()
            A = np.zeros((dim_v * dim_w, R))
            for b in range(dim_v):
                for c in range(dim_w):
                    idx = b * dim_w + c
                    A[idx, :] = V[b, :] * W[c, :]
            try:
                sol, _, _, _ = np.linalg.lstsq(A, target, rcond=None)
                U_new[a, :] = np.round(sol * 2) / 2
            except:
                U_new[a, :] = U[a, :]
        U = U_new
        
        # Update V
        V_new = np.zeros_like(V)
        for b in range(dim_v):
            target = T[:, b, :].flatten()
            A = np.zeros((dim_u * dim_w, R))
            for a in range(dim_u):
                for c in range(dim_w):
                    idx = a * dim_w + c
                    A[idx, :] = U[a, :] * W[c, :]
            try:
                sol, _, _, _ = np.linalg.lstsq(A, target, rcond=None)
                V_new[b, :] = np.round(sol * 2) / 2
            except:
                V_new[b, :] = V[b, :]
        V = V_new
        
        # Update W
        W_new = np.zeros_like(W)
        for c in range(dim_w):
            target = T[:, :, c].flatten()
            A = np.zeros((dim_u * dim_v, R))
            for a in range(dim_u):
                for b in range(dim_v):
                    idx = a * dim_v + b
                    A[idx, :] = U[a, :] * V[b, :]
            try:
                sol, _, _, _ = np.linalg.lstsq(A, target, rcond=None)
                W_new[c, :] = np.round(sol * 2) / 2
            except:
                W_new[c, :] = W[c, :]
        W = W_new
        
    # Final check
    T_decomp = _compute_tensor(U, V, W)
    if np.allclose(T_decomp, T, atol=1e-9):
        return True
        
    return False

def _compute_tensor(U, V, W):
    """Compute the tensor from factor matrices"""
    return np.einsum('ar,br,cr->abc', U, V, W)

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p"""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
