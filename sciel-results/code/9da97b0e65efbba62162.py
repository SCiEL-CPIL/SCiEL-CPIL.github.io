import time
import random
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 5]

def _check_decomposition(U, V, W, n, m, p, rank, tol=1e-8):
    """Check if U, V, W exactly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    T_decomp = np.zeros((n*m, m*p, p*n))
    for r in range(rank):
        T_decomp += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
    
    return np.allclose(T_decomp, T, atol=tol)

def _schoolbook_solution(n, m, p):
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Target Tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Shapes
    size_u = n * m
    size_v = m * p
    size_w = p * n
    total_size = size_u + size_v + size_w
    
    def objective(params, rank):
        # params is a flat array of length rank * total_size
        U = params[:rank*size_u].reshape((rank, size_u))
        V = params[rank*size_u:rank*(size_u+size_v)].reshape((rank, size_v))
        W = params[rank*(size_u+size_v):].reshape((rank, size_w))
        
        T_decomp = np.zeros((n*m, m*p, p*n))
        for r in range(rank):
            T_decomp += np.einsum('a,b,c->abc', U[r,:], V[r,:], W[r,:])
        
        diff = T_decomp - T
        return np.sum(diff * diff)
    
    # Try ranks from lower bound up to schoolbook
    # Lower bound for 3x3x5 is 15.
    # We try a range of ranks.
    # To save time, we try fewer restarts for higher ranks.
    
    best_U, best_V, best_W = _schoolbook_solution(n, m, p)
    best_rank = schoolbook_rank
    
    # Strategy: Try ranks in increasing order.
    # If we find a valid decomposition for rank R, we try R-1, etc.
    # But finding one is hard. We just try to find *any* valid decomposition
    # for a rank lower than 45.
    
    # We will try ranks: 20, 25, 30, 35, 40
    # If we find a solution for 30, we don't need to try 40.
    # But we might miss 25 if we only try 30.
    # Let's try a few specific ranks.
    
    ranks_to_try = [20, 22, 25, 28, 30, 35, 40]
    
    for rank in ranks_to_try:
        if time.time() > deadline:
            break
            
        # Multiple restarts for each rank
        max_restarts = 5 if rank > 30 else 10
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
                
            # Random initialization
            # Scale: values around 1.0
            init_params = np.random.uniform(-1, 1, size=(rank, total_size))
            
            # Use L-BFGS-B
            res = minimize(
                objective, 
                init_params.flatten(), 
                args=(rank,), 
                method='L-BFGS-B', 
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if res.fun < 1e-8:
                # Found a solution!
                params = res.x
                U = params[:rank*size_u].reshape((rank, size_u)).T
                V = params[rank*size_u:rank*(size_u+size_v)].reshape((rank, size_v)).T
                W = params[rank*(size_u+size_v):].reshape((rank, size_w)).T
                
                # Verify
                if _check_decomposition(U, V, W, n, m, p, rank):
                    best_rank = rank
                    best_U = U
                    best_V = V
                    best_W = W
                    # If we found a low rank, we can break or try lower
                    # For now, break to return this good solution
                    break
        
        if best_rank < schoolbook_rank:
            # Found a better solution, try to see if we can go even lower?
            # Due to time constraints, we might just return the first found.
            # But let's try one lower rank if we have time.
            if time.time() < deadline - 20:
                lower_rank = best_rank - 1
                if lower_rank >= 15:
                    # Try one restart for lower_rank
                    init_params = np.random.uniform(-1, 1, size=(lower_rank, total_size))
                    res = minimize(
                        objective, 
                        init_params.flatten(), 
                        args=(lower_rank,), 
                        method='L-BFGS-B', 
                        options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
                    )
                    if res.fun < 1e-8:
                        params = res.x
                        U = params[:lower_rank*size_u].reshape((lower_rank, size_u)).T
                        V = params[lower_rank*size_u:lower_rank*(size_u+size_v)].reshape((lower_rank, size_v)).T
                        W = params[lower_rank*(size_u+size_v):].reshape((lower_rank, size_w)).T
                        if _check_decomposition(U, V, W, n, m, p, lower_rank):
                            best_rank = lower_rank
                            best_U = U
                            best_V = V
                            best_W = W
            break # Stop trying higher ranks if we found a good one
    
    # Final verification
    if not _check_decomposition(best_U, best_V, best_W, n, m, p, best_rank):
        best_U, best_V, best_W = _schoolbook_solution(n, m, p)
        best_rank = schoolbook_rank
    
    return {
        "U": [[float(x) for x in row] for row in best_U],
        "V": [[float(x) for x in row] for row in best_V],
        "W": [[float(x) for x in row] for row in best_W]
    }
