import numpy as np
import time
import random

DIMENSIONS = [2, 3, 4]

def _build_target_tensor(n, m, p):
    """
    Build the target tensor T of shape (n*m, m*p, p*n)
    T[i*m + j, j*p + k, k*n + i] = 1
    """
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0
    return T

def _verify(U, V, W, T, tol=1e-9):
    """
    Verify that sum_r U[:,r] * V[:,r] * W[:,r] == T
    """
    try:
        recon = np.einsum('ur,vr,wr->uvw', U, V, W)
    except Exception:
        return False
    return np.allclose(recon, T, atol=tol, rtol=0)

def _try_rank_exhaustive(R, n, m, p, T, values, time_limit):
    """
    Try to find a rank-R decomposition by enumerating W columns.
    For each candidate W column w (a vector of length p*n), we have a linear system
    in the remaining 2*R unknowns (U and V columns).
    
    Specifically, for a fixed W column w, the tensor equation is:
    sum_{r=1}^{R} u_r * v_r * w_r = T
    
    For r=R, we fix w_R. Then for each (u,v) pair where T[u,v,:] is non-zero,
    we have: sum_{r=1}^{R-1} u_r * v_r * w_r + u_R * v_R * w_R = T[u,v,:]
    
    This is still non-linear in u_R and v_R.
    
    Alternative approach: Use the fact that for each fixed w, the problem reduces to
    finding u and v such that sum_r u_r v_r w_r = T - sum_{r<R} u_r v_r w_r.
    
    This is getting complex. Let's use a different strategy:
    
    Strategy: Enumerate all possible W matrices (each column from values^(p*n)).
    For each W, solve for U and V.
    
    Given W, the tensor equation is:
    sum_r U[:,r] * V[:,r] * W[:,r] = T
    
    This is a system of (n*m)*(m*p)*(p*n) linear equations in the unknowns U and V.
    But U and V appear as products, so it's not linear.
    
    However, if we fix W, we can view this as:
    For each r, the term U[:,r] * V[:,r] * W[:,r] is a rank-1 tensor in (u,v) for fixed w.
    
    Let's try a simpler approach: random search with local refinement, but smarter.
    """
    start_time = time.time()
    
    # Number of possible W columns
    p_n = p * n
    num_w_cols = len(values) ** p_n
    
    # If num_w_cols is too large, we can't enumerate
    if num_w_cols > 100000:
        # Use random sampling of W
        num_samples = min(1000, int(time_limit * 100))
        for _ in range(num_samples):
            if time.time() - start_time > time_limit:
                break
            W = np.random.choice(values, size=(p_n, R))
            # Now solve for U and V
            # This is a non-linear system, so we use iterative refinement
            # Initialize U and V randomly
            U = np.random.choice(values, size=(n * m, R))
            V = np.random.choice(values, size=(m * p, R))
            
            # Local search on U and V only
            for _ in range(1000):
                if time.time() - start_time > time_limit:
                    break
                if _verify(U, V, W, T):
                    return U, V, W
                
                # Pick a random element from U or V
                if random.random() < 0.5:
                    mat = U
                    idx = random.randint(0, U.size - 1)
                else:
                    mat = V
                    idx = random.randint(0, V.size - 1)
                
                old_val = mat.flat[idx]
                new_val = random.choice(values)
                mat.flat[idx] = new_val
                
                if _verify(U, V, W, T):
                    return U, V, W
                else:
                    mat.flat[idx] = old_val
    
    # If we get here, we didn't find a solution
    return None

def _try_rank_with_greedy(R, n, m, p, T, values, time_limit):
    """
    Greedy approach: build the decomposition one column at a time.
    """
    start_time = time.time()
    
    # We want to find U, V, W such that sum_r U[:,r] * V[:,r] * W[:,r] = T
    # Start with residual = T
    residual = T.copy()
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    for r in range(R):
        if time.time() - start_time > time_limit:
            break
        
        # Find the best (u, v, w) that reduces the residual
        # We can do this by trying all combinations of values for u, v, w
        # But that's |values|^3 which is manageable
        
        best_u = None
        best_v = None
        best_w = None
        best_residual_norm = np.linalg.norm(residual)
        
        # Sample some combinations
        num_samples = 1000
        for _ in range(num_samples):
            u = np.random.choice(values, size=n * m)
            v = np.random.choice(values, size=m * p)
            w = np.random.choice(values, size=p * n)
            
            # Compute the tensor u * v * w
            tensor = np.einsum('u,v,w->uvw', u, v, w)
            
            # New residual
            new_residual = residual - tensor
            new_norm = np.linalg.norm(new_residual)
            
            if new_norm < best_residual_norm - 1e-9:
                best_residual_norm = new_norm
                best_u = u
                best_v = v
                best_w = w
        
        if best_u is None:
            break
        
        U_cols.append(best_u)
        V_cols.append(best_v)
        W_cols.append(best_w)
        residual = residual - np.einsum('u,v,w->uvw', best_u, best_v, best_w)
        
        if np.allclose(residual, 0, atol=1e-9):
            break
    
    if np.allclose(residual, 0, atol=1e-9):
        U = np.column_stack(U_cols)
        V = np.column_stack(V_cols)
        W = np.column_stack(W_cols)
        return U, V, W
    
    return None

def _try_rank_structured(R, n, m, p, T, values, time_limit):
    """
    Try a structured decomposition based on known algorithms for small matrix multiplication.
    For 2x3x4, we can try to exploit the structure.
    """
    start_time = time.time()
    
    # The tensor T has entries T[i*m+j, j*p+k, k*n+i] = 1
    # Let's try to find a decomposition with the same structure
    # i.e., U[i*m+j, r] depends on (i,j), V[j*p+k, r] depends on (j,k), W[k*n+i, r] depends on (k,i)
    
    # This is a constrained search
    # We can parameterize U, V, W with fewer parameters
    
    # For now, let's just do random search with local refinement
    best_error = float('inf')
    best_U = None
    best_V = None
    best_W = None
    
    num_restarts = 50
    time_per_restart = time_limit / num_restarts
    
    for _ in range(num_restarts):
        if time.time() - start_time > time_limit:
            break
        
        U = np.random.choice(values, size=(n * m, R))
        V = np.random.choice(values, size=(m * p, R))
        W = np.random.choice(values, size=(p * n, R))
        
        # Local search
        for _ in range(10000):
            if time.time() - start_time > time_limit:
                break
            if _verify(U, V, W, T):
                return U, V, W
            
            # Pick a random matrix and element
            mat = random.choice([U, V, W])
            idx = random.randint(0, mat.size - 1)
            
            old_val = mat.flat[idx]
            new_val = random.choice(values)
            mat.flat[idx] = new_val
            
            if _verify(U, V, W, T):
                return U, V, W
            else:
                mat.flat[idx] = old_val
    
    return None

def construct():
    n, m, p = DIMENSIONS
    T = _build_target_tensor(n, m, p)
    
    # Schoolbook solution as fallback
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n * m, rank_schoolbook))
    V_sb = np.zeros((m * p, rank_schoolbook))
    W_sb = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
                
    best_result = {'U': U_sb.tolist(), 'V': V_sb.tolist(), 'W': W_sb.tolist()}
    best_rank = rank_schoolbook
    
    # Define search space
    values = [-1.0, -0.5, 0.0, 0.5, 1.0]
    
    # Try ranks lower than schoolbook
    start_time = time.time()
    deadline = 140.0  # Safety margin
    
    # Try structured search first for lower ranks
    for R in range(12, rank_schoolbook):
        if time.time() - start_time > deadline:
            break
        
        time_budget = min(20.0, (deadline - (time.time() - start_time)) / max(1, rank_schoolbook - R))
        if time_budget < 1.0:
            break
        
        result = _try_rank_structured(R, n, m, p, T, values, time_budget)
        
        if result is not None:
            U, V, W = result
            if _verify(U, V, W, T):
                best_result = {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
                best_rank = R
                break
    
    # If no better found, return schoolbook
    return best_result
