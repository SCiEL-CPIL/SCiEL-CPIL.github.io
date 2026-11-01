"""Matrix multiplication 2x3x8 - rank-minimal tensor decomposition search."""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 8]

def _verify(U, V, W, tol=1e-9):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    # Compute the tensor from decomposition
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Should equal 1 if (i,j,k) is a valid index triple, 0 otherwise
    # Actually for matrix multiplication: C = A*B where A is n x m, B is m x p
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # Tensor T[i,j,k] = 1 for all valid i,j,k (it's the multiplication tensor)
    # Wait - the multiplication tensor has T[i,j,k] = 1 if the term A[i,j]*B[j,k] appears in C[i,k]
    # For standard matrix multiplication, every (i,j,k) triple contributes, so T[i,j,k]=1 for all i,j,k
    
    # Actually let me reconsider. The matrix multiplication tensor T ∈ R^{n×m × m×p × p×n}
    # T[i*m+j, j*p+k, k*n+i] = 1 (the schoolbook terms)
    # And all other entries are 0.
    
    # So we need: for all (a,b,c) where a=i*m+j, b=j*p+k, c=k*n+i:
    #   sum_r U[a,r]*V[b,r]*W[c,r] = 1
    # And for all other (a,b,c): sum_r U[a,r]*V[b,r]*W[c,r] = 0
    
    # Let's verify by computing the full tensor
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Compute tensor: T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
    # This is an outer product sum
    # T = sum_r np.outer(np.outer(U[:,r], V[:,r]), W[:,r])
    T = np.zeros((n*m, m*p, p*n))
    for r in range(rank):
        T += np.einsum('a,b,c->abc', U_arr[:,r], V_arr[:,r], W_arr[:,r])
    
    # Check the target tensor
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                err = abs(T[a,b,c] - 1.0)
                if err > max_err:
                    max_err = err
    
    # Check that non-target entries are ~0
    # Build a mask of target positions
    target = np.zeros((n*m, m*p, p*n), dtype=bool)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i*m+j, j*p+k, k*n+i] = True
    
    non_target_err = np.max(np.abs(T[~target])) if (~target).any() else 0.0
    
    return max_err <= tol and non_target_err <= tol

def _rank_of_decomposition(U, V, W):
    """Return the rank (number of columns) of the decomposition."""
    return len(U[0])

def _random_decomposition(rank, scale=1.0):
    """Generate a random decomposition of given rank."""
    n, m, p = DIMENSIONS
    U = np.random.randn(n*m, rank) * scale
    V = np.random.randn(m*p, rank) * scale
    W = np.random.randn(p*n, rank) * scale
    return U.tolist(), V.tolist(), W.tolist()

def _compute_residual(U, V, W):
    """Compute the residual tensor (decomposed - target)."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    T = np.zeros((n*m, m*p, p*n))
    for r in range(rank):
        T += np.einsum('a,b,c->abc', U_arr[:,r], V_arr[:,r], W_arr[:,r])
    
    # Target tensor
    target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i*m+j, j*p+k, k*n+i] = 1.0
    
    return T - target

def _residual_norm(residual):
    """Compute the L2 norm of the residual tensor."""
    return np.linalg.norm(residual)

def _local_search(U, V, W, max_iters=500, lr_init=0.01, deadline=None):
    """
    Local search to minimize residual norm.
    Uses a simple gradient-free approach: try small perturbations.
    """
    n, m, p = DIMENSIONS
    rank = len(U[0])
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    residual = _compute_residual(U_arr.tolist(), V_arr.tolist(), W_arr.tolist())
    current_norm = _residual_norm(residual)
    
    best_U, best_V, best_W = U_arr.copy(), V_arr.copy(), W_arr.copy()
    best_norm = current_norm
    
    lr = lr_init
    step_size = 0.1
    
    for it in range(max_iters):
        if deadline and time.time() > deadline:
            break
        
        # Try random perturbations
        improved = False
        for _ in range(10):
            if deadline and time.time() > deadline:
                break
            
            # Perturb one of the matrices
            mat_choice = random.randint(0, 2)
            delta = np.random.randn(*((n*m, rank) if mat_choice==0 else 
                                       (m*p, rank) if mat_choice==1 else 
                                       (p*n, rank))) * step_size
            
            if mat_choice == 0:
                new_U = U_arr + delta
                new_V = V_arr
                new_W = W_arr
            elif mat_choice == 1:
                new_U = U_arr
                new_V = V_arr + delta
                new_W = W_arr
            else:
                new_U = U_arr
                new_V = V_arr
                new_W = W_arr + delta
            
            new_residual = _compute_residual(new_U.tolist(), new_V.tolist(), new_W.tolist())
            new_norm = _residual_norm(new_residual)
            
            if new_norm < current_norm:
                U_arr, V_arr, W_arr = new_U, new_V, new_W
                current_norm = new_norm
                improved = True
                if current_norm < best_norm:
                    best_norm = current_norm
                    best_U, best_V, best_W = U_arr.copy(), V_arr.copy(), W_arr.copy()
                # Decrease step size slightly
                step_size *= 0.95
            else:
                step_size *= 0.9
        
        if not improved:
            step_size *= 0.7
            if step_size < 1e-12:
                break
        
        # Check convergence
        if current_norm < 1e-10:
            break
    
    return best_U.tolist(), best_V.tolist(), best_W.tolist(), best_norm

def _try_rank(rank, num_trials=5, deadline=None):
    """Try to find a valid decomposition of given rank."""
    n, m, p = DIMENSIONS
    best_result = None
    best_norm = float('inf')
    
    for trial in range(num_trials):
        if deadline and time.time() > deadline:
            break
        
        # Random initialization
        U, V, W = _random_decomposition(rank, scale=1.0)
        
        # Local search
        U, V, W, norm = _local_search(U, V, W, max_iters=300, lr_init=0.05, deadline=deadline)
        
        if norm < best_norm:
            best_norm = norm
            best_result = (U, V, W)
        
        if norm < 1e-9:
            # Verify
            if _verify(U, V, W):
                return U, V, W
    
    return best_result, best_norm

def _greedy_add_term(current_U, current_V, current_W, deadline=None):
    """
    Greedily add a term to reduce the residual.
    Given current decomposition, find the best single term to add.
    """
    n, m, p = DIMENSIONS
    rank = len(current_U[0])
    
    U_arr = np.array(current_U, dtype=float)
    V_arr = np.array(current_V, dtype=float)
    W_arr = np.array(current_W, dtype=float)
    
    residual = _compute_residual(U_arr.tolist(), V_arr.tolist(), W_arr.tolist())
    current_norm = _residual_norm(residual)
    
    # Try adding random terms
    best_new_norm = current_norm
    best_term = None
    
    for _ in range(50):
        if deadline and time.time() > deadline:
            break
        
        new_u = np.random.randn(n*m)
        new_v = np.random.randn(m*p)
        new_w = np.random.randn(p*n)
        
        # Scale the term to best reduce residual
        # We want to minimize ||residual - alpha * outer(new_u, new_v, new_w)||^2
        # Optimal alpha = <residual, outer> / ||outer||^2
        outer = np.einsum('a,b,c->abc', new_u, new_v, new_w)
        outer_norm_sq = np.sum(outer**2)
        if outer_norm_sq < 1e-15:
            continue
        alpha = np.sum(residual * outer) / outer_norm_sq
        
        new_residual = residual - alpha * outer
        new_norm = _residual_norm(new_residual)
        
        if new_norm < best_new_norm:
            best_new_norm = new_norm
            best_term = (new_u, new_v, new_w, alpha)
    
    if best_term is not None:
        new_u, new_v, new_w, alpha = best_term
        # Add the term (scaled)
        U_arr = np.column_stack([U_arr, alpha * new_u])
        V_arr = np.column_stack([V_arr, new_v])
        W_arr = np.column_stack([W_arr, new_w])
        # Actually we need to distribute alpha properly
        # T_new = T_old + alpha * outer(new_u, new_v, new_w)
        # So U_new[:, -1] = new_u, V_new[:, -1] = alpha * new_v... no
        # T = sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r]
        # We want T + alpha * (new_u ⊗ new_v ⊗ new_w)
        # So we can set U_new = [U, new_u], V_new = [V, alpha*new_v], W_new = [W, new_w]
        # Or distribute alpha among the three
        # Let's put alpha in one of them
        V_arr = np.column_stack([V_arr, alpha * new_v])
        U_arr = np.column_stack([U_arr, new_u])
        W_arr = np.column_stack([W_arr, new_w])
        
        return U_arr.tolist(), V_arr.tolist(), W_arr.tolist()
    
    return current_U, current_V, current_W

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 48
    
    # Start with schoolbook as fallback
    U_fb = [[0] * schoolbook_rank for _ in range(n * m)]
    V_fb = [[0] * schoolbook_rank for _ in range(m * p)]
    W_fb = [[0] * schoolbook_rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][column] = 1
                V_fb[j * p + k][column] = 1
                W_fb[k * n + i][column] = 1
                column += 1
    
    best_U, best_V, best_W = U_fb, V_fb, W_fb
    best_rank = schoolbook_rank
    
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    # Try to find lower rank decompositions
    # Known lower bounds: rank >= max(n*m, m*p, p*n) = max(6, 24, 16) = 24
    # Also rank >= n*p = 16 (from Strassen-type arguments)
    # For 2x3x8, the rank is at least 24 (since m*p = 24)
    
    # Try ranks from 24 upward
    for rank in range(24, schoolbook_rank):
        if time.time() > deadline:
            break
        
        # Try multiple random initializations
        found = False
        num_trials = 3
        for trial in range(num_trials):
            if time.time() > deadline:
                break
            
            U, V, W = _random_decomposition(rank, scale=1.0)
            
            # Local search
            U, V, W, norm = _local_search(U, V, W, max_iters=200, lr_init=0.05, deadline=deadline)
            
            if norm < 1e-8:
                if _verify(U, V, W):
                    best_U, best_V, best_W = U, V, W
                    best_rank = rank
                    found = True
                    break
        
        if found:
            # Try to go even lower
            continue
    
    # If we found something better, also try to polish
    if best_rank < schoolbook_rank:
        # Try to reduce further
        for rank in range(24, best_rank):
            if time.time() > deadline:
                break
            for trial in range(2):
                if time.time() > deadline:
                    break
                U, V, W = _random_decomposition(rank, scale=1.0)
                U, V, W, norm = _local_search(U, V, W, max_iters=200, lr_init=0.05, deadline=deadline)
                if norm < 1e-8 and _verify(U, V, W):
                    best_U, best_V, best_W = U, V, W
                    best_rank = rank
                    break
    
    return {"U": best_U, "V": best_V, "W": best_W}
