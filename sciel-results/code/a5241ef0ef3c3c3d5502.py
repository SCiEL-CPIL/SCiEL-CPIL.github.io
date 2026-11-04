<<<<<<< SEARCH
def _try_rank(n, m, p, target_rank, max_iterations=5000, deadline=None):
    """
    Attempt to find a decomposition of exactly target_rank using iterative refinement.
    Uses a randomized initialization and gradient-free local search.
    """
    if deadline is not None:
        if time.time() > deadline:
            return None
    
    # Strategy: start with schoolbook and try to zero out columns via row operations
    # This is a known technique: if we can find linear dependencies among the rank-1 terms,
    # we can reduce the rank.
    
    # Alternative: use the fact that for small matrices, we can set up a system of equations.
    # The tensor T is of size n*m*p = 63.
    # We want to find U (n*m x r), V (m*p x r), W (p*n x r) such that
    # for all i,j,k: sum_{r=1}^{r} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # This is a system of 63 equations in r*(n*m + m*p + p*n) unknowns.
    # For r < n*m*p, this is underdetermined, so solutions exist if the rank of the
    # associated tensor allows it.
    
    # We'll use a simple approach: try to find a solution by fixing some variables and solving.
    # Given the small size, we can use a randomized local search.
    
    # Initialize with random small values
    np.random.seed(42)
    U = np.random.randn(n * m, target_rank) * 0.1
    V = np.random.randn(m * p, target_rank) * 0.1
    W = np.random.randn(p * n, target_rank) * 0.1
    
    # Scale to get closer to the right magnitude
    # The product U*V*W should be ~1, so each factor should be ~1
    U = U / (np.linalg.norm(U) / np.sqrt(n * m))
    V = V / (np.linalg.norm(V) / np.sqrt(m * p))
    W = W / (np.linalg.norm(W) / np.sqrt(p * n))
    
    best_err = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iterations):
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute error
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(target_rank):
                        val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    err += (val - 1.0) ** 2
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if best_err < 1e-9:
            break
        
        # Perturb one factor randomly
        factor = random.choice(['U', 'V', 'W'])
        if factor == 'U':
            row = random.randint(0, n * m - 1)
            col = random.randint(0, target_rank - 1)
            U[row, col] += random.gauss(0, 0.1)
        elif factor == 'V':
            row = random.randint(0, m * p - 1)
            col = random.randint(0, target_rank - 1)
            V[row, col] += random.gauss(0, 0.1)
        else:
            row = random.randint(0, p * n - 1)
            col = random.randint(0, target_rank - 1)
            W[row, col] += random.gauss(0, 0.1)
    
    if best_err < 1e-9:
        return best_U, best_V, best_W
    return None
=======
def _compute_error(U, V, W, n, m, p):
    """Compute the total squared error of the decomposition."""
    rank = U.shape[1]
    err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                err += (val - 1.0) ** 2
    return err


def _try_rank(n, m, p, target_rank, max_iterations=5000, deadline=None):
    """
    Attempt to find a decomposition of exactly target_rank using structured search.
    Uses a more informed initialization and local search.
    """
    if deadline is not None:
        if time.time() > deadline:
            return None
    
    # Strategy: Use structured initialization based on block structure
    # For 3x3 * 3x7, we can think of this as 7 independent 3x3 * 3x1 multiplications
    # or use Strassen-like ideas for the 3x3 part
    
    # Initialize with a structured pattern that respects the tensor structure
    np.random.seed(42)
    
    # Start with near-schoolbook but with some mixing
    U = np.zeros((n * m, target_rank))
    V = np.zeros((m * p, target_rank))
    W = np.zeros((p * n, target_rank))
    
    # Initialize with random orthogonal-like structure
    for r in range(target_rank):
        # Random unit vectors for each factor
        u_vec = np.random.randn(n * m)
        v_vec = np.random.randn(m * p)
        w_vec = np.random.randn(p * n)
        
        u_norm = np.linalg.norm(u_vec)
        v_norm = np.linalg.norm(v_vec)
        w_norm = np.linalg.norm(w_vec)
        
        if u_norm > 0 and v_norm > 0 and w_norm > 0:
            U[:, r] = u_vec / u_norm
            V[:, r] = v_vec / v_norm
            W[:, r] = w_vec / w_norm
    
    best_err = _compute_error(U, V, W, n, m, p)
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Local search with adaptive step size
    step_size = 0.1
    no_improve_count = 0
    
    for it in range(max_iterations):
        if deadline is not None and time.time() > deadline:
            break
        
        if best_err < 1e-9:
            break
        
        # Compute current error
        err = _compute_error(U, V, W, n, m, p)
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
            # Reduce step size if no improvement
            if no_improve_count % 100 == 0:
                step_size *= 0.9
        
        # Perturb one factor with structured moves
        factor = random.choice(['U', 'V', 'W'])
        if factor == 'U':
            row = random.randint(0, n * m - 1)
            col = random.randint(0, target_rank - 1)
            U[row, col] += random.gauss(0, step_size)
        elif factor == 'V':
            row = random.randint(0, m * p - 1)
            col = random.randint(0, target_rank - 1)
            V[row, col] += random.gauss(0, step_size)
        else:
            row = random.randint(0, p * n - 1)
            col = random.randint(0, target_rank - 1)
            W[row, col] += random.gauss(0, step_size)
    
    if best_err < 1e-9:
        return best_U, best_V, best_W
    return None
>>>>>>> REPLACE

