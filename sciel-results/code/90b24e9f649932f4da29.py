"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly reproduces matrix multiplication."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(rank))
                if abs(val - 1.0) > tol:
                    return False
    return True

def _build_from_terms(all_terms, n, m, p):
    """Build U, V, W matrices from a list of (u_vec, v_vec, w_vec) terms."""
    rank = len(all_terms)
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    for r, (u_vec, v_vec, w_vec) in enumerate(all_terms):
        U[:, r] = u_vec
        V[:, r] = v_vec
        W[:, r] = w_vec
    return U, V, W

def _strassen_2x2_terms():
    """Return the 7 Strassen terms for 2x2x2 matrix multiplication."""
    terms = []
    terms.append(([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]))
    terms.append(([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, 1]))
    terms.append(([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]))
    terms.append(([0, 0, 0, 1], [0, 0, 1, -1], [1, 1, 0, 0]))
    terms.append(([1, 1, 0, 0], [0, 0, 0, 1], [1, 1, 0, 0]))
    terms.append(([1, 0, -1, 0], [1, 1, 0, 0], [0, 1, 0, 0]))
    terms.append(([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 1]))
    return terms

def _make_schoolbook_terms(n, m, p, k_start=0, k_end=None):
    """Make schoolbook terms for columns k_start to k_end-1."""
    if k_end is None:
        k_end = p
    terms = []
    for i in range(n):
        for j in range(m):
            for k in range(k_start, k_end):
                u_vec = [0.0] * (n * m)
                u_vec[i * m + j] = 1.0
                v_vec = [0.0] * (m * p)
                v_vec[j * p + k] = 1.0
                w_vec = [0.0] * (p * n)
                w_vec[k * n + i] = 1.0
                terms.append((u_vec, v_vec, w_vec))
    return terms

def _extend_2x2_term_to_2x5(u_vec, v_vec_2x2, w_vec_2x2, n=2, m=2, p=5):
    """Extend a 2x2x2 term to a 2x2x5 term (only uses k=0,1)."""
    u_full = list(u_vec)
    v_full = [0.0] * (m * p)
    for j in range(m):
        for k in range(2):
            v_full[j * p + k] = v_vec_2x2[j * 2 + k]
    w_full = [0.0] * (p * n)
    for k in range(2):
        for i in range(n):
            w_full[k * n + i] = w_vec_2x2[k * 2 + i]
    return u_full, v_full, w_full

def _tensor_error(U, V, W, n, m, p):
    """Compute the squared error of the tensor decomposition."""
    error = 0.0
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(rank))
                error += (val - 1.0) ** 2
    return error

def _flatten_params(U, V, W):
    """Flatten U, V, W into a single vector."""
    return np.concatenate([U.flatten(), V.flatten(), W.flatten()])

def _unflatten_params(params, n, m, p, rank):
    """Unflatten a parameter vector into U, V, W."""
    U_size = n * m * rank
    V_size = m * p * rank
    W_size = p * n * rank
    
    U = params[:U_size].reshape(n * m, rank)
    V = params[U_size:U_size+V_size].reshape(m * p, rank)
    W = params[U_size+V_size:U_size+V_size+W_size].reshape(p * n, rank)
    return U, V, W

def _optimize_rank_from_init(rank, n, m, p, deadline, init_U, init_V, init_W, seed=0):
    """Try to find a valid decomposition of the given rank starting from an initialization."""
    rng = np.random.RandomState(seed)
    
    def objective(params):
        U, V, W = _unflatten_params(params, n, m, p, rank)
        return _tensor_error(U, V, W, n, m, p)
    
    # Use the provided initialization
    x0 = _flatten_params(init_U, init_V, init_W)
    
    # Use L-BFGS-B for efficiency with tighter tolerances
    result = minimize(objective, x0, method='L-BFGS-B', 
                     options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-14})
    
    if result.fun < 1e-10:
        U_opt, V_opt, W_opt = _unflatten_params(result.x, n, m, p, rank)
        if _verify_decomposition(U_opt, V_opt, W_opt, n, m, p, tol=1e-8):
            return U_opt, V_opt, W_opt
    
    # Try a second optimization pass from the best point found
    if result.fun < 1e-6:
        result2 = minimize(objective, result.x, method='L-BFGS-B', 
                         options={'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-14})
        if result2.fun < 1e-10:
            U_opt, V_opt, W_opt = _unflatten_params(result2.x, n, m, p, rank)
            if _verify_decomposition(U_opt, V_opt, W_opt, n, m, p, tol=1e-8):
                return U_opt, V_opt, W_opt
    
    return None

def _try_reduce_rank(U, V, W, n, m, p, deadline, target_rank, seed=0):
    """Try to reduce the rank by finding a valid decomposition at target_rank."""
    current_rank = U.shape[1]
    if target_rank >= current_rank:
        return None
    
    # Strategy: Start from the current solution and try to merge terms
    # Create an initial guess by projecting the current solution
    rng = np.random.RandomState(seed)
    
    # Method 1: Random perturbation and optimization
    for attempt in range(5):
        if time.time() > deadline - 20:
            break
        
        # Create a noisy version of the current solution
        U_init = U[:, :target_rank].copy()
        V_init = V[:, :target_rank].copy()
        W_init = W[:, :target_rank].copy()
        
        # Add small random perturbations
        noise_scale = 0.01 * (1 + attempt * 0.5)
        U_init = U_init + rng.randn(n * m, target_rank) * noise_scale
        V_init = V_init + rng.randn(m * p, target_rank) * noise_scale
        W_init = W_init + rng.randn(p * n, target_rank) * noise_scale
        
        result = _optimize_rank_from_init(target_rank, n, m, p, deadline, 
                                           U_init, V_init, W_init, seed=seed + attempt)
        if result is not None:
            return result
    
    # Method 2: Try with different column selections
    for attempt in range(4):
        if time.time() > deadline - 15:
            break
        
        # Select a random subset of columns
        cols = rng.choice(current_rank, size=target_rank, replace=False)
        U_init = U[:, cols].copy()
        V_init = V[:, cols].copy()
        W_init = W[:, cols].copy()
        
        result = _optimize_rank_from_init(target_rank, n, m, p, deadline,
                                           U_init, V_init, W_init, seed=seed + 100 + attempt)
        if result is not None:
            return result
    
    # Method 3: Try to use structure - keep Strassen terms and optimize the rest
    if target_rank >= 7:
        if time.time() > deadline - 10:
            return None
        
        # Keep first 7 terms (Strassen) and optimize the rest
        num_strassen = 7
        num_extra = target_rank - num_strassen
        
        U_init = np.zeros((n * m, target_rank))
        V_init = np.zeros((m * p, target_rank))
        W_init = np.zeros((p * n, target_rank))
        
        # Copy Strassen terms
        U_init[:, :num_strassen] = U[:, :num_strassen]
        V_init[:, :num_strassen] = V[:, :num_strassen]
        W_init[:, :num_strassen] = W[:, :num_strassen]
        
        # Initialize extra terms with random values
        U_init[:, num_strassen:] = rng.randn(n * m, num_extra) * 0.1
        V_init[:, num_strassen:] = rng.randn(m * p, num_extra) * 0.1
        W_init[:, num_strassen:] = rng.randn(p * n, num_extra) * 0.1
        
        result = _optimize_rank_from_init(target_rank, n, m, p, deadline,
                                           U_init, V_init, W_init, seed=seed + 200)
        if result is not None:
            return result
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 130
    
    n, m, p = DIMENSIONS
    
    # Build initial rank-13 decomposition
    strassen_terms = _strassen_2x2_terms()
    all_terms = []
    
    for u_vec, v_vec, w_vec in strassen_terms:
        u_full, v_full, w_full = _extend_2x2_term_to_2x5(u_vec, v_vec, w_vec, n, m, p)
        all_terms.append((u_full, v_full, w_full))
    
    schoolbook_terms = _make_schoolbook_terms(n, m, p, k_start=2, k_end=5)
    all_terms.extend(schoolbook_terms)
    
    U, V, W = _build_from_terms(all_terms, n, m, p)
    
    if not _verify_decomposition(U, V, W, n, m, p):
        rank = n * m * p
        U_fb = np.zeros((n * m, rank))
        V_fb = np.zeros((m * p, rank))
        W_fb = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m+j, col] = 1
                    V_fb[j*p+k, col] = 1
                    W_fb[k*n+i, col] = 1
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = U.shape[1]
    
    # Try to find lower-rank decompositions by progressively reducing
    for target_rank in range(best_rank - 1, 9, -1):
        if time.time() > deadline - 25:
            break
        
        result = _try_reduce_rank(best_U, best_V, best_W, n, m, p, deadline, 
                                   target_rank, seed=target_rank)
        
        if result is not None:
            U_opt, V_opt, W_opt = result
            best_U, best_V, best_W = U_opt.copy(), V_opt.copy(), W_opt.copy()
            best_rank = target_rank
            # Continue trying to reduce further
            continue
    
    # Final polish: try to improve numerical precision of the best solution
    if time.time() < deadline - 5:
        def polish_objective(params):
            U, V, W = _unflatten_params(params, n, m, p, best_rank)
            return _tensor_error(U, V, W, n, m, p)
        
        x0 = _flatten_params(best_U, best_V, best_W)
        polish_result = minimize(polish_objective, x0, method='L-BFGS-B',
                                options={'maxiter': 200, 'ftol': 1e-16, 'gtol': 1e-15})
        
        if polish_result.fun < 1e-12:
            U_polish, V_polish, W_polish = _unflatten_params(polish_result.x, n, m, p, best_rank)
            if _verify_decomposition(U_polish, V_polish, W_polish, n, m, p, tol=1e-10):
                best_U, best_V, best_W = U_polish, V_polish, W_polish
    
    # Clean up: convert near-integers to exact integers
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    for mat in [U_list, V_list, W_list]:
        for row in mat:
            for idx, val in enumerate(row):
                if abs(val - round(val)) < 1e-8:
                    row[idx] = round(val)
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
