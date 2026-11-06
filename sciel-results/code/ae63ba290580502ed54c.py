"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Cache for optimal angles per graph
_angle_cache = {}
_T0 = time.time()
_DEADLINE = 150.0
_SAFETY_MARGIN = 15.0

def _get_remaining_time():
    return _DEADLINE - _SAFETY_MARGIN - (time.time() - _T0)

def _build_c_diagonal(n, edges):
    """
    Compute the diagonal values of the MaxCut Hamiltonian C.
    C(b) = number of edges crossing the cut defined by b.
    """
    dim = 2**n
    diag = np.zeros(dim)
    for mask in range(dim):
        cut_val = 0
        for u, v in edges:
            if ((mask >> u) & 1) != ((mask >> v) & 1):
                cut_val += 1
        diag[mask] = cut_val
    return diag

def _apply_mixer(state, beta, n):
    """
    Apply exp(-i * beta * sum_j X_j) to state vector.
    Uses the fact that M = sum_j X_j and applies single-qubit rotations.
    For small n, use efficient tensor operations.
    """
    dim = 2**n
    if dim <= 256:
        # For small dimensions, we can use the fact that exp(-i*beta*X_j)
        # is a 2x2 rotation on each qubit
        # But since M = sum X_j, we need to be careful
        # Actually, exp(-i*beta*sum_j X_j) is NOT a product of single-qubit gates
        # because the X_j don't commute with each other in the exponent
        # We need to use the full matrix exponential or a different approach
        
        # For n <= 8 (dim <= 256), we can build the matrix
        M = np.zeros((dim, dim), dtype=complex)
        for state_idx in range(dim):
            for j in range(n):
                flipped = state_idx ^ (1 << j)
                M[state_idx, flipped] = 1.0
        
        # Use Taylor series or scipy if available, but we want to avoid scipy
        # Use the fact that M^2 = n * I + 2 * (sum of pairs that differ in 2 bits)
        # Actually, let's just use a simple approach: build M and use expm-like
        # For small dim, we can use numpy's matrix_power or series
        
        # Better: use the spectral decomposition or just compute directly
        # For dim <= 256, we can afford to compute the matrix exponential
        # using the scaling and squaring method
        
        # Simple approach: use the fact that for small beta, we can use series
        # But for general beta, let's build the matrix and use a simple expm
        
        # Build M
        # Apply expm(-i * beta * M) @ state
        # For small dim, we can use numpy.linalg.eigh or similar
        
        # Actually, let's use a different approach: 
        # exp(-i*beta*M) where M = sum X_j
        # The eigenvalues of M are n - 2k where k is the number of 1s in the state
        # but that's not quite right because M doesn't commute with Z
        
        # Let's just use a simple matrix exponential for small dim
        # Build M
        M = np.zeros((dim, dim), dtype=complex)
        for s in range(dim):
            for j in range(n):
                M[s, s ^ (1 << j)] = 1.0
        
        # Compute exp(-i * beta * M) using scaling and squaring
        # Or just use the fact that for small dim, we can iterate
        # Let's use a simple Taylor series with enough terms
        # Actually, for correctness, let's use the matrix exponential
        
        # For dim <= 256, we can use numpy's built-in or implement expm
        # Let's implement a simple expm using scaling and squaring
        
        A = -1j * beta * M
        # Scale
        s = 1
        while s < 2**int(np.log2(np.max(np.abs(A)))) + 1:
            A = A / 2
            s *= 2
        
        # Taylor series
        result = np.eye(dim, dtype=complex)
        term = np.eye(dim, dtype=complex)
        for k in range(1, 20):
            term = term @ A / k
            result = result + term
            if np.max(np.abs(term)) < 1e-12:
                break
        
        # Unscale
        for _ in range(int(np.log2(s))):
            result = result @ result
        
        return result @ state
    
    # For larger dimensions, use the tensor approach
    # Note: exp(-i*beta*sum_j X_j) is NOT a product of single-qubit gates
    # So we need a different approach
    
    # Actually, for the QAOA mixer, the standard mixer is exp(-i*beta*sum_j X_j)
    # This is NOT a product of single-qubit gates because the X_j operators
    # don't factorize in the exponent
    
    # For n > 8, we need a more efficient method
    # One approach: use the fact that the mixer can be applied using
    # a sequence of single-qubit gates if we use a different mixer
    # But the task specifies exp(-i * beta * sum X_j)
    
    # For now, let's use a fallback that's slower but correct
    # Build the matrix for n <= 10 (dim <= 1024)
    if dim <= 1024:
        M = np.zeros((dim, dim), dtype=complex)
        for s in range(dim):
            for j in range(n):
                M[s, s ^ (1 << j)] = 1.0
        
        A = -1j * beta * M
        s = 1
        while s < 2**int(np.log2(np.max(np.abs(A)))) + 1:
            A = A / 2
            s *= 2
        
        result = np.eye(dim, dtype=complex)
        term = np.eye(dim, dtype=complex)
        for k in range(1, 20):
            term = term @ A / k
            result = result + term
            if np.max(np.abs(term)) < 1e-12:
                break
        
        for _ in range(int(np.log2(s))):
            result = result @ result
        
        return result @ state
    
    # For very large n, we'd need a different approach
    # But the task only uses n=6,8,10, so dim <= 1024
    raise ValueError("n too large for current implementation")

def _expected_cut(n, edges, gammas, betas, c_diag):
    """
    Compute expected cut value for given angles.
    """
    p = len(gammas)
    dim = 2**n
    
    # Initial state |+>^n
    state = np.ones(dim, dtype=complex) / np.sqrt(dim)
    
    for k in range(p):
        # Apply Cost: multiply by exp(-i * gamma_k * c_diag)
        phase = np.exp(-1j * gammas[k] * c_diag)
        state = state * phase
        
        # Apply Mixer
        state = _apply_mixer(state, betas[k], n)
    
    # Compute expectation <psi| C |psi>
    prob = np.abs(state)**2
    return float(np.dot(prob, c_diag))

def _nelder_mead_time_bounded(f, x0, bounds, max_iter=100, time_limit=10.0):
    """
    Custom Nelder-Mead optimizer with time limit.
    """
    n = len(x0)
    alpha = 1.0  # Reflection
    gamma = 2.0  # Expansion
    rho = 0.5   # Contraction
    sigma = 0.5 # Shrink
    
    # Initialize simplex
    simplex = [x0.copy()]
    f0 = f(x0)
    simplex.append(np.array(x0, dtype=float))
    for i in range(n):
        xi = x0.copy()
        if abs(xi[i]) > 1e-8:
            xi[i] *= 1.05
        else:
            xi[i] += 0.05
        simplex.append(xi)
    
    # Clip to bounds
    def clip(x):
        return np.clip(x, [b[0] for b in bounds], [b[1] for b in bounds])
    
    simplex = [clip(x) for x in simplex]
    fvals = [f(x) for x in simplex]
    
    start_time = time.time()
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        # Sort by function value
        order = np.argsort(fvals)
        simplex = [simplex[i] for i in order]
        fvals = [fvals[i] for i in order]
        
        # Check convergence
        if abs(fvals[-1] - fvals[0]) < 1e-8:
            break
        
        # Centroid of all but worst point
        centroid = np.mean(simplex[:-1], axis=0)
        centroid = clip(centroid)
        
        # Reflection
        xr = centroid + alpha * (centroid - simplex[-1])
        xr = clip(xr)
        fr = f(xr)
        
        if fr < fvals[0]:
            # Expansion
            xe = centroid + gamma * (centroid - simplex[-1])
            xe = clip(xe)
            fe = f(xe)
            if fe < fr:
                simplex[-1] = xe
                fvals[-1] = fe
            else:
                simplex[-1] = xr
                fvals[-1] = fr
        elif fr < fvals[-2]:
            simplex[-1] = xr
            fvals[-1] = fr
        else:
            # Contraction
            if fr < fvals[-1]:
                # Outside contraction
                xc = centroid + rho * (xr - centroid)
            else:
                # Inside contraction
                xc = centroid - rho * (centroid - simplex[-1])
            xc = clip(xc)
            fc = f(xc)
            
            if fc < fvals[-1]:
                simplex[-1] = xc
                fvals[-1] = fc
            else:
                # Shrink
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    simplex[i] = clip(simplex[i])
                    fvals[i] = f(simplex[i])
    
    # Return best point
    best_idx = np.argmin(fvals)
    return simplex[best_idx], fvals[best_idx]

def _optimize_angles(n, edges, p):
    """
    Optimize angles for given graph.
    """
    c_diag = _build_c_diagonal(n, edges)
    
    # Objective function to minimize (negative expected cut)
    def obj(params):
        gammas = params[:p]
        betas = params[p:]
        val = _expected_cut(n, edges, gammas, betas, c_diag)
        return -val
    
    # Bounds
    bounds = [(0, np.pi)] * (2 * p)
    
    # Generate diverse initial guesses
    starts = []
    
    # Heuristic-based starts
    heuristic_starts = [
        [0.35, 0.25],
        [0.5, 0.2],
        [1.0, 0.5],
        [np.pi/4, np.pi/8],
        [0.2, 0.4],
        [0.8, 0.3],
        [0.6, 0.15],
        [np.pi/6, np.pi/12],
        [0.4, 0.3],
        [0.7, 0.25],
        [0.9, 0.1],
        [0.15, 0.6],
        [1.2, 0.4],
        [0.25, 0.35],
        [0.55, 0.15],
    ]
    
    for gs, bs in heuristic_starts:
        starts.append(np.array([gs] * p + [bs] * p))
    
    # Random starts with fixed seed for reproducibility
    rng = np.random.RandomState(123)
    for _ in range(8):
        g_rand = rng.uniform(0.05, 1.5, p)
        b_rand = rng.uniform(0.05, 0.8, p)
        starts.append(np.concatenate([g_rand, b_rand]))
    
    # Also try scaled versions
    for scale in [0.5, 1.5, 2.0]:
        starts.append(np.array([0.35 * scale] * p + [0.25 * scale] * p))
    
    best_val = np.inf
    best_params = None
    best_angles = np.tile([0.35, 0.25], (p, 1))
    
    # Coarse grid search first to find good regions
    grid_points = []
    g_vals = np.linspace(0.1, 1.5, 5)
    b_vals = np.linspace(0.05, 0.6, 4)
    for g in g_vals:
        for b in b_vals:
            grid_points.append(np.array([g] * p + [b] * p))
    
    # Evaluate grid points
    for gp in grid_points:
        if _get_remaining_time() < 3.0:
            break
        val = obj(gp)
        if val < best_val:
            best_val = val
            best_params = gp.copy()
            gammas = best_params[:p]
            betas = best_params[p:]
            best_angles = np.column_stack((gammas, betas))
    
    # Local optimization from best grid point and diverse starts
    all_starts = [best_params] if best_params is not None else []
    all_starts.extend(starts)
    
    for idx, start in enumerate(all_starts):
        remaining = _get_remaining_time()
        if remaining < 2.0:
            break
        
        # Estimate time per optimization
        num_remaining = len(all_starts) - idx
        time_per_opt = min(remaining / (num_remaining + 1), 8.0)
        
        try:
            x_opt, f_opt = _nelder_mead_time_bounded(
                obj, start, bounds, max_iter=80, time_limit=time_per_opt
            )
            if f_opt < best_val:
                best_val = f_opt
                best_params = x_opt.copy()
                gammas = best_params[:p]
                betas = best_params[p:]
                best_angles = np.column_stack((gammas, betas))
        except Exception:
            continue
    
    if best_params is None:
        # Fallback to seed
        return np.tile([0.35, 0.25], (p, 1))
        
    return best_angles

def angles(n, edges, p):
    # Create a cache key
    edges_tuple = tuple(sorted((u, v) if u < v else (v, u) for u, v in edges))
    key = (n, edges_tuple, p)
    
    if key in _angle_cache:
        return _angle_cache[key]
        
    # Check time
    if _get_remaining_time() < 3.0:
        # Not enough time to optimize, return default
        res = np.tile([0.35, 0.25], (p, 1))
        return res
        
    res = _optimize_angles(n, edges, p)
    
    # Store in cache
    _angle_cache[key] = res
    
    return res
# EVOLVE-BLOCK-END
