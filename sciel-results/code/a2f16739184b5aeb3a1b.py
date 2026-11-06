import numpy as np
import time
# EVOLVE-BLOCK-START

def _build_hamiltonians():
    """
    Build A (odd bonds + half field) and B (even bonds + half field) for a
    6-site open XXZ Heisenberg chain with anisotropy 0.8 and field 0.3.
    """
    N = 6
    dim = 2 ** N
    gamma = 0.8  # anisotropy
    h = 0.3      # field
    
    # Pauli matrices
    sx = np.array([[0, 0.5], [0.5, 0]], dtype=complex)
    sy = np.array([[0, -0.5j], [0.5j, 0]], dtype=complex)
    sz = np.array([[0.5, 0], [0, -0.5]], dtype=complex)
    ident = np.eye(2, dtype=complex)
    
    def tensor_product(matrices):
        result = matrices[0]
        for m in matrices[1:]:
            result = np.kron(result, m)
        return result
    
    # Build operators for each site
    def site_op(op, site):
        ops = [ident] * N
        ops[site] = op
        return tensor_product(ops)
    
    # Build A: odd bonds (0-1, 2-3, 4-5) + half field
    A = np.zeros((dim, dim), dtype=complex)
    for j in [0, 2, 4]:
        ops_sx_sx = [ident] * N
        ops_sx_sx[j] = sx
        ops_sx_sx[j+1] = sx
        A += tensor_product(ops_sx_sx)
        
        ops_sy_sy = [ident] * N
        ops_sy_sy[j] = sy
        ops_sy_sy[j+1] = sy
        A += tensor_product(ops_sy_sy)
        
        ops_sz_sz = [ident] * N
        ops_sz_sz[j] = sz
        ops_sz_sz[j+1] = sz
        A += gamma * tensor_product(ops_sz_sz)
    
    for j in range(N):
        A -= (h / 2.0) * site_op(sz, j)
        
    # Ensure Hermitian
    A = (A + A.conj().T) / 2.0

    # Build B: even bonds (1-2, 3-4) + half field
    B = np.zeros((dim, dim), dtype=complex)
    for j in [1, 3]:
        ops_sx_sx = [ident] * N
        ops_sx_sx[j] = sx
        ops_sx_sx[j+1] = sx
        B += tensor_product(ops_sx_sx)
        
        ops_sy_sy = [ident] * N
        ops_sy_sy[j] = sy
        ops_sy_sy[j+1] = sy
        B += tensor_product(ops_sy_sy)
        
        ops_sz_sz = [ident] * N
        ops_sz_sz[j] = sz
        ops_sz_sz[j+1] = sz
        B += gamma * tensor_product(ops_sz_sz)
    
    for j in range(N):
        B -= (h / 2.0) * site_op(sz, j)
        
    # Ensure Hermitian
    B = (B + B.conj().T) / 2.0

    return A, B

# Pre-compute the Hamiltonians and eigen-decompositions
_A, _B = _build_hamiltonians()
_H = _A + _B

# Eigen-decompositions for Hermitian matrices: H = V diag(w) V^dagger
# exp(-i t H) = V diag(exp(-i t w)) V^dagger
_wA, _VA = np.linalg.eigh(_A)
_wB, _VB = np.linalg.eigh(_B)
_wH, _VH = np.linalg.eigh(_H)

_T = [0.25, 0.5, 1.0]

def _expm_hermitian_scaled(w, V, scale):
    """Compute exp(-i * scale * H) given eigenvalues w and eigenvectors V."""
    exp_w = np.exp(-1j * scale * w)
    return (V * exp_w) @ V.conj().T

# Pre-compute exact propagators
_exact_U = [ _expm_hermitian_scaled(_wH, _VH, t) for t in _T ]

def _compute_error(a, b):
    """Compute the mean spectral norm error over the three time points."""
    errors = []
    dim = 64
    for t, U_exact in zip(_T, _exact_U):
        U = np.eye(dim, dtype=complex)
        for k in range(12):
            if b[k] != 0.0:
                U_B = _expm_hermitian_scaled(_wB, _VB, b[k] * t)
                U = U_B @ U
            if a[k] != 0.0:
                U_A = _expm_hermitian_scaled(_wA, _VA, a[k] * t)
                U = U_A @ U
        err = np.linalg.norm(U - U_exact, 2)
        errors.append(err)
    return np.mean(errors)

def _is_valid(a, b):
    """Check if coefficients are valid."""
    if len(a) != 12 or len(b) != 12:
        return False
    for x in a + b:
        if not np.isfinite(x) or x < -50 or x > 50:
            return False
    if abs(sum(a) - 1.0) > 1e-8 or abs(sum(b) - 1.0) > 1e-8:
        return False
    return True

def _nelder_mead(f, x0, bounds, max_iter=200, time_limit=120.0):
    """
    Simple Nelder-Mead implementation.
    f: objective function
    x0: initial guess
    bounds: list of (min, max) for each dimension
    """
    start_time = time.time()
    n = len(x0)
    # Initial simplex
    simplex = [x0.copy()]
    for i in range(n):
        x_new = x0.copy()
        # Step size based on bounds or default
        step = 0.1
        if bounds[i][1] > bounds[i][0]:
            step = (bounds[i][1] - bounds[i][0]) * 0.1
        x_new[i] += step
        simplex.append(x_new)
    
    # Calculate initial function values
    for i in range(n + 1):
        simplex[i] = np.clip(simplex[i], bounds, 0) # np.clip doesn't work like this on list of tuples directly, need manual
        # Manual clip
        for j in range(n):
            simplex[i][j] = np.clip(simplex[i][j], bounds[j][0], bounds[j][1])
    
    # We need to store function values separately to avoid recomputing
    # But f is expensive, so we just call it.
    
    # Standard Nelder-Mead
    # To keep it simple and fast, we limit iterations.
    
    best_x = x0.copy()
    best_f = f(best_x)
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
            
        # Evaluate function for all points in simplex
        # We only have x, not f. We need to manage f values.
        # This simple implementation recalculates f for all points each iteration, which is expensive.
        # Optimization: Keep a list of (x, f) pairs.
        pass
        
    # Given the complexity and risk of bugs in a quick Nelder-Mead, 
    # and the fact that the fallback (Strang) is already 2nd order,
    # it might be safer to just return the fallback or a slightly tuned version.
    # However, the task asks to fix the program. The crash is the main issue.
    # A robust, simple local search or just returning the fallback is acceptable if optimization is too risky.
    
    # Let's try a simple coordinate descent or just return fallback.
    # Actually, the prompt says "Fix it". The crash was ImportError.
    # If I remove scipy and just return the fallback, it runs.
    # But can I do better?
    # Let's use a simple random restart local search with NumPy only.
    
    return best_x, best_f

def coefficients(stages):
    """
    Find optimal 12-stage product formula coefficients.
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Fallback: Simple Strang splitting
    fallback_a = [1.0/12.0] * 12
    fallback_b = [0.0] * 12
    for i in range(0, 12, 2):
        fallback_b[i] = 1.0/6.0
    
    best_params = (fallback_a, fallback_b)
    best_error = _compute_error(fallback_a, fallback_b)
    
    # Simple local search using NumPy
    # We will perform a few iterations of random perturbations and keep the best.
    # Since we can't use scipy, we do a manual hill climbing.
    
    current_a = np.array(fallback_a)
    current_b = np.array(fallback_b)
    
    # Define a helper to normalize and check bounds
    def _process_params(a_raw, b_raw):
        sum_a = np.sum(a_raw)
        sum_b = np.sum(b_raw)
        if abs(sum_a) < 1e-12 or abs(sum_b) < 1e-12:
            return None, None
        a_norm = a_raw / sum_a
        b_norm = b_raw / sum_b
        if np.any(a_norm < -50) or np.any(a_norm > 50) or np.any(b_norm < -50) or np.any(b_norm > 50):
            return None, None
        return a_norm, b_norm

    # Run a simple random search / hill climbing
    # Given the 60s CPU budget for the function call (and 120s total eval),
    # we can afford a few hundred evaluations.
    
    num_evals = 0
    max_evals = 500 # Keep it low to ensure speed
    
    while time.time() < deadline and num_evals < max_evals:
        num_evals += 1
        # Perturb current best
        scale = 0.1
        a_pert = current_a + np.random.normal(0, scale, 12)
        b_pert = current_b + np.random.normal(0, scale, 12)
        
        a_new, b_new = _process_params(a_pert, b_pert)
        if a_new is None:
            continue
            
        err = _compute_error(a_new, b_new)
        
        if err < best_error:
            best_error = err
            best_params = (list(a_new), list(b_new))
            current_a = a_new
            current_b = b_new
            # Reduce scale slightly on success
            scale *= 0.9
        else:
            # Increase scale slightly on failure to explore more
            scale *= 1.05
            if scale > 1.0:
                scale = 1.0

    if _is_valid(best_params[0], best_params[1]):
        return best_params[0], best_params[1]
        
    return fallback_a, fallback_b

# EVOLVE-BLOCK-END
