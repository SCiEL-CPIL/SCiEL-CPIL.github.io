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

def _symmetric_to_full(sym_a, sym_b):
    """
    Expand symmetric 6-element coefficients to full 12-element lists.
    Symmetry: a[11-i] = a[i], b[11-i] = b[i].
    Normalizes so that sum(a) = 1 and sum(b) = 1.
    Returns (a, b) or (None, None) if invalid.
    """
    if np.any(~np.isfinite(sym_a)) or np.any(~np.isfinite(sym_b)):
        return None, None
    
    sum_a = np.sum(sym_a)
    sum_b = np.sum(sym_b)
    
    if abs(sum_a) < 1e-15 or abs(sum_b) < 1e-15:
        return None, None
    
    norm_a = sym_a * (0.5 / sum_a)
    norm_b = sym_b * (0.5 / sum_b)
    
    if np.any(norm_a < -50) or np.any(norm_a > 50) or np.any(norm_b < -50) or np.any(norm_b > 50):
        return None, None
    
    a = np.zeros(12)
    b = np.zeros(12)
    for i in range(6):
        a[i] = norm_a[i]
        a[11-i] = norm_a[i]
        b[i] = norm_b[i]
        b[11-i] = norm_b[i]
    
    return a, b

def _differential_evolution(objective, x0, bounds, pop_size=20, max_iter=100, F=0.8, CR=0.9, time_limit=130.0):
    """
    Differential Evolution optimizer.
    """
    start_time = time.time()
    n = len(x0)
    
    # Initialize population
    pop = np.zeros((pop_size, n))
    pop[0] = x0
    for i in range(1, pop_size):
        for j in range(n):
            pop[i, j] = np.random.uniform(bounds[j][0], bounds[j][1])
    
    # Evaluate initial population
    fit = np.zeros(pop_size)
    for i in range(pop_size):
        fit[i] = objective(pop[i])
    
    best_idx = np.argmin(fit)
    best_x = pop[best_idx].copy()
    best_fit = fit[best_idx]
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        new_pop = pop.copy()
        new_fit = fit.copy()
        
        for i in range(pop_size):
            # Select three distinct random indices
            r1, r2, r3 = np.random.choice(pop_size, 3, replace=False)
            
            # Mutation
            v = pop[r1] + F * (pop[r2] - pop[r3])
            
            # Clamp to bounds
            for j in range(n):
                v[j] = np.clip(v[j], bounds[j][0], bounds[j][1])
            
            # Crossover
            c = np.random.rand(n)
            j_rand = np.random.randint(n)
            trial = np.zeros(n)
            for j in range(n):
                if c[j] < CR or j == j_rand:
                    trial[j] = v[j]
                else:
                    trial[j] = pop[i, j]
            
            # Clamp again to be safe
            for j in range(n):
                trial[j] = np.clip(trial[j], bounds[j][0], bounds[j][1])
            
            # Evaluate trial
            f_trial = objective(trial)
            
            # Selection
            if f_trial < fit[i]:
                new_pop[i] = trial
                new_fit[i] = f_trial
        
        pop = new_pop
        fit = new_fit
        
        current_best_idx = np.argmin(fit)
        if fit[current_best_idx] < best_fit:
            best_fit = fit[current_best_idx]
            best_x = pop[current_best_idx].copy()
            
        # Convergence check
        if np.max(fit) - np.min(fit) < 1e-10:
            break
            
    return best_x, best_fit

def coefficients(stages):
    """
    Find optimal 12-stage product formula coefficients using Differential Evolution.
    """
    start_time = time.time()
    deadline = start_time + 140
    
    # Fallback: Simple Strang splitting (symmetric)
    fallback_sym_a = np.array([1.0/12.0] * 6)
    fallback_sym_b = np.array([1.0/6.0, 0.0, 1.0/6.0, 0.0, 1.0/6.0, 0.0])
    
    fallback_a, fallback_b = _symmetric_to_full(fallback_sym_a, fallback_sym_b)
    if fallback_a is None:
        fallback_a = [1.0/12.0] * 12
        fallback_b = [0.0] * 12
        for i in range(0, 12, 2):
            fallback_b[i] = 1.0/6.0
    
    best_params = (list(fallback_a), list(fallback_b))
    best_error = _compute_error(fallback_a, fallback_b)
    
    # Define objective function for DE
    def objective(params):
        sym_a = params[:6]
        sym_b = params[6:12]
        a, b = _symmetric_to_full(sym_a, sym_b)
        if a is None:
            return 1e10
        return _compute_error(a, b)
    
    # Initial guess for DE
    x0 = np.concatenate([fallback_sym_a, fallback_sym_b])
    
    # Bounds for symmetric parameters
    # We want to allow a wide range of exploration
    bounds = [(-5.0, 5.0)] * 12
    
    # Run Differential Evolution
    try:
        best_x, best_val = _differential_evolution(objective, x0, bounds, pop_size=20, max_iter=100, F=0.8, CR=0.9, time_limit=130.0)
        
        sym_a_best = best_x[:6]
        sym_b_best = best_x[6:12]
        a_best, b_best = _symmetric_to_full(sym_a_best, sym_b_best)
        
        if a_best is not None and best_val < best_error:
            best_error = best_val
            best_params = (list(a_best), list(b_best))
    except Exception as e:
        pass
    
    # Final validation
    if _is_valid(best_params[0], best_params[1]):
        return best_params[0], best_params[1]
    
    return fallback_a, fallback_b

# EVOLVE-BLOCK-END
