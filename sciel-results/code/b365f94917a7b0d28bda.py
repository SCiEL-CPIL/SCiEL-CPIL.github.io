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

def _nelder_mead(objective, x0, bounds, max_iter=1000, time_limit=10.0):
    """
    Robust Nelder-Mead optimization.
    """
    start_time = time.time()
    n = len(x0)
    
    # Clip initial point
    x0 = np.array([np.clip(x0[i], bounds[i][0], bounds[i][1]) for i in range(n)])
    
    # Initialize simplex
    simplex = [x0.copy()]
    for i in range(n):
        x_new = x0.copy()
        step = 0.1
        if bounds[i][1] > bounds[i][0]:
            step = (bounds[i][1] - bounds[i][0]) * 0.1
        x_new[i] += step
        x_new[i] = np.clip(x_new[i], bounds[i][0], bounds[i][1])
        simplex.append(x_new)
    
    # Evaluate initial simplex
    simplex_vals = [objective(x) for x in simplex]
    
    # Sort
    order = np.argsort(simplex_vals)
    simplex = [simplex[i] for i in order]
    simplex_vals = [simplex_vals[i] for i in order]
    
    best_x = simplex[0].copy()
    best_val = simplex_vals[0]
    
    alpha = 1.0
    gamma = 2.0
    rho = 0.5
    sigma = 0.5
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        # Reflection
        x_worst = simplex[-1]
        x_centroid = np.mean(simplex[:-1], axis=0)
        x_ref = x_centroid + alpha * (x_centroid - x_worst)
        x_ref = np.array([np.clip(x_ref[i], bounds[i][0], bounds[i][1]) for i in range(n)])
        f_ref = objective(x_ref)
        
        if f_ref < simplex_vals[0]:
            # Expansion
            x_exp = x_centroid + gamma * (x_ref - x_centroid)
            x_exp = np.array([np.clip(x_exp[i], bounds[i][0], bounds[i][1]) for i in range(n)])
            f_exp = objective(x_exp)
            if f_exp < f_ref:
                simplex[-1] = x_exp
                simplex_vals[-1] = f_exp
            else:
                simplex[-1] = x_ref
                simplex_vals[-1] = f_ref
        elif f_ref < simplex_vals[-2]:
            # Accept reflection
            simplex[-1] = x_ref
            simplex_vals[-1] = f_ref
        else:
            # Contraction
            x_con = x_centroid + rho * (x_worst - x_centroid)
            x_con = np.array([np.clip(x_con[i], bounds[i][0], bounds[i][1]) for i in range(n)])
            f_con = objective(x_con)
            if f_con < simplex_vals[-1]:
                simplex[-1] = x_con
                simplex_vals[-1] = f_con
            else:
                # Shrink
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    simplex[i] = np.array([np.clip(simplex[i][j], bounds[j][0], bounds[j][1]) for j in range(n)])
                    simplex_vals[i] = objective(simplex[i])
        
        # Sort
        order = np.argsort(simplex_vals)
        simplex = [simplex[i] for i in order]
        simplex_vals = [simplex_vals[i] for i in order]
        
        if simplex_vals[0] < best_val:
            best_val = simplex_vals[0]
            best_x = simplex[0].copy()
            
        # Convergence check
        if np.max(simplex_vals) - np.min(simplex_vals) < 1e-10:
            break
            
    return best_x, best_val

def coefficients(stages):
    """
    Find optimal 12-stage product formula coefficients using multi-restart Nelder-Mead.
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
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
    
    # Define objective function for Nelder-Mead
    def objective(params):
        sym_a = params[:6]
        sym_b = params[6:12]
        a, b = _symmetric_to_full(sym_a, sym_b)
        if a is None:
            return 1e10
        return _compute_error(a, b)
    
    # Bounds for symmetric parameters
    bounds = [(-10.0, 10.0)] * 12
    
    # Initial guesses
    # 1. Uniform (1st order)
    x0_1 = np.concatenate([np.array([1.0/12.0]*6), np.array([1.0/6.0, 0.0, 1.0/6.0, 0.0, 1.0/6.0, 0.0])])
    
    # 2. 4th order approximation (scaled to fit symmetric 6-element sum=0.5)
    # Standard 4th order coefficients for A (d) and B (c)
    c1 = 1.0 / (4.0 - 4.0**(1.0/3.0))
    c2 = -4.0**(1.0/3.0) / (4.0 - 4.0**(1.0/3.0))
    c3 = 0.5 - c1
    d1 = 1.0 / (2.0 * (1.0 - 2.0**(1.0/3.0)))
    d2 = -2.0**(1.0/3.0) / (2.0 * (1.0 - 2.0**(1.0/3.0)))
    d3 = 0.5 - d1
    
    # Map to symmetric 6-element vectors
    # Sequence: B A B A B A B A B A B (11 terms)
    # Our symmetric 12-stage: b1 a1 b2 a2 b3 a3 b4 a4 b5 a5 b6 a6 b6 a5 b4 a4 b3 a3 b2 a2 b1 a1
    # We can distribute the 4th order weights into this structure.
    # For simplicity, we'll just use a few random starts and the uniform start.
    
    initial_points = [x0_1]
    
    # Add random starts
    np.random.seed(42)
    for _ in range(5):
        sym_a_rand = np.random.uniform(0.01, 0.2, 6)
        sym_b_rand = np.random.uniform(0.01, 0.2, 6)
        # Ensure they are not too similar
        initial_points.append(np.concatenate([sym_a_rand, sym_b_rand]))
    
    # Run Nelder-Mead from each initial point
    for x0 in initial_points:
        if time.time() > deadline:
            break
        try:
            best_x, best_val = _nelder_mead(objective, x0, bounds, max_iter=500, time_limit=20.0)
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
