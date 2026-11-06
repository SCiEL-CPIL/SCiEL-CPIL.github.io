"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
import scipy.optimize as opt

# EVOLVE-BLOCK-START

# Build the Heisenberg chain Hamiltonian components
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

# Pre-compute the Hamiltonians and eigen-decompositions for fast expm
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
    if abs(sum(a) - 1.0) > 1e-10 or abs(sum(b) - 1.0) > 1e-10:
        return False
    return True

def coefficients(stages):
    """
    Find optimal 12-stage product formula coefficients using symmetric optimization.
    Uses L-BFGS-B with multiple starting points for robustness.
    """
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit
    
    # Symmetric structure: a[0..5] determine a[11..6], b[0..5] determine b[11..6]
    # a[11-i] = a[i], b[11-i] = b[i]
    # Sum(a) = 1 => 2 * sum(a[0..5]) = 1 => sum(a[0..5]) = 0.5
    
    # Parameterization:
    # We optimize 5 independent a's and 5 independent b's.
    # a_5 = 0.5 - sum(a_0..a_4)
    # b_5 = 0.5 - sum(b_0..b_4)
    
    def expand_params(params):
        a_sym = params[:5]
        b_sym = params[5:10]
        
        a_full_sym = np.zeros(6)
        a_full_sym[:5] = a_sym
        a_full_sym[5] = 0.5 - np.sum(a_sym)
        
        b_full_sym = np.zeros(6)
        b_full_sym[:5] = b_sym
        b_full_sym[5] = 0.5 - np.sum(b_sym)
        
        a = np.zeros(12)
        b = np.zeros(12)
        for i in range(6):
            a[i] = a_full_sym[i]
            a[11-i] = a_full_sym[i]
            b[i] = b_full_sym[i]
            b[11-i] = b_full_sym[i]
            
        return a, b

    def objective(params):
        a, b = expand_params(params)
        # Check bounds
        if np.any(np.abs(a) > 50) or np.any(np.abs(b) > 50):
            return 1e10
        # Check for non-finite
        if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
            return 1e10
        return _compute_error(a, b)

    # Starting points
    # 1. Uniform
    uniform_a = np.full(5, 0.5/6.0)
    uniform_b = np.full(5, 0.5/6.0)
    x0_uniform = np.concatenate([uniform_a, uniform_b])
    
    # 2. 4th Order Suzuki-Trotter like
    # Standard 4th order 3-stage (Strang-like with extra steps) coefficients are complex.
    # Let's try a "bunched" distribution: put more weight in the middle.
    # Or just use random perturbations of uniform.
    
    start_points = [x0_uniform]
    
    # Add a few random starts
    rng = np.random.default_rng(42)
    for _ in range(4):
        a_rand = rng.normal(0.5/6.0, 0.05, 5)
        b_rand = rng.normal(0.5/6.0, 0.05, 5)
        # Ensure the derived 6th element is not too far out
        # a_5 = 0.5 - sum(a_rand)
        # If sum(a_rand) is close to 0.5, a_5 is close to 0.
        # This is a valid starting point.
        start_points.append(np.concatenate([a_rand, b_rand]))

    best_val = np.inf
    best_x = x0_uniform.copy()
    
    bounds = [(-50, 50)] * 10
    
    for x0 in start_points:
        if time.time() > deadline - 5:
            break
            
        try:
            res = opt.minimize(objective, x0, method='L-BFGS-B', bounds=bounds, 
                               options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8})
            
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        except Exception:
            pass

    # Final expansion
    a, b = expand_params(best_x)
    
    # Fallback to uniform if optimization failed or produced invalid results
    if not _is_valid(list(a), list(b)) or best_val > 1e-3:
        a = np.full(12, 1.0/12.0)
        b = np.full(12, 1.0/12.0)
    else:
        # Small cleanup to ensure exact sums due to floating point
        sum_a = np.sum(a)
        sum_b = np.sum(b)
        if sum_a != 0:
            a = a * (1.0 / sum_a)
        if sum_b != 0:
            b = b * (1.0 / sum_b)
            
        if not _is_valid(list(a), list(b)):
            a = np.full(12, 1.0/12.0)
            b = np.full(12, 1.0/12.0)

    return list(a), list(b)

# EVOLVE-BLOCK-END
