"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.linalg import expm

# EVOLVE-BLOCK-START

# Build the Heisenberg chain Hamiltonian components
def _build_hamiltonians():
    """
    Build A (odd bonds + half field) and B (even bonds + half field) for a
    6-site open XXZ Heisenberg chain with anisotropy 0.8 and field 0.3.
    
    The full Hamiltonian is H = A + B where:
    - A contains odd-bond interactions (bonds 1-2, 3-4, 5-6) + half the uniform field
    - B contains even-bond interactions (bonds 2-3, 4-5) + half the uniform field
    
    XXZ Heisenberg model: H = sum_j [Jx (Sx_j Sx_{j+1} + Sy_j Sy_{j+1}) + Jz Sz_j Sz_{j+1}] - h sum_j Sz_j
    With anisotropy gamma = Jz/Jx = 0.8, we set Jx = 1, Jz = 0.8, h = 0.3.
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
    # Odd bonds: (0,1), (2,3), (4,5)
    for j in [0, 2, 4]:
        # Jx * (Sx_j Sx_{j+1} + Sy_j Sy_{j+1}) + Jz * Sz_j Sz_{j+1}
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
    
    # Half field for A
    for j in range(N):
        A -= (h / 2.0) * site_op(sz, j)
    
    # Build B: even bonds (1-2, 3-4) + half field
    B = np.zeros((dim, dim), dtype=complex)
    # Even bonds: (1,2), (3,4)
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
    
    # Half field for B
    for j in range(N):
        B -= (h / 2.0) * site_op(sz, j)
    
    return A, B

# Pre-compute the Hamiltonians and exact propagators
_A, _B = _build_hamiltonians()
_H = _A + _B

# Pre-compute matrix exponentials for the three time points
_T = [0.25, 0.5, 1.0]
_exact_U = [expm(-1j * t * _H) for t in _T]

def _compute_error(a, b):
    """Compute the mean spectral norm error over the three time points."""
    errors = []
    for t, U_exact in zip(_T, _exact_U):
        U = np.eye(2**6, dtype=complex)
        for k in range(12):
            if b[k] != 0.0:
                U = expm(-1j * b[k] * t * _B) @ U
            if a[k] != 0.0:
                U = expm(-1j * a[k] * t * _A) @ U
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
    Find optimal 12-stage product formula coefficients using gradient-based optimization.
    """
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    # Start with a reasonable initial guess: symmetric Strang-like splitting
    # For 12 stages: 6 substeps of Strang splitting
    # Each substep: exp(-i b t B) exp(-i a t A) exp(-i b t B)
    # But our form is prod_k exp(-i b_k t B) exp(-i a_k t A)
    # So for Strang: a = [1/(2*6)]*6 pattern won't work directly.
    # Let's use: b_k = 1/6 for k=1..6, a_k = 1/6 for k=1..6, rest 0
    # Actually, let's try a symmetric pattern.
    
    # Initial guess: uniform distribution
    a = [1.0/12] * 12
    b = [1.0/12] * 12
    
    best_a = list(a)
    best_b = list(b)
    best_err = _compute_error(a, b)
    
    # Gradient-based optimization using finite differences
    # We optimize over 24 variables (12 a's and 12 b's) with constraints sum(a)=1, sum(b)=1
    
    # Use a change of variables: express a[0..10] and b[0..10] as free,
    # then a[11] = 1 - sum(a[0..10]), b[11] = 1 - sum(b[0..10])
    
    def _objective(params):
        a_local = list(params[:12])
        b_local = list(params[12:24])
        a_local[11] = 1.0 - sum(a_local[:11])
        b_local[11] = 1.0 - sum(b_local[:11])
        
        # Check bounds
        for x in a_local + b_local:
            if x < -50 or x > 50:
                return 1e10
        return _compute_error(a_local, b_local)
    
    # Nelder-Mead style simplex optimization
    # This avoids needing gradients and handles the constraint naturally
    
    x0 = np.array(a + b)
    x0[11] = 1.0 - sum(x0[:11])
    x0[23] = 1.0 - sum(x0[12:23])
    
    # Simplex method
    n = 24
    simplex = [x0.copy()]
    for i in range(n):
        xi = x0.copy()
        xi[i] += 0.01 if x0[i] >= 0 else -0.01
        simplex.append(xi)
    
    # Order simplex by objective value
    simplex_vals = [_objective(s) for s in simplex]
    order = np.argsort(simplex_vals)
    simplex = [simplex[i] for i in order]
    simplex_vals = [simplex_vals[i] for i in order]
    
    best_val = simplex_vals[0]
    best_x = simplex[0].copy()
    
    alpha = 1.0  # reflection
    gamma = 2.0  # expansion
    rho = 0.5   # contraction
    sigma = 0.5 # shrink
    
    max_iter = 5000
    for it in range(max_iter):
        if time.time() > deadline - 10:
            break
        
        # Reflect worst point
        x_worst = simplex[-1]
        x_centroid = np.mean(simplex[:-1], axis=0)
        x_ref = x_centroid + alpha * (x_centroid - x_worst)
        f_ref = _objective(x_ref)
        
        if f_ref < simplex_vals[0]:
            # Try expansion
            x_exp = x_centroid + gamma * (x_ref - x_centroid)
            f_exp = _objective(x_exp)
            if f_exp < f_ref:
                simplex[-1] = x_exp
                simplex_vals[-1] = f_exp
            else:
                simplex[-1] = x_ref
                simplex_vals[-1] = f_ref
        elif f_ref < simplex_vals[-2]:
            simplex[-1] = x_ref
            simplex_vals[-1] = f_ref
        else:
            # Contraction
            x_con = x_centroid + rho * (x_worst - x_centroid)
            f_con = _objective(x_con)
            if f_con < simplex_vals[-1]:
                simplex[-1] = x_con
                simplex_vals[-1] = f_con
            else:
                # Shrink
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    simplex_vals[i] = _objective(simplex[i])
        
        # Reorder
        order = np.argsort(simplex_vals)
        simplex = [simplex[i] for i in order]
        simplex_vals = [simplex_vals[i] for i in order]
        
        if simplex_vals[0] < best_val:
            best_val = simplex_vals[0]
            best_x = simplex[0].copy()
        
        # Convergence check
        if max(simplex_vals) - min(simplex_vals) < 1e-12:
            break
    
    # Extract best solution
    a_best = list(best_x[:12])
    b_best = list(best_x[12:24])
    a_best[11] = 1.0 - sum(a_best[:11])
    b_best[11] = 1.0 - sum(b_best[:11])
    
    # Clamp to bounds
    for i in range(12):
        a_best[i] = max(-50, min(50, a_best[i]))
        b_best[i] = max(-50, min(50, b_best[i]))
    
    # Re-normalize
    sum_a = sum(a_best)
    sum_b = sum(b_best)
    if abs(sum_a) > 1e-15:
        a_best = [x / sum_a for x in a_best]
    if abs(sum_b) > 1e-15:
        b_best = [x / sum_b for x in b_best]
    
    # Verify
    if not _is_valid(a_best, b_best):
        # Fall back to initial guess
        return [1.0/12] * 12, [1.0/12] * 12
    
    return a_best, b_best

# EVOLVE-BLOCK-END
