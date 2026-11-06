import numpy as np
import time
from scipy.linalg import expm
from scipy.optimize import minimize, differential_evolution

def _build_hamiltonians():
    """Build A (odd bonds) and B (even bonds) for 6-site open XXZ chain.
    Anisotropy gamma=0.8, field h=0.3.
    XXZ Hamiltonian: sum_j [gamma/2 (S_j^+ S_{j+1}^- + S_j^- S_{j+1}^+) + S_j^z S_{j+1}^z] - h sum_j S_j^z
    Split: A = odd bonds (1-2, 3-4, 5-6) + half field, B = even bonds (2-3, 4-5) + half field
    """
    N = 6
    dim = 2 ** N
    gamma = 0.8
    h = 0.3
    
    # Build spin operators
    sx = np.array([[0, 0.5], [0.5, 0]], dtype=complex)
    sy = np.array([[0, -0.5j], [0.5j, 0]], dtype=complex)
    sz = np.array([[0.5, 0], [0, -0.5]], dtype=complex)
    sp = sx + 1j * sy  # sigma_+ = |0><1|
    sm = sx - 1j * sy  # sigma_- = |1><0|
    I2 = np.eye(2, dtype=complex)
    
    def op_on_site(op, site):
        """Apply single-site operator at given site (0-indexed)."""
        ops = [I2] * N
        ops[site] = op
        result = ops[0]
        for o in ops[1:]:
            result = np.kron(result, o)
        return result
    
    A = np.zeros((dim, dim), dtype=complex)
    B = np.zeros((dim, dim), dtype=complex)
    
    # Odd bonds: (0,1), (2,3), (4,5) -> indices 1-2, 3-4, 5-6 in 1-based
    odd_bonds = [(0, 1), (2, 3), (4, 5)]
    # Even bonds: (1,2), (3,4) -> indices 2-3, 4-5 in 1-based
    even_bonds = [(1, 2), (3, 4)]
    
    def bond_hamiltonian(i, j):
        """XXZ bond Hamiltonian between sites i and j."""
        # gamma/2 (S_i^+ S_j^- + S_i^- S_j^+) + S_i^z S_j^z
        sp_i = op_on_site(sp, i)
        sm_j = op_on_site(sm, j)
        sm_i = op_on_site(sm, i)
        sp_j = op_on_site(sp, j)
        sz_i = op_on_site(sz, i)
        sz_j = op_on_site(sz, j)
        return gamma / 2 * (sp_i @ sm_j + sm_i @ sp_j) + sz_i @ sz_j
    
    for i, j in odd_bonds:
        A += bond_hamiltonian(i, j)
    for i, j in even_bonds:
        B += bond_hamiltonian(i, j)
    
    # Half field on each
    half_field = 0.5 * h
    for s in range(N):
        sz_s = op_on_site(sz, s)
        A -= half_field * sz_s
        B -= half_field * sz_s
    
    return A, B

def _error(a, b, A, B, times):
    """Compute mean spectral norm error."""
    dim = A.shape[0]
    errors = []
    for t in times:
        # Build product formula U(t) = prod_k exp(-i b_k t B) exp(-i a_k t A)
        U = np.eye(dim, dtype=complex)
        for k in range(len(a)):
            # exp(-i b_k t B) then exp(-i a_k t A)
            if b[k] != 0:
                U = expm(-1j * b[k] * t * B) @ U
            if a[k] != 0:
                U = expm(-1j * a[k] * t * A) @ U
        # Exact propagator
        H = A + B
        U_exact = expm(-1j * t * H)
        err = np.linalg.norm(U - U_exact, 2)
        errors.append(err)
    return np.mean(errors)

def coefficients(stages):
    start_time = time.time()
    deadline = 130.0  # safety margin before 150s kill
    
    A, B = _build_hamiltonians()
    dim = A.shape[0]
    times = [0.25, 0.5, 1.0]
    
    # Precompute expm for various coefficients to speed up evaluation
    # We'll use a cached approach
    
    # Initial guess: Strang splitting
    substeps = stages // 2
    a0 = [0.5 / substeps] * (2 * substeps)
    b0 = []
    for _ in range(substeps):
        b0 += [1.0 / substeps, 0.0]
    while len(a0) < stages:
        a0.append(0.0)
        b0.append(0.0)
    
    # Try a few different initial strategies
    best_a = a0[:]
    best_b = b0[:]
    best_err = _error(best_a, best_b, A, B, times)
    
    elapsed = time.time() - start_time
    if elapsed > deadline:
        return best_a, best_b
    
    # Use differential evolution for global search
    def objective(x):
        a = x[:stages]
        b = x[stages:]
        # Enforce sum constraints via penalty
        sum_a = np.sum(a)
        sum_b = np.sum(b)
        penalty = 100 * (sum_a - 1)**2 + 100 * (sum_b - 1)**2
        err = _error(a, b, A, B, times)
        return err + penalty
    
    # Try differential evolution with limited population
    try:
        bounds = [(-50, 50)] * (2 * stages)
        result = differential_evolution(
            objective, bounds,
            maxiter=50, popsize=8,
            tol=1e-6, seed=42,
            polish=True,
            workers=1
        )
        if result.fun < best_err:
            x = result.x
            a = x[:stages]
            b = x[stages:]
            # Normalize to ensure sum=1
            sa = np.sum(a)
            sb = np.sum(b)
            if abs(sa) > 1e-10:
                a = a / sa
            if abs(sb) > 1e-10:
                b = b / sb
            # Clip to [-50, 50]
            a = np.clip(a, -50, 50).tolist()
            b = np.clip(b, -50, 50).tolist()
            best_a = a
            best_b = b
            best_err = result.fun
    except Exception:
        pass
    
    elapsed = time.time() - start_time
    if elapsed > deadline:
        return best_a, best_b
    
    # Polish with local optimization
    try:
        def objective_local(x):
            a = x[:stages]
            b = x[stages:]
            sum_a = np.sum(a)
            sum_b = np.sum(b)
            penalty = 1000 * (sum_a - 1)**2 + 1000 * (sum_b - 1)**2
            err = _error(a, b, A, B, times)
            return err + penalty
        
        x0 = np.array(best_a + best_b)
        result2 = minimize(
            objective_local, x0,
            method='L-BFGS-B',
            bounds=[(-50, 50)] * (2 * stages),
            options={'maxiter': 200, 'ftol': 1e-12}
        )
        if result2.fun < best_err:
            x = result2.x
            a = x[:stages]
            b = x[stages:]
            sa = np.sum(a)
            sb = np.sum(b)
            if abs(sa) > 1e-10:
                a = a / sa
            if abs(sb) > 1e-10:
                b = b / sb
            a = np.clip(a, -50, 50).tolist()
            b = np.clip(b, -50, 50).tolist()
            best_a = a
            best_b = b
            best_err = result2.fun
    except Exception:
        pass
    
    # Final normalization and clipping
    sa = sum(best_a)
    sb = sum(best_b)
    if abs(sa) > 1e-10:
        best_a = [x / sa for x in best_a]
    if abs(sb) > 1e-10:
        best_b = [x / sb for x in best_b]
    best_a = [max(-50, min(50, x)) for x in best_a]
    best_b = [max(-50, min(50, x)) for x in best_b]
    
    # Re-normalize after clipping
    sa = sum(best_a)
    sb = sum(best_b)
    if abs(sa) > 1e-10:
        best_a = [x / sa for x in best_a]
    if abs(sb) > 1e-10:
        best_b = [x / sb for x in best_b]
    
    return best_a, best_b
