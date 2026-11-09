"""Product-formula coefficient optimizer for 6-site Heisenberg chain."""
import numpy as np
import time

# Deadline guard
_DEADLINE = 140.0
_T0 = time.time()

def _remaining():
    return _DEADLINE - (time.time() - _T0)

def _build_hamiltonians():
    """Build A (odd bonds + half field) and B (even bonds + half field) for 6-site open XXZ chain."""
    N = 6
    delta = 0.8
    field = 0.3
    
    sx = np.array([[0, 0.5], [0.5, 0]], dtype=complex)
    sy = np.array([[0, -0.5j], [0.5j, 0]], dtype=complex)
    sz = np.array([[0.5, 0], [0, -0.5]], dtype=complex)
    I = np.eye(2, dtype=complex)
    
    def kron_all(ops):
        result = np.array([[1]], dtype=complex)
        for op in ops:
            result = np.kron(result, op)
        return result
    
    def site_op(op, site):
        ops = [I] * N
        ops[site] = op
        return kron_all(ops)
    
    def bond_op(i, j):
        ops_sz = [I] * N
        ops_sz[i] = sz
        ops_sz[j] = sz
        ops_sx = [I] * N
        ops_sx[i] = sx
        ops_sx[j] = sx
        ops_sy = [I] * N
        ops_sy[i] = sy
        ops_sy[j] = sy
        return delta * (kron_all(ops_sz) + kron_all(ops_sx) + kron_all(ops_sy))
    
    B = bond_op(0, 1) + bond_op(2, 3) + bond_op(4, 5)
    A = bond_op(1, 2) + bond_op(3, 4)
    
    half_field = field / 2.0
    field_op = np.zeros((2**N, 2**N), dtype=complex)
    for i in range(N):
        field_op += half_field * site_op(sz, i)
    
    A += field_op
    B += field_op
    
    return A, B

def _compute_error(a, b, A, B, H_eig, ev_a, vec_a, ev_b, vec_b):
    """Compute mean spectral norm error at t = 0.25, 0.5, 1.0."""
    times = [0.25, 0.5, 1.0]
    H = A + B
    eigvals_h, eigvecs_h = H_eig
    
    errors = []
    for t in times:
        exact = (eigvecs_h * np.exp(-1j * t * eigvals_h)) @ eigvecs_h.conj().T
        
        U = np.eye(A.shape[0], dtype=complex)
        for k in range(len(a)):
            exp_b = (vec_b * np.exp(-1j * b[k] * t * ev_b)) @ vec_b.conj().T
            exp_a = (vec_a * np.exp(-1j * a[k] * t * ev_a)) @ vec_a.conj().T
            U = exp_b @ exp_a @ U
        
        err = np.linalg.norm(U - exact, ord=2)
        errors.append(err)
    return np.mean(errors)

def _nelder_mead(f, x0, max_iter=200, step=0.01, deadline=None):
    """Simple Nelder-Mead implementation."""
    n = len(x0)
    simplex = [x0[:]]
    for i in range(n):
        xi = x0[:]
        xi[i] += step
        simplex.append(xi)
    
    fvals = [f(x) for x in simplex]
    
    for it in range(max_iter):
        if deadline is not None and _remaining() < deadline:
            break
        
        order = np.argsort(fvals)
        simplex = [simplex[i] for i in order]
        fvals = [fvals[i] for i in order]
        
        centroid = np.mean(simplex[:n], axis=0)
        
        alpha = 1.0
        x_ref = centroid + alpha * (centroid - simplex[n])
        f_ref = f(x_ref)
        
        if f_ref < fvals[0]:
            gamma = 2.0
            x_exp = centroid + gamma * (centroid - simplex[n])
            f_exp = f(x_exp)
            if f_exp < f_ref:
                simplex[n] = x_exp
                fvals[n] = f_exp
            else:
                simplex[n] = x_ref
                fvals[n] = f_ref
        elif f_ref < fvals[n-1]:
            simplex[n] = x_ref
            fvals[n] = f_ref
        else:
            rho = 0.5
            x_con = centroid + rho * (simplex[n] - centroid)
            f_con = f(x_con)
            if f_con < fvals[n]:
                simplex[n] = x_con
                fvals[n] = f_con
            else:
                for i in range(1, n+1):
                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                    fvals[i] = f(simplex[i])
    
    best_idx = np.argmin(fvals)
    return simplex[best_idx], fvals[best_idx]

def coefficients(stages):
    """Optimize 12-stage product formula coefficients."""
    if stages != 12:
        a = [1.0/stages] * stages
        b = [1.0/stages] * stages
        return a, b
    
    A, B = _build_hamiltonians()
    H = A + B
    
    # Precompute eigen-decompositions
    ev_h, vec_h = np.linalg.eigh(H)
    ev_a, vec_a = np.linalg.eigh(A)
    ev_b, vec_b = np.linalg.eigh(B)
    
    H_eig = (ev_h, vec_h)
    
    # Start with equal weights
    best_a = [1.0/12] * 12
    best_b = [1.0/12] * 12
    best_err = _compute_error(best_a, best_b, A, B, H_eig, ev_a, vec_a, ev_b, vec_b)
    
    # Use full 24-parameter optimization (no symmetry constraint)
    # This allows asymmetric patterns that can better cancel commutator errors
    
    def objective(params):
        a = params[:12]
        b = params[12:24]
        sa = sum(a)
        sb = sum(b)
        if abs(sa) < 1e-15 or abs(sb) < 1e-15:
            return 1e10
        a = [x/sa for x in a]
        b = [x/sb for x in b]
        a = [max(-50, min(50, x)) for x in a]
        b = [max(-50, min(50, x)) for x in b]
        return _compute_error(a, b, A, B, H_eig, ev_a, vec_a, ev_b, vec_b)
    
    x0 = [1.0/12] * 24
    
    # First Nelder-Mead pass
    if _remaining() > 40:
        x_opt, f_opt = _nelder_mead(objective, x0, max_iter=150, step=0.02, deadline=30)
        if f_opt < best_err:
            best_err = f_opt
            best_a = x_opt[:12]
            best_b = x_opt[12:24]
    
    # Second pass with smaller steps from the first result
    if _remaining() > 20:
        x_opt2, f_opt2 = _nelder_mead(objective, x_opt, max_iter=100, step=0.005, deadline=15)
        if f_opt2 < best_err:
            best_err = f_opt2
            best_a = x_opt2[:12]
            best_b = x_opt2[12:24]
    
    # Try a few random restarts if time permits
    if _remaining() > 15:
        rng = np.random.default_rng(42)
        for restart in range(3):
            if _remaining() < 10:
                break
            x_rand = best_a + rng.normal(0, 0.05, 12)
            x_rand = np.concatenate([x_rand, best_b + rng.normal(0, 0.05, 12)])
            x_rand = np.clip(x_rand, -1, 1)
            x_opt3, f_opt3 = _nelder_mead(objective, x_rand, max_iter=50, step=0.01, deadline=8)
            if f_opt3 < best_err:
                best_err = f_opt3
                best_a = x_opt3[:12]
                best_b = x_opt3[12:24]
    
    # Final normalization
    sa = sum(best_a)
    sb = sum(best_b)
    if abs(sa) > 1e-15:
        best_a = [x/sa for x in best_a]
    if abs(sb) > 1e-15:
        best_b = [x/sb for x in best_b]
    
    return best_a, best_b
