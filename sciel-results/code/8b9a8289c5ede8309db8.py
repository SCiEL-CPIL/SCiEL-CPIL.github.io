"""Symmetric Nelder-Mead optimization for 12-stage product formula."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _build_hamiltonians():
    """Build A (odd bonds) and B (even bonds) for 6-site open XXZ chain.
    Anisotropy gamma=0.8, field h=0.3.
    Split: A = odd bonds (0-1, 2-3, 4-5) + half field, B = even bonds (1-2, 3-4) + half field
    """
    N = 6
    dim = 2 ** N
    gamma = 0.8
    h = 0.3

    sx = np.array([[0, 0.5], [0.5, 0]], dtype=complex)
    sy = np.array([[0, -0.5j], [0.5j, 0]], dtype=complex)
    sz = np.array([[0.5, 0], [0, -0.5]], dtype=complex)
    sp = sx + 1j * sy
    sm = sx - 1j * sy
    I2 = np.eye(2, dtype=complex)

    def op_on_site(op, site):
        ops = [I2] * N
        ops[site] = op
        result = ops[0]
        for o in ops[1:]:
            result = np.kron(result, o)
        return result

    A = np.zeros((dim, dim), dtype=complex)
    B = np.zeros((dim, dim), dtype=complex)

    odd_bonds = [(0, 1), (2, 3), (4, 5)]
    even_bonds = [(1, 2), (3, 4)]

    def bond_hamiltonian(i, j):
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

    half_field = 0.5 * h
    for s in range(N):
        sz_s = op_on_site(sz, s)
        A -= half_field * sz_s
        B -= half_field * sz_s

    return A, B

def _error_fast(a, b, A_eigvals, A_eigvecs, B_eigvals, B_eigvecs, U_exact_cache, times):
    """Compute mean spectral norm error efficiently using precomputed eigendecompositions."""
    dim = A_eigvecs.shape[0]
    errors = []
    for t in times:
        U = np.eye(dim, dtype=complex)
        for k in range(len(a)):
            if abs(b[k]) > 1e-15:
                theta_b = b[k] * t
                exp_b = np.exp(-1j * theta_b * B_eigvals)
                U = U @ (B_eigvecs @ np.diag(exp_b) @ B_eigvecs.conj().T)
            if abs(a[k]) > 1e-15:
                theta_a = a[k] * t
                exp_a = np.exp(-1j * theta_a * A_eigvals)
                U = U @ (A_eigvecs @ np.diag(exp_a) @ A_eigvecs.conj().T)
        U_exact = U_exact_cache[t]
        err = np.linalg.norm(U - U_exact, 2)
        errors.append(err)
    return float(np.mean(errors))

def _normalize_and_clip(a, b):
    """Normalize so sums are 1 and clip to [-50, 50]."""
    sa = sum(a)
    sb = sum(b)
    if abs(sa) > 1e-10:
        a = [x / sa for x in a]
    if abs(sb) > 1e-10:
        b = [x / sb for x in b]
    a = [max(-50.0, min(50.0, x)) for x in a]
    b = [max(-50.0, min(50.0, x)) for x in b]
    sa = sum(a)
    sb = sum(b)
    if abs(sa) > 1e-10:
        a = [x / sa for x in a]
    if abs(sb) > 1e-10:
        b = [x / sb for x in b]
    return a, b

def _nelder_mead(f, x0, max_iter=300, tol=1e-8, timeout=None, start_time=None):
    """Nelder-Mead simplex optimization. Returns (best_x, best_f)."""
    n = len(x0)
    alpha = 1.0
    gamma = 2.0
    rho = 0.5
    sigma = 0.5

    # Build simplex
    simplex = [np.array(x0, dtype=float)]
    for i in range(n):
        xi = np.array(x0, dtype=float).copy()
        xi[i] += 0.1
        simplex.append(xi)

    fvals = []
    for v in simplex:
        try:
            fv = f(v)
        except Exception:
            fv = 1e10
        fvals.append(fv)

    best_x = simplex[0].copy()
    best_f = min(fvals)

    for it in range(max_iter):
        if timeout is not None and start_time is not None:
            if time.time() - start_time > timeout:
                break

        # Sort simplex by f-value
        order = np.argsort(fvals)
        simplex = [simplex[i] for i in order]
        fvals = [fvals[i] for i in order]

        # Check convergence
        if abs(fvals[-1] - fvals[0]) < tol:
            break

        # Centroid of all but worst
        centroid = np.mean(simplex[:-1], axis=0)
        worst = simplex[-1]

        # Reflection
        x_r = centroid + alpha * (centroid - worst)
        try:
            f_r = f(x_r)
        except Exception:
            f_r = 1e10

        if f_r < fvals[0]:
            # Expansion
            x_e = centroid + gamma * (x_r - centroid)
            try:
                f_e = f(x_e)
            except Exception:
                f_e = 1e10
            if f_e < f_r:
                simplex[-1] = x_e
                fvals[-1] = f_e
            else:
                simplex[-1] = x_r
                fvals[-1] = f_r
        elif f_r < fvals[-2]:
            simplex[-1] = x_r
            fvals[-1] = f_r
        else:
            # Contraction
            x_c = centroid + rho * (worst - centroid)
            try:
                f_c = f(x_c)
            except Exception:
                f_c = 1e10
            if f_c < fvals[-1]:
                simplex[-1] = x_c
                fvals[-1] = f_c
            else:
                # Shrink
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    try:
                        fvals[i] = f(simplex[i])
                    except Exception:
                        fvals[i] = 1e10

        # Track best
        for i in range(n + 1):
            if fvals[i] < best_f:
                best_f = fvals[i]
                best_x = simplex[i].copy()

    return best_x, best_f

def coefficients(stages):
    start_time = time.time()
    deadline = 140.0

    if stages != 12:
        substeps = max(1, stages // 2)
        a = [0.5 / substeps] * (2 * substeps)
        b = []
        for _ in range(substeps):
            b += [1.0 / substeps, 0.0]
        while len(a) < stages:
            a.append(0.0)
            b.append(0.0)
        return a, b

    # Build Hamiltonians
    A, B = _build_hamiltonians()
    dim = A.shape[0]
    times = [0.25, 0.5, 1.0]

    # Precompute eigendecompositions
    A_eigvals, A_eigvecs = np.linalg.eigh(A)
    B_eigvals, B_eigvecs = np.linalg.eigh(B)
    H = A + B
    H_eigvals, H_eigvecs = np.linalg.eigh(H)

    U_exact_cache = {}
    for t in times:
        exp_h = np.exp(-1j * t * H_eigvals)
        U_exact_cache[t] = H_eigvecs @ np.diag(exp_h) @ H_eigvecs.conj().T

    # Symmetric parameterization:
    # a[k] = a[11-k], b[k] = b[11-k]
    # Sum a = 1 => 2 * sum(a[0..5]) = 1 => sum(a[0..5]) = 0.5
    # Let a_k = 1/12 + x_k for k=0..5.
    # Sum = 6/12 + sum(x) = 0.5 + sum(x) = 0.5 => sum(x) = 0.
    # Use 5 free vars x0..x4. x5 = -sum(x0..x4).
    # Same for b. Total 10 vars.

    def _error_from_sym_params(params):
        xa = params[:5]
        xb = params[5:]
        
        a_half = np.zeros(6)
        a_half[:5] = xa
        a_half[5] = -np.sum(xa)
        
        a_coeffs = [0.0] * 12
        for k in range(6):
            val = 1.0/12.0 + a_half[k]
            a_coeffs[k] = val
            a_coeffs[11-k] = val
            
        b_half = np.zeros(6)
        b_half[:5] = xb
        b_half[5] = -np.sum(xb)
        
        b_coeffs = [0.0] * 12
        for k in range(6):
            val = 1.0/12.0 + b_half[k]
            b_coeffs[k] = val
            b_coeffs[11-k] = val

        a_list = [max(-50.0, min(50.0, v)) for v in a_coeffs]
        b_list = [max(-50.0, min(50.0, v)) for v in b_coeffs]
        
        return _error_fast(a_list, b_list, A_eigvals, A_eigvecs, B_eigvals, B_eigvecs, U_exact_cache, times)

    params0 = np.zeros(10)
    
    remaining = deadline - (time.time() - start_time)
    if remaining > 5.0:
        best_params, best_f = _nelder_mead(
            _error_from_sym_params, params0,
            max_iter=1000, tol=1e-8,
            timeout=remaining - 5.0,
            start_time=start_time
        )
        
        xa = best_params[:5]
        xb = best_params[5:]
        
        a_half = np.zeros(6)
        a_half[:5] = xa
        a_half[5] = -np.sum(xa)
        a_final = [0.0]*12
        for k in range(6):
            val = 1.0/12.0 + a_half[k]
            a_final[k] = val
            a_final[11-k] = val
            
        b_half = np.zeros(6)
        b_half[:5] = xb
        b_half[5] = -np.sum(xb)
        b_final = [0.0]*12
        for k in range(6):
            val = 1.0/12.0 + b_half[k]
            b_final[k] = val
            b_final[11-k] = val
            
        err_opt = _error_fast(a_final, b_final, A_eigvals, A_eigvecs, B_eigvals, B_eigvecs, U_exact_cache, times)
        
        uniform_a = [1.0/12.0]*12
        uniform_b = [1.0/12.0]*12
        err_uniform = _error_fast(uniform_a, uniform_b, A_eigvals, A_eigvecs, B_eigvals, B_eigvecs, U_exact_cache, times)
        
        if err_opt < err_uniform:
            return _normalize_and_clip(a_final, b_final)
        else:
            return _normalize_and_clip(uniform_a, uniform_b)

    return _normalize_and_clip([1.0/12.0]*12, [1.0/12.0]*12)
# EVOLVE-BLOCK-END
