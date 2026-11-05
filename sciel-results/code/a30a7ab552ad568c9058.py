"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_DEADLINE = time.time() + 140.0  # safety margin before 150s wall-clock kill

def _build_AB():
    """Build A (odd bonds) and B (even bonds) for 6-site open XXZ chain."""
    n = 6
    aniso = 0.8
    field = 0.3
    dim = 2 ** n
    # Build full Hamiltonian H = A + B
    # XXZ: sum_j [ (1/2)(Sx_j Sx_{j+1} + Sy_j Sy_{j+1} + aniso Sz_j Sz_{j+1}) ] + field * sum_j Sz_j
    # Sx = sigma_x/2, Sy = sigma_y/2, Sz = sigma_z/2
    sx = np.array([[0, 1], [1, 0]], dtype=complex) / 2
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex) / 2
    sz = np.array([[1, 0], [0, -1]], dtype=complex) / 2
    I2 = np.eye(2, dtype=complex)
    
    def op_on_site(op, site):
        """Apply operator op on given site (0-indexed from left)."""
        ops = [I2] * n
        ops[site] = op
        result = ops[0]
        for op in ops[1:]:
            result = np.kron(result, op)
        return result
    
    H = np.zeros((dim, dim), dtype=complex)
    for j in range(n - 1):
        sxx = op_on_site(sx, j) @ op_on_site(sx, j + 1)
        syy = op_on_site(sy, j) @ op_on_site(sy, j + 1)
        szz = op_on_site(sz, j) @ op_on_site(sz, j + 1)
        bond = 0.5 * (sxx + syy + aniso * szz)
        H += bond
    
    # Uniform field: split half to A, half to B
    field_term = np.zeros((dim, dim), dtype=complex)
    for j in range(n):
        field_term += field * op_on_site(sz, j)
    
    # A: odd bonds (j=0,2,4) + half field
    # B: even bonds (j=1,3,5) + half field
    A = np.zeros((dim, dim), dtype=complex)
    B = np.zeros((dim, dim), dtype=complex)
    for j in range(n - 1):
        sxx = op_on_site(sx, j) @ op_on_site(sx, j + 1)
        syy = op_on_site(sy, j) @ op_on_site(sy, j + 1)
        szz = op_on_site(sz, j) @ op_on_site(sz, j + 1)
        bond = 0.5 * (sxx + syy + aniso * szz)
        if j % 2 == 0:  # odd bonds: j=0,2,4
            A += bond
        else:  # even bonds: j=1,3,5
            B += bond
    A += 0.5 * field_term
    B += 0.5 * field_term
    return A, B

def _fourier_to_coeffs(params, stages=12):
    """Convert Fourier parameters to a, b coefficient lists with sum=1."""
    # params: [a0, a1, a2, b0, b1, b2] (low-frequency terms)
    k = np.arange(stages)
    # a_k = a0 + a1*cos(2*pi*k/stages) + a2*sin(2*pi*k/stages)
    a_raw = params[0] + params[1] * np.cos(2 * np.pi * k / stages) + params[2] * np.sin(2 * np.pi * k / stages)
    # b_k = b0 + b1*cos(2*pi*k/stages) + b2*sin(2*pi*k/stages)
    b_raw = params[3] + params[4] * np.cos(2 * np.pi * k / stages) + params[5] * np.sin(2 * np.pi * k / stages)
    # Normalize to sum=1
    a_sum = np.sum(a_raw)
    b_sum = np.sum(b_raw)
    if abs(a_sum) > 1e-12:
        a = a_raw / a_sum
    else:
        a = np.ones(stages) / stages
    if abs(b_sum) > 1e-12:
        b = b_raw / b_sum
    else:
        b = np.ones(stages) / stages
    # Clip to [-50, 50] and re-normalize
    a = np.clip(a, -50, 50)
    b = np.clip(b, -50, 50)
    a = a / np.sum(a)
    b = b / np.sum(b)
    return a, b

def _objective(params, A, B, times, exact_propagators):
    """Compute mean spectral norm error."""
    a, b = _fourier_to_coeffs(params)
    total_err = 0.0
    for idx, t in enumerate(times):
        # Build product formula U(t) = prod_k exp(-i b_k t B) exp(-i a_k t A)
        U = np.eye(A.shape[0], dtype=complex)
        for k in range(len(a)):
            if abs(b[k]) > 1e-15:
                U = U @ np.linalg.matrix_exp(-1j * b[k] * t * B)
            if abs(a[k]) > 1e-15:
                U = U @ np.linalg.matrix_exp(-1j * a[k] * t * A)
        err = np.linalg.norm(U - exact_propagators[idx], 2)
        total_err += err
    return total_err / len(times)

def _nelder_mead(f, x0, bounds=None, max_iter=200, tol=1e-8):
    """Simple Nelder-Mead optimizer."""
    n = len(x0)
    # Initialize simplex
    simplex = [np.array(x0, dtype=float)]
    f0 = f(simplex[0])
    simplex.append(simplex[0].copy() + 0.1)
    for i in range(1, n):
        simplex.append(simplex[0].copy())
        simplex[i][i] += 0.1
    fvals = [f0]
    for i in range(1, n + 1):
        fvals.append(f(simplex[i]))
    
    for _ in range(max_iter):
        # Sort
        order = np.argsort(fvals)
        simplex = [simplex[i] for i in order]
        fvals = [fvals[i] for i in order]
        
        if abs(fvals[-1] - fvals[0]) < tol:
            break
        
        # Centroid of all but worst
        centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflection
        xr = centroid + 1.0 * (centroid - simplex[-1])
        fr = f(xr)
        
        if fr < fvals[0]:
            # Expansion
            xe = centroid + 2.0 * (centroid - simplex[-1])
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
            xc = centroid + 0.5 * (simplex[-1] - centroid)
            fc = f(xc)
            if fc < fvals[-1]:
                simplex[-1] = xc
                fvals[-1] = fc
            else:
                # Shrink
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                    fvals[i] = f(simplex[i])
    
    order = np.argsort(fvals)
    return simplex[order[0]], fvals[order[0]]

def coefficients(stages):
    """Fourier-parameterized product formula with local optimization."""
    global _DEADLINE
    stages = 12
    t_start = time.time()
    
    # Build Hamiltonian parts
    A, B = _build_AB()
    times = [0.25, 0.5, 1.0]
    exact_propagators = [np.linalg.matrix_exp(-1j * t * (A + B)) for t in times]
    
    # Initial guess: uniform coefficients
    x0 = np.array([1.0/stages, 0.0, 0.0, 1.0/stages, 0.0, 0.0])
    
    # Run Nelder-Mead
    best_x = x0.copy()
    best_f = _objective(x0, A, B, times, exact_propagators)
    
    # Try a few different initial perturbations
    rng = np.random.RandomState(42)
    n_starts = 5
    for i in range(n_starts):
        if time.time() > _DEADLINE - 10:
            break
        if i == 0:
            x_try = x0.copy()
        else:
            x_try = x0 + rng.randn(6) * 0.05
        x_opt, f_opt = _nelder_mead(
            lambda p: _objective(p, A, B, times, exact_propagators),
            x_try,
            max_iter=150,
            tol=1e-9
        )
        if f_opt < best_f:
            best_f = f_opt
            best_x = x_opt
    
    a, b = _fourier_to_coeffs(best_x, stages)
    # Ensure finite and in range
    a = np.clip(np.nan_to_num(a), -50, 50)
    b = np.clip(np.nan_to_num(b), -50, 50)
    # Final normalization
    a = a / np.sum(a)
    b = b / np.sum(b)
    return list(a), list(b)
# EVOLVE-BLOCK-END
