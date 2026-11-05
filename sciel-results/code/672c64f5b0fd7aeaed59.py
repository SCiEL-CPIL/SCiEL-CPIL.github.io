"""High-order product formula design for Heisenberg-chain time evolution."""
import numpy as np
import time

# EVOLVE-BLOCK-START
def _build_heisenberg_matrices():
    """Build A (odd bonds) and B (even bonds) for 6-site open XXZ Heisenberg chain."""
    n = 6
    dim = 2 ** n
    Sx = np.array([[0, 0.5], [0.5, 0]], dtype=complex)
    Sy = np.array([[0, -0.5j], [0.5j, 0]], dtype=complex)
    Sz = np.array([[0.5, 0], [0, -0.5]], dtype=complex)
    
    def kron_all(ops):
        result = np.array([[1]], dtype=complex)
        for op in ops:
            result = np.kron(result, op)
        return result
    
    anisotropy = 0.8
    field = 0.3
    
    A = np.zeros((dim, dim), dtype=complex)
    B = np.zeros((dim, dim), dtype=complex)
    
    for i in range(n):
        ops = [np.eye(2)] * n
        ops[i] = Sx
        Sx_i = kron_all(ops)
        ops[i] = Sy
        Sy_i = kron_all(ops)
        ops[i] = Sz
        Sz_i = kron_all(ops)
        
        A -= 0.5 * field * Sz_i
        B -= 0.5 * field * Sz_i
    
    for i in [0, 2, 4]:
        j = i + 1
        ops = [np.eye(2)] * n
        ops[i] = Sx
        ops[j] = Sx
        Sx_i_Sx_j = kron_all(ops)
        
        ops[i] = Sy
        ops[j] = Sy
        Sy_i_Sy_j = kron_all(ops)
        
        ops[i] = Sz
        ops[j] = Sz
        Sz_i_Sz_j = kron_all(ops)
        
        A += Sx_i_Sx_j + Sy_i_Sy_j + anisotropy * Sz_i_Sz_j
    
    for i in [1, 3]:
        j = i + 1
        ops = [np.eye(2)] * n
        ops[i] = Sx
        ops[j] = Sx
        Sx_i_Sx_j = kron_all(ops)
        
        ops[i] = Sy
        ops[j] = Sy
        Sy_i_Sy_j = kron_all(ops)
        
        ops[i] = Sz
        ops[j] = Sz
        Sz_i_Sz_j = kron_all(ops)
        
        B += Sx_i_Sx_j + Sy_i_Sy_j + anisotropy * Sz_i_Sz_j
    
    return A, B


def _spectral_norm(M):
    """Compute spectral norm (largest singular value) of matrix M."""
    try:
        s = np.linalg.svd(M, compute_uv=False)
        return s[0] if len(s) > 0 else 0.0
    except:
        try:
            return np.linalg.norm(M, 2)
        except:
            return 1e10


def _fourier_to_coeffs(params, n_stages=12, n_harmonics=3):
    """Convert Fourier parameters to symmetric coefficient arrays.
    
    params: [a0, a1, a2, a3, b0, b1, b2, b3]
    a0, b0: DC offsets (ensuring sum=1 after normalization)
    a1..a3, b1..b3: cosine coefficients for harmonics 1..3
    
    Produces symmetric patterns: c[k] = c[n-1-k]
    """
    n = n_stages
    k = np.arange(n)
    
    # For symmetric sequences: c[k] = c0 + sum_m cm * cos(m * pi * k / (n-1))
    # This ensures c[k] = c[n-1-k] since cos is even
    
    a = np.zeros(n)
    b = np.zeros(n)
    
    a0 = params[0]
    b0 = params[4]
    
    for m in range(1, n_harmonics + 1):
        a_m = params[m]
        b_m = params[4 + m]
        a += a_m * np.cos(m * np.pi * k / (n - 1))
        b += b_m * np.cos(m * np.pi * k / (n - 1))
    
    a += a0
    b += b0
    
    # Normalize to sum to 1
    sa = np.sum(a)
    sb = np.sum(b)
    if abs(sa) > 1e-12:
        a = a / sa
    else:
        a = np.ones(n) / n
    if abs(sb) > 1e-12:
        b = b / sb
    else:
        b = np.ones(n) / n
    
    # Clip to valid range
    a = np.clip(a, -50, 50)
    b = np.clip(b, -50, 50)
    
    # Re-normalize after clipping
    sa = np.sum(a)
    sb = np.sum(b)
    if abs(sa) > 1e-12:
        a = a / sa
    if abs(sb) > 1e-12:
        b = b / sb
    
    return a, b


def _objective_from_fourier(params, A, B, times):
    """Objective function for Fourier-parameterized coefficients."""
    a, b = _fourier_to_coeffs(params)
    
    # Verify constraints
    if abs(np.sum(a) - 1.0) > 1e-6 or abs(np.sum(b) - 1.0) > 1e-6:
        return 1e10
    
    if np.any(np.abs(a) > 50) or np.any(np.abs(b) > 50):
        return 1e10
    
    errors = []
    for t in times:
        U = np.eye(64, dtype=complex)
        for k in range(12):
            try:
                eB = np.linalg.expm(-1j * b[k] * t * B)
            except:
                return 1e10
            U = eB @ U
            try:
                eA = np.linalg.expm(-1j * a[k] * t * A)
            except:
                return 1e10
            U = eA @ U
        
        try:
            U_exact = np.linalg.expm(-1j * t * (A + B))
        except:
            return 1e10
        
        err = _spectral_norm(U - U_exact)
        errors.append(err)
    
    return np.mean(errors)


def _objective(coeffs, A, B, times):
    """Compute mean spectral norm error for given coefficients."""
    a, b = coeffs
    n = len(a)
    if n != 12:
        return 1e10
    if abs(sum(a) - 1.0) > 1e-6 or abs(sum(b) - 1.0) > 1e-6:
        return 1e10 + abs(sum(a) - 1.0) + abs(sum(b) - 1.0)
    
    errors = []
    for t in times:
        U = np.eye(2 ** 6, dtype=complex)
        for k in range(n):
            try:
                eB = np.linalg.expm(-1j * b[k] * t * B)
            except:
                return 1e10
            U = eB @ U
            try:
                eA = np.linalg.expm(-1j * a[k] * t * A)
            except:
                return 1e10
            U = eA @ U
        
        try:
            U_exact = np.linalg.expm(-1j * t * (A + B))
        except:
            return 1e10
        
        err = _spectral_norm(U - U_exact)
        errors.append(err)
    
    return np.mean(errors)


def _nelder_mead(f, x0, bounds, max_iter=200, tol=1e-8):
    """Pure NumPy Nelder-Mead implementation."""
    n = len(x0)
    x = np.array(x0, dtype=float)
    f0 = f(x)
    
    simplex = [x.copy()]
    for i in range(n):
        xi = x.copy()
        xi[i] += 0.1 if x[i] != 0 else 0.1
        simplex.append(xi)
    
    fvals = [f(simplex[i]) for i in range(n + 1)]
    for i in range(n + 1):
        for j in range(i + 1, n + 1):
            if fvals[j] < fvals[i]:
                simplex[i], simplex[j] = simplex[j], simplex[i]
                fvals[i], fvals[j] = fvals[j], fvals[i]
    
    alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
    
    for it in range(max_iter):
        if max(fvals) - min(fvals) < tol:
            break
        
        centroid = np.mean(simplex[:n], axis=0)
        
        xr = centroid + alpha * (centroid - simplex[-1])
        fr = f(xr)
        
        if fr < fvals[0]:
            xe = centroid + gamma * (xr - centroid)
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
            xc = centroid + rho * (simplex[-1] - centroid)
            fc = f(xc)
            if fc < fvals[-1]:
                simplex[-1] = xc
                fvals[-1] = fc
            else:
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    fvals[i] = f(simplex[i])
        
        for i in range(n + 1):
            for j in range(i + 1, n + 1):
                if fvals[j] < fvals[i]:
                    simplex[i], simplex[j] = simplex[j], simplex[i]
                    fvals[i], fvals[j] = fvals[j], fvals[i]
    
    return simplex[0], fvals[0]


def coefficients(stages):
    """High-order RKT splitting with Fourier-parameterized optimization."""
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    A, B = _build_heisenberg_matrices()
    times = [0.25, 0.5, 1.0]
    
    best_coeffs = None
    best_val = 1e10
    
    # Strategy 1: Fourier-series parameterization (8 params instead of 24)
    # params = [a0, a1, a2, a3, b0, b1, b2, b3]
    # Start with uniform distribution (all harmonics zero, DC offset gives 1/12 each)
    
    # Try multiple initial points for Fourier optimization
    fourier_inits = [
        # Uniform: all params zero (DC will normalize to 1/12)
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        # Slight center weighting (negative harmonic 1 means peaks in middle)
        np.array([0.0, -0.1, 0.0, 0.0, 0.0, -0.1, 0.0, 0.0]),
        # RKT-like pattern: alternating signs
        np.array([0.0, 0.05, -0.03, 0.01, 0.0, 0.05, -0.03, 0.01]),
        # More aggressive center weighting
        np.array([0.0, -0.2, 0.05, 0.0, 0.0, -0.2, 0.05, 0.0]),
    ]
    
    for init in fourier_inits:
        if time.time() > deadline - 20:
            break
        
        try:
            x_opt, f_opt = _nelder_mead(
                lambda x: _objective_from_fourier(x, A, B, times),
                init, None, max_iter=80, tol=1e-7
            )
            if f_opt < best_val:
                best_val = f_opt
                a_opt, b_opt = _fourier_to_coeffs(x_opt)
                best_coeffs = (a_opt.tolist(), b_opt.tolist())
        except:
            pass
    
    # Strategy 2: Direct optimization from good starting points
    if time.time() < deadline - 25:
        # Use the best Fourier result as a starting point for direct optimization
        if best_coeffs is not None:
            a_start, b_start = best_coeffs
            coeffs_start = np.array(a_start + b_start, dtype=float)
            
            try:
                x_opt, f_opt = _nelder_mead(
                    lambda x: _objective((x[:12].tolist(), x[12:24].tolist()), A, B, times),
                    coeffs_start, None, max_iter=60, tol=1e-7
                )
                if f_opt < best_val:
                    best_val = f_opt
                    best_coeffs = (x_opt[:12].tolist(), x_opt[12:24].tolist())
            except:
                pass
        
        # Also try from RKT-inspired starting point
        c = 2.0 ** (1.0/6.0)
        c1 = 1.0 / (2.0 * (1.0 - 1.0/c))
        c2 = -c1 / c
        c3 = c1 / (c * c)
        c4 = -c3 / c
        c5 = c4 / c
        c6 = c5 / c
        
        base = [c1, c2, c3, c4, c5, c6]
        a_rkt = base + base[::-1]
        b_rkt = base + base[::-1]
        sa = sum(a_rkt)
        sb = sum(b_rkt)
        if abs(sa) > 1e-10:
            a_rkt = [x/sa for x in a_rkt]
        if abs(sb) > 1e-10:
            b_rkt = [x/sb for x in b_rkt]
        
        if time.time() < deadline - 15:
            coeffs_rkt = np.array(a_rkt + b_rkt, dtype=float)
            try:
                x_opt, f_opt = _nelder_mead(
                    lambda x: _objective((x[:12].tolist(), x[12:24].tolist()), A, B, times),
                    coeffs_rkt, None, max_iter=50, tol=1e-7
                )
                if f_opt < best_val:
                    best_val = f_opt
                    best_coeffs = (x_opt[:12].tolist(), x_opt[12:24].tolist())
            except:
                pass
    
    # Fallback: uniform coefficients
    if best_coeffs is None:
        a = [1.0/12.0] * 12
        b = [1.0/12.0] * 12
        return a, b
    
    a_opt, b_opt = best_coeffs
    
    # Final validation and normalization
    sa = sum(a_opt)
    sb = sum(b_opt)
    if abs(sa) > 1e-12:
        a_opt = [x / sa for x in a_opt]
    else:
        a_opt = [1.0/12.0] * 12
    if abs(sb) > 1e-12:
        b_opt = [x / sb for x in b_opt]
    else:
        b_opt = [1.0/12.0] * 12
    
    # Clip to valid range
    a_opt = [max(-50, min(50, x)) for x in a_opt]
    b_opt = [max(-50, min(50, x)) for x in b_opt]
    
    # Re-normalize after clipping
    sa = sum(a_opt)
    sb = sum(b_opt)
    if abs(sa) > 1e-12:
        a_opt = [x / sa for x in a_opt]
    if abs(sb) > 1e-12:
        b_opt = [x / sb for x in b_opt]
    
    return a_opt, b_opt
# EVOLVE-BLOCK-END
