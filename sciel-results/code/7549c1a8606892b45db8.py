"""Product-formula design for Heisenberg-chain time evolution using direct numerical optimization."""
import time
import numpy as np

_DEADLINE = 140.0  # safety margin before 150s kill


def _build_hamiltonians():
    """Build A (odd bonds) and B (even bonds) for 6-site open XXZ chain.
    Anisotropy gamma=0.8, field h=0.3.
    H = sum_j [0.5*(Sx_j Sx_{j+1} + Sy_j Sy_{j+1} + gamma Sz_j Sz_{j+1}) - h Sz_j]
    Split into even bonds (j=0,2,4) -> B, odd bonds (j=1,3) -> A.
    Each part includes half the field.
    """
    N = 6
    gamma = 0.8
    h = 0.3
    dim = 2 ** N
    
    # Pauli matrices
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    I2 = np.eye(2, dtype=complex)
    
    def op_on_site(op, site):
        """Apply op on a specific site (0-indexed from left)."""
        result = op
        for s in range(N - 1, -1, -1):
            if s != site:
                result = np.kron(result, I2)
            else:
                result = np.kron(result, op)
        return result
    
    def bond_hamiltonian(j):
        """Bond Hamiltonian between sites j and j+1."""
        sxj = op_on_site(sx, j)
        sxj1 = op_on_site(sx, j + 1)
        syj = op_on_site(sy, j)
        syj1 = op_on_site(sy, j + 1)
        szj = op_on_site(sz, j)
        szj1 = op_on_site(sz, j + 1)
        return 0.5 * (sxj @ sxj1 + syj @ syj1 + gamma * szj @ szj1)
    
    # Field term (split equally)
    field = np.zeros((dim, dim), dtype=complex)
    for j in range(N):
        szj = op_on_site(sz, j)
        field += -h * szj
    
    # A: odd bonds (j=1, 3) + half field
    A = np.zeros((dim, dim), dtype=complex)
    for j in [1, 3]:
        A += bond_hamiltonian(j)
    A += 0.5 * field
    
    # B: even bonds (j=0, 2, 4) + half field
    B = np.zeros((dim, dim), dtype=complex)
    for j in [0, 2, 4]:
        B += bond_hamiltonian(j)
    B += 0.5 * field
    
    return A, B


def _mat_exp(M):
    """Matrix exponential."""
    from scipy.linalg import expm
    return expm(M)


def _compute_error(a, b, A, B, target_times):
    """Compute mean spectral norm error at target times."""
    total_err = 0.0
    for t in target_times:
        U = np.eye(A.shape[0], dtype=complex)
        for k in range(len(a)):
            U = _mat_exp(-1j * b[k] * t * B) @ U
            U = _mat_exp(-1j * a[k] * t * A) @ U
        U_exact = _mat_exp(-1j * t * (A + B))
        err = np.linalg.norm(U - U_exact, 2)
        total_err += err
    return total_err / len(target_times)


def _normalize_and_clip(a, b, stages):
    """Ensure sums are 1 and values in [-50, 50]."""
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    a = np.clip(a, -50.0, 50.0)
    b = np.clip(b, -50.0, 50.0)
    # Normalize
    sa = np.sum(a)
    sb = np.sum(b)
    if abs(sa) > 1e-12:
        a = a / sa
    else:
        a = np.ones(stages) / stages
    if abs(sb) > 1e-12:
        b = b / sb
    else:
        b = np.ones(stages) / stages
    # Re-clip after normalization
    a = np.clip(a, -50.0, 50.0)
    b = np.clip(b, -50.0, 50.0)
    # Final normalization to ensure exact sum=1
    sa = np.sum(a)
    sb = np.sum(b)
    if abs(sa) > 1e-12:
        a = a / sa
    if abs(sb) > 1e-12:
        b = b / sb
    return a, b


def _nelder_mead(f, x0, bounds, max_iter=500, tol=1e-8, deadline=None):
    """Nelder-Mead optimization with bounds and deadline."""
    n = len(x0)
    x0 = np.array(x0, dtype=float)
    
    # Build simplex
    simplex = [x0.copy()]
    for i in range(n):
        xi = x0.copy()
        step = 0.1 if x0[i] != 0 else 0.1
        xi[i] += step
        simplex.append(xi)
    
    # Evaluate initial simplex
    fx = [f(x) for x in simplex]
    
    alpha = 1.0
    gamma = 2.0
    rho = 0.5
    sigma = 0.5
    
    for iteration in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Sort by function value
        order = np.argsort(fx)
        simplex = [simplex[i] for i in order]
        fx = [fx[i] for i in order]
        
        # Check convergence
        if abs(fx[-1] - fx[0]) < tol:
            break
        
        # Centroid of all but worst point
        centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflection
        xr = centroid + alpha * (centroid - simplex[-1])
        fr = f(xr)
        
        if fr < fx[0]:
            # Expansion
            xe = centroid + gamma * (xr - centroid)
            fe = f(xe)
            if fe < fr:
                simplex[-1] = xe
                fx[-1] = fe
            else:
                simplex[-1] = xr
                fx[-1] = fr
        elif fr < fx[-2]:
            simplex[-1] = xr
            fx[-1] = fr
        else:
            # Contraction
            if fr < fx[-1]:
                xc = centroid + rho * (xr - centroid)
            else:
                xc = centroid - rho * (simplex[-1] - centroid)
            fc = f(xc)
            if fc < min(fr, fx[-1]):
                simplex[-1] = xc
                fx[-1] = fc
            else:
                # Shrink
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    fx[i] = f(simplex[i])
    
    # Return best point
    best_idx = np.argmin(fx)
    return simplex[best_idx], fx[best_idx]


def coefficients(stages):
    """
    Direct numerical optimization of product formula coefficients.
    Minimizes mean spectral norm error at t=0.25, 0.5, 1.0.
    """
    start_time = time.time()
    deadline = start_time + _DEADLINE
    
    A, B = _build_hamiltonians()
    target_times = [0.25, 0.5, 1.0]
    
    # Initial guess: uniform 4th-order-like coefficients
    c = 2 ** (1.0 / 3.0)
    denom = 2.0 - c
    c1 = 1.0 / denom
    c2 = -c / denom
    
    # Start with a reasonable symmetric pattern
    a0 = np.zeros(stages)
    b0 = np.zeros(stages)
    
    # Spread the 4th-order coefficients across stages
    # 4th order has 4 A terms and 3 B terms per step
    # We'll create a symmetric pattern
    n_sub = stages // 4  # number of substeps
    if n_sub < 1:
        n_sub = 1
    
    # Simple symmetric initialization
    a0[:] = 1.0 / stages
    b0[:] = 1.0 / stages
    
    # Create a more structured initial guess based on 4th-order
    # Each 4th-order step: A: [c1/2, (c1+c2)/2, (c1+c2)/2, c1/2], B: [c1, c2, c1]
    # For 12 stages, use 3 substeps
    if stages == 12:
        a0 = np.array([
            c1/6.0, (c1+c2)/6.0, (c1+c2)/6.0, c1/6.0,
            c1/6.0, (c1+c2)/6.0, (c1+c2)/6.0, c1/6.0,
            c1/6.0, (c1+c2)/6.0, (c1+c2)/6.0, c1/6.0
        ])
        b0 = np.array([
            c1/3.0, c2/3.0, c1/3.0,
            c1/3.0, c2/3.0, c1/3.0,
            c1/3.0, c2/3.0, c1/3.0,
            0.0, 0.0, 0.0
        ])
        # Redistribute B to have 12 entries
        b0 = np.array([
            c1/9.0, c2/9.0, c1/9.0,
            c1/9.0, c2/9.0, c1/9.0,
            c1/9.0, c2/9.0, c1/9.0,
            c1/9.0, c2/9.0, c1/9.0
        ])
    
    a0, b0 = _normalize_and_clip(a0, b0, stages)
    
    # Combine into single vector for optimization
    x0 = np.concatenate([a0, b0])
    
    def objective(x):
        if time.time() > deadline:
            return 1e10
        a = x[:stages]
        b = x[stages:]
        a, b = _normalize_and_clip(a, b, stages)
        return _compute_error(a, b, A, B, target_times)
    
    # Run Nelder-Mead
    best_x = x0.copy()
    best_f = objective(x0)
    
    # Multiple restarts with different initializations
    restarts = 5
    for r in range(restarts):
        if time.time() > deadline - 20:
            break
        
        if r == 0:
            x_start = x0.copy()
        else:
            # Perturb initial guess
            noise_scale = 0.1 * (r + 1)
            x_start = x0 + noise_scale * np.random.randn(len(x0))
        
        x_opt, f_opt = _nelder_mead(
            objective, x_start, 
            max_iter=300, tol=1e-8,
            deadline=deadline - 10
        )
        
        if f_opt < best_f:
            best_f = f_opt
            best_x = x_opt
    
    # Extract final coefficients
    a_opt = best_x[:stages]
    b_opt = best_x[stages:]
    a_opt, b_opt = _normalize_and_clip(a_opt, b_opt, stages)
    
    # Final safety check
    if time.time() > deadline:
        # Return the initial guess if we're running out of time
        a_opt, b_opt = _normalize_and_clip(a0, b0, stages)
    
    return list(a_opt), list(b_opt)
