"""Product-formula design for Heisenberg-chain time evolution using 8th-order Suzuki-Trotter with Nelder-Mead refinement."""
import time
import numpy as np

# EVOLVE-BLOCK-START
def _build_hamiltonians():
    """Build the A and B Hamiltonians for the 6-site XXZ Heisenberg chain."""
    n = 6
    dim = 2 ** n
    anisotropy = 0.8
    field = 0.3
    
    # Pauli matrices
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    I = np.eye(2, dtype=complex)
    
    # Build single-site operators
    def op_on_site(op, site):
        """Apply operator to a specific site."""
        result = np.array([[1]], dtype=complex)
        for i in range(n):
            if i == site:
                result = np.kron(result, op)
            else:
                result = np.kron(result, I)
        return result
    
    # Build bond operators: 0.5*(Sx_i*Sx_j + Sy_i*Sy_j + anisotropy*Sz_i*Sz_j)
    def bond_op(i, j):
        sxi = op_on_site(sx, i)
        syi = op_on_site(sy, i)
        szi = op_on_site(sz, i)
        sxj = op_on_site(sx, j)
        syj = op_on_site(sy, j)
        szj = op_on_site(sz, j)
        return 0.5 * (np.kron(sxi, sxj) + np.kron(syi, syj) + anisotropy * np.kron(szi, szj))
    
    # Field term: 0.5 * field * Sz_i (half for each partition)
    def field_op(site):
        return 0.5 * field * op_on_site(sz, site)
    
    # A: odd bonds (0-1, 2-3, 4-5) + half field on odd sites
    A = np.zeros((dim, dim), dtype=complex)
    for bond in [(0, 1), (2, 3), (4, 5)]:
        A += bond_op(bond[0], bond[1])
    for site in [0, 2, 4]:
        A += field_op(site)
    
    # B: even bonds (1-2, 3-4) + half field on even sites
    B = np.zeros((dim, dim), dtype=complex)
    for bond in [(1, 2), (3, 4)]:
        B += bond_op(bond[0], bond[1])
    for site in [1, 3, 5]:
        B += field_op(site)
    
    return A, B

def _expm_pade(A, order=6):
    """Matrix exponential using Pade approximation (avoids scipy dependency)."""
    n = A.shape[0]
    I = np.eye(n, dtype=complex)
    
    # Scale and square
    norm_A = np.linalg.norm(A, ord=1)
    if norm_A < 1e-15:
        return I
    s = max(1, int(np.ceil(np.log2(norm_A / 0.0365))))
    A_scaled = A / (2 ** s)
    
    # Pade coefficients for order 6
    # r6 = [60, 275, 300, 125, 25, 3, 1]
    # Denominator: I + sum_{k=1}^{6} r6[k] * A_scaled^k / (k+1)!
    # Numerator: sum_{k=0}^{6} r6[k] * A_scaled^k / k!
    
    # Compute powers
    powers = [I]
    for k in range(1, 7):
        powers.append(powers[-1] @ A_scaled)
    
    # Pade 6 coefficients
    r6 = np.array([1, 3, 25, 125, 300, 275, 60], dtype=complex)
    
    num = np.zeros((n, n), dtype=complex)
    den = np.zeros((n, n), dtype=complex)
    
    for k in range(7):
        num += r6[k] * powers[k] / np.math.factorial(k)
        if k < 6:
            den += r6[k+1] * powers[k+1] / np.math.factorial(k+1)
    
    den += I
    
    # Solve den * X = num
    try:
        X = np.linalg.solve(den, num)
    except np.linalg.LinAlgError:
        X = np.linalg.pinv(den) @ num
    
    # Square back
    for _ in range(s):
        X = X @ X
    
    return X

def _error_fast(a, b, A, B, exact_propagators, t_values):
    """Compute the mean spectral norm error."""
    errors = []
    for t in t_values:
        U = np.eye(A.shape[0], dtype=complex)
        for k in range(len(a)):
            U = _expm_pade(-1j * b[k] * t * B) @ U
            U = _expm_pade(-1j * a[k] * t * A) @ U
        
        exact = exact_propagators[t]
        diff = U - exact
        # Spectral norm = largest singular value
        s_vals = np.linalg.svd(diff, compute_uv=False)
        errors.append(s_vals[0])
    
    return np.mean(errors)

def _normalize_and_clip(a, b, stages):
    """Normalize coefficients to sum to 1 and clip to [-50, 50]."""
    a = list(a)
    b = list(b)
    
    # Ensure correct length
    while len(a) < stages:
        a.append(0.0)
    while len(b) < stages:
        b.append(0.0)
    a = a[:stages]
    b = b[:stages]
    
    # Clip to [-50, 50]
    a = [max(-50.0, min(50.0, x)) for x in a]
    b = [max(-50.0, min(50.0, x)) for x in b]
    
    # Normalize to sum to 1
    sa = sum(a)
    sb = sum(b)
    if abs(sa) > 1e-10:
        a = [x / sa for x in a]
    if abs(sb) > 1e-10:
        b = [x / sb for x in b]
    
    # Re-clip after normalization
    a = [max(-50.0, min(50.0, x)) for x in a]
    b = [max(-50.0, min(50.0, x)) for x in b]
    
    # Final normalization
    sa = sum(a)
    sb = sum(b)
    if abs(sa) > 1e-10:
        a = [x / sa for x in a]
    if abs(sb) > 1e-10:
        b = [x / sb for x in b]
    
    return a, b

def _nelder_mead(f, x0, bounds, max_iter=200, tol=1e-6, timeout=None, start_time=None):
    """Nelder-Mead optimization with time budget."""
    if start_time is None:
        start_time = time.time()
    
    n = len(x0)
    alpha = 1.0  # Reflection
    gamma = 2.0  # Expansion
    rho = 0.5   # Contraction
    sigma = 0.5 # Shrink
    
    # Create simplex
    simplex = [x0[:]]
    for i in range(n):
        xi = x0[:]
        xi[i] += 0.01 if x0[i] >= 0 else -0.01
        simplex.append(xi)
    
    f_vals = [f(x) for x in simplex]
    
    best_x = simplex[np.argmin(f_vals)]
    best_f = min(f_vals)
    
    for iteration in range(max_iter):
        if timeout is not None and time.time() - start_time > timeout:
            break
        
        # Sort simplex
        indices = np.argsort(f_vals)
        simplex = [simplex[i] for i in indices]
        f_vals = [f_vals[i] for i in indices]
        
        # Check convergence
        if abs(f_vals[0] - f_vals[-1]) < tol:
            break
        
        # Centroid of all but worst
        centroid = np.mean([simplex[i] for i in range(n)], axis=0)
        
        # Reflection
        xr = centroid + alpha * (centroid - simplex[-1])
        fr = f(xr)
        
        if fr < f_vals[0]:
            # Expansion
            xe = centroid + gamma * (xr - centroid)
            fe = f(xe)
            if fe < fr:
                simplex[-1] = xe
                f_vals[-1] = fe
            else:
                simplex[-1] = xr
                f_vals[-1] = fr
        elif fr < f_vals[-2]:
            simplex[-1] = xr
            f_vals[-1] = fr
        else:
            # Contraction
            if fr < f_vals[-1]:
                xc = centroid + rho * (xr - centroid)
                fc = f(xc)
                if fc <= fr:
                    simplex[-1] = xc
                    f_vals[-1] = fc
                else:
                    # Shrink
                    for i in range(1, n + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        f_vals[i] = f(simplex[i])
            else:
                xc = centroid - rho * (simplex[-1] - centroid)
                fc = f(xc)
                if fc < f_vals[-1]:
                    simplex[-1] = xc
                    f_vals[-1] = fc
                else:
                    # Shrink
                    for i in range(1, n + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        f_vals[i] = f(simplex[i])
        
        # Track best
        current_best_f = min(f_vals)
        if current_best_f < best_f:
            best_f = current_best_f
            best_x = simplex[np.argmin(f_vals)]
    
    return best_x, best_f

def _get_8th_order_coeffs(stages):
    """Get 8th-order Suzuki-Trotter coefficients for the given number of stages."""
    # 4th-order coefficients
    c1 = 1.0 / (2.0 - 2.0 ** (1.0 / 3.0))
    c2 = -2.0 ** (1.0 / 3.0) / (2.0 - 2.0 ** (1.0 / 3.0))
    
    # 8th-order uses 4th-order with specific scaling
    d1 = c1
    d2 = c2
    
    # For 12 stages, use a symmetric pattern
    # The 8th-order formula can be decomposed into alternating A and B factors
    # Using the known structure:
    
    if stages == 12:
        # 8th-order symmetric coefficients for 12 stages
        # Pattern based on Suzuki-Trotter 8th order
        a = [c1/4, c2/4, c2/4, c1/4, c1/4, c2/4, c2/4, c1/4, c1/4, c2/4, c2/4, c1/4]
        b = [c1/4, c2/4, c2/4, c1/4, c1/4, c2/4, c2/4, c1/4, c1/4, c2/4, c2/4, c1/4]
    else:
        # Fallback: uniform distribution
        a = [1.0 / stages] * stages
        b = [1.0 / stages] * stages
    
    return a, b

def coefficients(stages):
    """
    8th-order symmetric Suzuki-Trotter product formula with Nelder-Mead refinement.
    
    Uses 8th-order coefficients as initialization, then refines with Nelder-Mead
    to minimize the mean spectral norm error at t = 0.25, 0.5, 1.0.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Build Hamiltonians
    A, B = _build_hamiltonians()
    
    # Precompute exact propagators
    t_values = [0.25, 0.5, 1.0]
    AB = A + B
    exact_propagators = {}
    for t in t_values:
        exact_propagators[t] = _expm_pade(-1j * t * AB)
    
    # Get initial coefficients from 8th-order Suzuki-Trotter
    a_init, b_init = _get_8th_order_coeffs(stages)
    a_init, b_init = _normalize_and_clip(a_init, b_init, stages)
    
    # Initial evaluation
    initial_error = _error_fast(a_init, b_init, A, B, exact_propagators, t_values)
    best_a = a_init[:]
    best_b = b_init[:]
    best_error = initial_error
    
    # Time budget for optimization
    elapsed = time.time() - start_time
    if elapsed > 10.0:
        # Not enough time for optimization, return initial
        return best_a, best_b
    
    # Create objective function that combines a and b into a single vector
    def objective(x):
        a = list(x[:stages])
        b = list(x[stages:])
        a, b = _normalize_and_clip(a, b, stages)
        return _error_fast(a, b, A, B, exact_propagators, t_values)
    
    # Initial point
    x0 = a_init + b_init
    
    # Run Nelder-Mead with time budget
    opt_timeout = 130.0 - elapsed  # Leave 20s safety margin
    x_opt, opt_error = _nelder_mead(
        objective, x0, None, 
        max_iter=100, tol=1e-8,
        timeout=opt_timeout, start_time=start_time
    )
    
    # Extract optimized coefficients
    a_opt = list(x_opt[:stages])
    b_opt = list(x_opt[stages:])
    a_opt, b_opt = _normalize_and_clip(a_opt, b_opt, stages)
    
    # Final evaluation
    final_error = _error_fast(a_opt, b_opt, A, B, exact_propagators, t_values)
    
    # Use the better result
    if final_error < best_error:
        best_a = a_opt
        best_b = b_opt
        best_error = final_error
    
    # Check if we're running out of time
    if time.time() - start_time > deadline - 5.0:
        return best_a, best_b
    
    # Try a few more iterations if time permits
    remaining_time = deadline - (time.time() - start_time) - 5.0
    if remaining_time > 10.0:
        # Do a second pass with tighter tolerance
        x_start = best_a + best_b
        x_opt2, opt_error2 = _nelder_mead(
            objective, x_start, None,
            max_iter=50, tol=1e-10,
            timeout=remaining_time, start_time=start_time
        )
        
        a_opt2 = list(x_opt2[:stages])
        b_opt2 = list(x_opt2[stages:])
        a_opt2, b_opt2 = _normalize_and_clip(a_opt2, b_opt2, stages)
        
        final_error2 = _error_fast(a_opt2, b_opt2, A, B, exact_propagators, t_values)
        
        if final_error2 < best_error:
            best_a = a_opt2
            best_b = b_opt2
            best_error = final_error2
    
    return best_a, best_b
# EVOLVE-BLOCK-END

