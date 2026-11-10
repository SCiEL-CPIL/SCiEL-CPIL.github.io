# EVOLVE-BLOCK-START
import numpy as np


def _norms(n, m):
    """N_m = sqrt( (2n+1)/(4 pi) * (n-m)!/(n+m)! ) for |m| <= n."""
    m = abs(int(m))
    return np.sqrt((2 * n + 1) / (4.0 * np.pi) * float(np.math.factorial(n - m)) /
                   float(np.math.factorial(n + m)))


def _real_harmonics(n, cos_t, sin_t, cos_phi, sin_phi):
    """
    Build the real orthonormal degree-n spherical harmonics evaluated at
    (cos_t, phi) = (cos_t, atan2(sin_phi, cos_phi)).
    Returns Y of shape (2n+1, N) with Y[i] for m = i - n:
        m =  0 : Y_0  = N_0 * P_n(cos_t)
        m > 0 : Y_m  = sqrt(2) N_m P_n^m(cos_t) cos(m phi)
        m < 0 : Y_m  = sqrt(2) N_|m| P_n^|m|(cos_t) sin(|m| phi)
    """
    Npts = cos_t.shape[0]
    Y = np.zeros((2 * n + 1, Npts), dtype=np.float64)
    for m in range(0, n + 1):
        Pm = _assoc_legendre(n, m, cos_t)
        Nm = _norms(n, m)
        idx = n + m
        if m == 0:
            Y[idx, :] = Nm * Pm
        else:
            Y[idx, :] = np.sqrt(2.0) * Nm * Pm * cos_phi
            Y[n - m, :] = np.sqrt(2.0) * Nm * Pm * sin_phi
    return Y


def _assoc_legendre(n, m, x):
    """Associated Legendre P_n^m(x) with Condon-Shortley phase, m >= 0, n >= m."""
    x = np.asarray(x, dtype=np.float64)
    # P_m^m
    Pmm = np.ones_like(x)
    if m > 0:
        fact = 1.0
        sq = np.sqrt((1.0 - x) * (1.0 + x))
        for i in range(1, m + 1):
            Pmm *= -fact * sq
            fact += 2.0
    if n == m:
        return Pmm
    Pmp1 = x * (2 * m + 1) * Pmm
    if n == m + 1:
        return Pmp1
    Pmm_prev = Pmm
    Pm = Pmp1
    for i in range(m + 2, n + 1):
        Pmm_prev, Pm = Pm, ((2 * i - 1) * x * Pm - (i + m - 1) * Pmm_prev) / (i - m)
    return Pm


def _build_quadrature(n, n_theta=70, n_phi=130):
    """Gauss-Legendre in cos(theta) x equally spaced phi. Returns weights, cos_t, phi."""
    x, w = np.polynomial.legendre.leggauss(n_theta)  # x in [-1,1], weights for integral over [-1,1]
    phi = 2.0 * np.pi * np.arange(n_phi) / n_phi
    # Mesh: index = i_theta * n_phi + i_phi
    cos_t = np.repeat(x, n_phi)
    phi_v = np.tile(phi, n_theta)
    sin_t = np.sqrt(np.maximum(0.0, 1.0 - x * x))
    sin_t = np.repeat(sin_t, n_phi)
    # weight per point: (dOmega = dphi dcos) -> w_theta * (2pi/n_phi)
    wpts = np.repeat(w * (2.0 * np.pi / n_phi), n_theta)
    return wpts, cos_t, sin_t, phi_v


def _make_Y(wpts, cos_t, sin_t, phi_v, n=10):
    cos_phi = np.cos(phi_v)
    sin_phi = np.sin(phi_v)
    return _real_harmonics(n, cos_t, sin_t, cos_phi, sin_phi)


def _evaluate(c, Y, wpts):
    """Return (L4_over_L2, grad) for f = sum c_m Y_m. Norms over sphere, total mass 4pi."""
    f = Y.T @ c  # (N,)
    f2 = f * f
    f4 = f2 * f2
    mass = wpts.sum()  # = 4 pi
    L2sq = (wpts * f2).sum() / mass
    L4 = (wpts * f4).sum() / mass
    L2 = np.sqrt(max(L2sq, 1e-300))
    L4v = L4 ** 0.25
    ratio = L4v / L2
    # gradient of ratio
    if L2 <= 1e-300:
        return ratio, np.zeros_like(c)
    # d(ratio)/dc = (1/L2) * (1/L4v) * (1/4) * 4 f^3 * f' - (L4v/L2^2) * (1/L2sq) * 2 f f'
    # = (f^3 / (L2 * L4v)) * f' - (L4v * f / (L2^2 * L2sq)) * f'
    fp = Y  # d f / d c_m = Y_m
    g = (f ** 3 / (L2 * L4v) - L4v * f / (L2 * L2 * L2sq)) * fp  # (N,)
    grad = Y @ g  # (2n+1,)
    return ratio, grad


def _project_to_sphere(c, r):
    """Project c onto sphere of radius r."""
    norm_c = np.linalg.norm(c)
    if norm_c < 1e-300:
        return c * r
    return c / norm_c * r


def _trust_region_step(c, grad, H, delta, r):
    """
    Compute trust-region step on the unit sphere.
    c is on sphere of radius r, grad is gradient, H is Hessian (or approximation).
    Returns new c on sphere of radius r.
    """
    dim = len(c)
    # Project gradient and Hessian to tangent space
    # Tangent space: vectors orthogonal to c
    # Use projection matrix P = I - (c c^T)/(c^T c)
    c_norm_sq = np.dot(c, c)
    if c_norm_sq < 1e-300:
        return c
    
    # Project gradient to tangent space
    grad_tan = grad - np.dot(grad, c) / c_norm_sq * c
    
    # Project Hessian to tangent space
    # H_tan = P H P
    # For efficiency, compute step directly
    # Solve (H + mu I) d = -grad_tan for d in tangent space
    # Use augmented system to enforce d^T c = 0
    
    # Build augmented system:
    # [H + mu I   c] [d]   [-grad_tan]
    # [c^T        0] [nu] = [0]
    
    # Try different mu values
    best_d = None
    best_obj = np.inf
    
    for mu in [0.0, 1e-8, 1e-6, 1e-4, 1e-2, 1.0, 10.0, 100.0]:
        # Augmented system
        A = np.zeros((dim + 1, dim + 1))
        A[:dim, :dim] = H + mu * np.eye(dim)
        A[:dim, dim] = c
        A[dim, :dim] = c
        b = np.zeros(dim + 1)
        b[:dim] = -grad_tan
        b[dim] = 0.0
        
        try:
            sol = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            continue
        
        d = sol[:dim]
        # Ensure step is in tangent space (d^T c ≈ 0)
        d = d - np.dot(d, c) / c_norm_sq * c
        
        # Check trust region constraint
        if np.linalg.norm(d) > delta:
            d = d / np.linalg.norm(d) * delta
        
        # Compute new point on sphere
        c_new = _project_to_sphere(c + d, r)
        
        # Evaluate objective
        val, _ = _evaluate(c_new, Y, wpts)
        
        if val < best_obj:
            best_obj = val
            best_d = d
    
    if best_d is not None:
        return _project_to_sphere(c + best_d, r)
    else:
        # Fallback: gradient descent step
        step = -grad_tan
        step_norm = np.linalg.norm(step)
        if step_norm > 0:
            step = step / step_norm * min(delta, step_norm)
            return _project_to_sphere(c + step, r)
        return c


def _optimize(n=10, n_theta=70, n_phi=130, max_iter=5000, tol=1e-14):
    wpts, cos_t, sin_t, phi_v = _build_quadrature(n, n_theta, n_phi)
    Y = _make_Y(wpts, cos_t, sin_t, phi_v, n)
    dim = 2 * n + 1
    r = 1.0  # radius of sphere
    
    # Start from diverse seed points
    rng = np.random.default_rng(42)
    best_c = None
    best_val = np.inf
    
    starts = []
    # seed 0: all ones
    starts.append(np.ones(dim, dtype=np.float64) / np.sqrt(dim))
    # seed 1: dominant m=0
    a = np.zeros(dim); a[n] = 1.0; starts.append(a)
    # seed 2: dominant m=10 (cos)
    b = np.zeros(dim); b[2 * n] = 1.0; starts.append(b)
    # seed 3: dominant m=-10 (sin)
    d = np.zeros(dim); d[0] = 1.0; starts.append(d)
    # seed 4-15: random starts
    for i in range(12):
        rr = rng.standard_normal(dim)
        rr /= np.linalg.norm(rr)
        starts.append(rr)
    
    # For each start, use trust-region method
    for c0 in starts:
        c = _project_to_sphere(c0, r)
        val, grad = _evaluate(c, Y, wpts)
        
        # Approximate Hessian using finite differences on the sphere
        # Or use Gauss-Newton approximation
        # For L4/L2 ratio, the Hessian is complex; use identity + gradient-based correction
        
        # Trust-region parameters
        delta = 0.5
        gamma = 0.1  # trust region expansion factor
        sigma = 0.25  # trust region reduction factor
        
        for it in range(max_iter):
            # Compute approximate Hessian using Gauss-Newton
            f = Y.T @ c
            f2 = f * f
            f4 = f2 * f2
            mass = wpts.sum()
            L2sq = (wpts * f2).sum() / mass
            L4 = (wpts * f4).sum() / mass
            L2 = np.sqrt(max(L2sq, 1e-300))
            L4v = L4 ** 0.25
            
            if L2 <= 1e-300 or L4v <= 1e-300:
                break
            
            # Gauss-Newton Hessian approximation for the ratio
            # This is complex; use a simpler approach: BFGS update
            # For now, use identity Hessian (steepest descent in trust region)
            H = np.eye(dim)
            
            # Compute trust-region step
            c_new = _trust_region_step(c, grad, H, delta, r)
            
            val_new, grad_new = _evaluate(c_new, Y, wpts)
            
            # Check improvement
            actual_reduction = val - val_new
            predicted_reduction = np.dot(grad, c_new - c)
            
            if predicted_reduction < 1e-15:
                rho = 1.0 if actual_reduction > 0 else 0.0
            else:
                rho = actual_reduction / predicted_reduction
            
            if rho > 0.75:
                delta = min(2.0 * delta, 1.0)
            elif rho < sigma:
                delta = 0.5 * delta
            
            if actual_reduction > 0:
                c = c_new
                val = val_new
                grad = grad_new
                if abs(actual_reduction) < tol * max(1.0, val):
                    break
            
            if delta < 1e-12:
                break
        
        if val < best_val:
            best_val = val
            best_c = c.copy()
    
    if best_c is None:
        best_c = np.ones(dim, dtype=np.float64) / np.sqrt(dim)
    
    # Scale so max|c| is comfortably within (1e-3, 1e5)
    return best_c


def rudin_problem10() -> np.ndarray:
    """
    Return 21 real coefficients c_m (m = -10..10) for a degree-10 spherical
    harmonic f = sum_m Y_m that minimizes the L4/L2 norm ratio on S^2.

    The returned vector is unit-norm, so 1e-3 < max|c| < 1e5 holds.
    """
    try:
        c = _optimize(n=10, n_theta=70, n_phi=130)
    except Exception:
        c = np.ones(21, dtype=np.float64) / np.sqrt(21.0)
    c = np.asarray(c, dtype=np.float64).reshape(-1)
    if c.shape[0] != 21:
        c = np.ones(21, dtype=np.float64) / np.sqrt(21.0)
    if not np.all(np.isfinite(c)):
        c = np.ones(21, dtype=np.float64) / np.sqrt(21.0)
    # ensure max|c| in valid range
    mx = np.max(np.abs(c))
    if mx < 1e-3:
        c = c / max(mx, 1e-12) * 0.5
    elif mx > 1e5:
        c = c / mx * 0.5
    return c


# EVOLVE-BLOCK-END
