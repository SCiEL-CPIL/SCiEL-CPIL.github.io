# EVOLVE-BLOCK-START
import numpy as np
import time


def _build_quadrature_and_harmonics():
    """Build quadrature nodes and the 21 x N matrix of real orthonormal
    degree-10 spherical harmonic values at each node."""
    k = 10
    n_theta = 70
    n_phi = 130

    # Gauss-Legendre nodes in [-1, 1]
    nodes, weights = np.polynomial.legendre.leggauss(n_theta)
    # weights sum to 2 over [-1,1]; surface element dOmega = dphi * d(cos theta)
    # total mass = 2*pi * 2 = 4*pi  ->  phi quadrature weight = 2*pi/n_phi
    w_phi = 2.0 * np.pi / n_phi
    w_theta = weights  # each multiplies d(cos theta)

    # Full weights: w_ij = w_phi * w_theta_i
    W = np.outer(w_theta, np.full(n_phi, w_phi))  # shape (n_theta, n_phi)
    W_flat = W.ravel()  # length N = n_theta*n_phi

    # Node coordinates
    cos_t = nodes  # (n_theta,)
    sin_t = np.sqrt(np.maximum(0.0, 1.0 - cos_t**2))
    phi = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)

    # Broadcast to full grid (n_theta, n_phi)
    cos_t_grid, phi_grid = np.meshgrid(cos_t, phi, indexing='ij')
    sin_t_grid, _ = np.meshgrid(sin_t, phi, indexing='ij')

    # Associated Legendre P_l^m(x) with Condon-Shortley phase.
    # Use scipy if available for accuracy; fallback to numpy recurrence.
    try:
        from scipy.special import lpmv
        def P_l_m(m, l, x):
            return lpmv(m, l, x)  # includes (-1)^m Condon-Shortley
    except Exception:
        # Fallback: standard recurrence (slow but fine for l=10)
        def P_l_m(m, l, x):
            # Compute P_l^m via recurrence
            # P_m^m = (-1)^m (2m-1)!! (1-x^2)^(m/2)
            Pmm = 1.0
            if m > 0:
                somx2 = np.sqrt(np.maximum(0.0, 1.0 - x**2))
                for i in range(1, m + 1):
                    Pmm *= -(2.0 * i - 1.0) * somx2
            if l == m:
                return Pmm
            Pmp1 = x * (2.0 * m + 1.0) * Pmm if m > 0 else x
            if l == m + 1:
                return Pmp1
            for ll in range(m + 2, l + 1):
                Pl = (x * (2.0 * ll - 1.0) * Pmp1 - (ll + m - 1.0) * Pmm) / (ll - m)
                Pmm = Pmp1
                Pmp1 = Pl
            return Pmp1

    N_pts = W_flat.size
    H = np.zeros((2 * k + 1, N_pts), dtype=np.float64)

    # m = 0
    N0 = np.sqrt((2 * k + 1) / (4.0 * np.pi) * 1.0)  # (10-0)!/(10+0)! = 1
    P0 = P_l_m(0, k, cos_t_grid)
    H[10, :] = (N0 * P0).ravel()

    for m in range(1, k + 1):
        Nm = np.sqrt((2 * k + 1) / (4.0 * np.pi) *
                     float(np.math.factorial(k - m)) / float(np.math.factorial(k + m)))
        Pm = P_l_m(m, k, cos_t_grid)
        sqrt2_Nm_Pm = (np.sqrt(2.0) * Nm * Pm).ravel()
        # m > 0: cos(m phi)
        H[10 + m, :] = sqrt2_Nm_Pm * np.cos(m * phi_grid).ravel()
        # m < 0: sin(|m| phi)
        H[10 - m, :] = sqrt2_Nm_Pm * np.sin(m * phi_grid).ravel()

    return H, W_flat, N_pts


def _compute_R(c, H, W):
    """Compute R = ||f||_4 / ||f||_2 using quadrature."""
    f = H @ c  # (N,)
    f2 = f * f
    f4 = f2 * f2
    n2 = float(np.dot(W, f2))
    if n2 <= 0.0:
        return np.inf
    n4 = float(np.dot(W, f4))
    if n4 <= 0.0:
        return 0.0
    norm2 = np.sqrt(n2)
    norm4 = n4 ** 0.25
    return norm4 / norm2


def _projected_gradient_step(c, H, W, lr, it):
    """One step of projected gradient descent on the unit sphere to
    minimize ||f||_4 (equivalently R since we keep ||c||=1)."""
    f = H @ c
    f2 = f * f
    f4 = f2 * f2
    n2 = float(np.dot(W, f2))
    if n2 <= 0.0:
        return c
    norm2 = np.sqrt(n2)
    norm4 = n4 = float(np.dot(W, f4))
    norm4_val = n4 ** 0.25
    R = norm4_val / norm2

    # Gradient of g(c) = ||f||_4 = (sum W f^4)^{1/4}
    # dg/dc_j = (1/4) g^{-3} * sum_i W_i * 4 f_i^3 H_ij = g^{-3} sum_i W_i f_i^3 H_ij
    # = g^{-3} (H^T (W * f^3))_j
    Wf3 = W * (f * f * f)
    grad = (H.T @ Wf3) / (norm4_val ** 3)

    # Normalize gradient to unit sphere direction for stable steps
    gc_norm = np.linalg.norm(grad)
    if gc_norm < 1e-15:
        return c

    # Adaptive learning rate
    base_lr = lr
    step = base_lr

    c_new = c - step * grad
    # Project back to unit sphere
    cn = np.linalg.norm(c_new)
    if cn < 1e-15:
        return c
    c_new = c_new / cn

    # Line search: accept if R improves
    R_new = _compute_R(c_new, H, W)
    if R_new < R:
        return c_new

    # Backtrack
    for _ in range(8):
        step *= 0.5
        c_try = c - step * grad
        cn = np.linalg.norm(c_try)
        if cn < 1e-15:
            continue
        c_try = c_try / cn
        R_try = _compute_R(c_try, H, W)
        if R_try < R:
            return c_try
    return c


def _local_polish(c, H, W, max_iters=2000, tol=1e-12):
    """High-precision polish: projected gradient with very small steps
    and tight tolerance."""
    best_c = c.copy()
    best_R = _compute_R(c, H, W)
    lr = 0.01
    for it in range(max_iters):
        c_new = _projected_gradient_step(c, H, W, lr, it)
        R_new = _compute_R(c_new, H, W)
        if R_new < best_R:
            best_R = R_new
            best_c = c_new
        if np.linalg.norm(c_new - c) < tol:
            break
        c = c_new
        # Slowly reduce lr
        if it % 200 == 0:
            lr *= 0.8
    return best_c, best_R


def _random_unit_vector(dim, rng):
    v = rng.standard_normal(dim)
    v /= np.linalg.norm(v)
    return v


def _try_directions(c, H, W, rng, n_dirs=20):
    """Try random directions on the sphere to escape local minima."""
    best_c = c.copy()
    best_R = _compute_R(c, H, W)
    for _ in range(n_dirs):
        v = _random_unit_vector(c.size, rng)
        # Move a small fraction toward v
        for frac in [0.05, 0.1, 0.2, 0.3]:
            c_try = (1.0 - frac) * c + frac * v
            cn = np.linalg.norm(c_try)
            if cn < 1e-15:
                continue
            c_try /= cn
            R_try = _compute_R(c_try, H, W)
            if R_try < best_R:
                best_R = R_try
                best_c = c_try
    return best_c, best_R


def rudin_problem10() -> np.ndarray:
    t0 = time.time()
    deadline = 600.0 - 10.0  # 10s safety margin
    k = 10
    dim = 2 * k + 1  # 21

    # Build quadrature and harmonic matrix (fast, ~ms)
    H, W, N_pts = _build_quadrature_and_harmonics()

    rng = np.random.default_rng(42)
    best_c = None
    best_R = np.inf

    # Baseline: all ones, normalized
    c0 = np.ones(dim)
    c0 /= np.linalg.norm(c0)
    R0 = _compute_R(c0, H, W)
    best_c = c0.copy()
    best_R = R0

    # Try some structured initial points
    structured = []
    # All ones (already done)
    # Alternating signs
    for pattern in [
        np.ones(dim),
        np.array([(-1)**i for i in range(dim)]),
        np.array([1 if i == 10 else 0 for i in range(dim)]),  # only m=0
        np.array([1 if i in [10-5, 10+5] else 0 for i in range(dim)]),
    ]:
        v = pattern.copy()
        n = np.linalg.norm(v)
        if n > 1e-15:
            v /= n
            structured.append(v)

    for v in structured:
        Rv = _compute_R(v, H, W)
        if Rv < best_R:
            best_R = Rv
            best_c = v.copy()

    # Multiple random restarts with projected gradient
    n_restarts = 30
    max_iters = 3000
    base_lr = 0.02

    for r in range(n_restarts):
        if time.time() > deadline - 30.0:
            break

        if r == 0:
            c = best_c.copy()
        else:
            c = _random_unit_vector(dim, rng)

        R_c = _compute_R(c, H, W)
        if R_c < best_R:
            best_R = R_c
            best_c = c.copy()

        # Gradient descent phase
        lr = base_lr
        for it in range(max_iters):
            if time.time() > deadline - 60.0:
                break
            c_new = _projected_gradient_step(c, H, W, lr, it)
            R_new = _compute_R(c_new, H, W)
            if R_new < best_R:
                best_R = R_new
                best_c = c_new.copy()
            if np.linalg.norm(c_new - c) < 1e-14:
                break
            c = c_new
            if it % 500 == 0 and it > 0:
                lr *= 0.7

        # Try escape directions
        if time.time() > deadline - 45.0:
            break
        c_esc, R_esc = _try_directions(c, H, W, rng, n_dirs=10)
        if R_esc < best_R:
            best_R = R_esc
            best_c = c_esc.copy()

    # Final high-precision polish
    if time.time() < deadline - 15.0:
        polished_c, polished_R = _local_polish(best_c, H, W, max_iters=5000, tol=1e-14)
        if polished_R < best_R:
            best_R = polished_R
            best_c = polished_c.copy()

    # Validate
    c_out = best_c.astype(np.float64)
    c_out = np.nan_to_num(c_out, nan=0.0, posinf=0.0, neginf=0.0)
    if c_out.size != 21:
        c_out = np.ones(21)
    max_abs = np.max(np.abs(c_out))
    if max_abs < 1e-3:
        c_out = np.ones(21)
    elif max_abs > 1e5:
        c_out = c_out / max_abs * 1.0

    return c_out


# EVOLVE-BLOCK-END
