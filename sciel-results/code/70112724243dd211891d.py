# EVOLVE-BLOCK-START
import numpy as np
import time

def _assoc_legendre(n, m, x):
    """Associated Legendre P_n^m(x) with Condon-Shortley phase, |m| <= n."""
    m = abs(int(m))
    n = int(n)
    # Compute P_n^m via recurrence starting from P_m^m
    Pmm = 1.0
    if m > 0:
        somx2 = np.sqrt((1.0 - x) * (1.0 + x))
        fact = 1.0
        for i in range(1, m + 1):
            Pmm = -Pmm * i * somx2
            fact *= i
        # Pmm now = (-1)^m (2m-1)!! (1-x^2)^{m/2} = P_m^m
    Pmmp1 = (2 * m + 1) * x * Pmm if m < n else np.zeros_like(x)
    if m == n:
        return Pmm
    Pn = np.zeros_like(x)
    for nn in range(m + 1, n + 1):
        Pn = ((2 * nn - 1) * x * Pmmp1 - (nn + m - 1) * Pmm) / (nn - m)
        Pmm = Pmmp1
        Pmmp1 = Pn
    return Pn

def _build_quadrature():
    """70-point Gauss-Legendre in cos(theta) x 130 equally spaced phi."""
    # Gauss-Legendre nodes/weights on [-1,1]
    mu, w = np.polynomial.legendre.leggauss(70)
    nphi = 130
    phi = np.linspace(0.0, 2.0 * np.pi, nphi, endpoint=False)
    dphi = 2.0 * np.pi / nphi
    # Surface measure: dOmega = dmu dphi (since dOmega = sin(theta)dtheta dphi = -dmu dphi, mu=cos theta)
    # Total mass should be 4*pi: sum(w)*sum(dphi) = 2 * 2pi = 4pi. Good.
    return mu, w, phi, dphi

def _make_Y(k, mu, phi):
    """Return array of shape (nmu*nphi, 2k+1) of real orthonormal harmonics Y_m, m=-k..k."""
    nmu = len(mu)
    nphi = len(phi)
    npts = nmu * nphi
    Y = np.zeros((npts, 2 * k + 1), dtype=np.float64)
    # Precompute associated Legendre for each m
    P = {}
    for m in range(k + 1):
        P[m] = _assoc_legendre(k, m, mu)
    # Normalization N_m
    N = {}
    for m in range(k + 1):
        N[m] = np.sqrt((2 * k + 1) / (4.0 * np.pi) * (np.math.factorial(k - m) / np.math.factorial(k + m)))
    # Build mu and phi grids
    mu_grid, phi_grid = np.meshgrid(mu, phi, indexing='ij')
    mu_flat = mu_grid.ravel()
    phi_flat = phi_grid.ravel()
    for m in range(k + 1):
        Pm = P[m][mu_grid].ravel()
        Nm = N[m]
        if m == 0:
            Y[:, m] = Nm * Pm  # m=0 index
        else:
            Y[:, k + m] = np.sqrt(2.0) * Nm * Pm * np.cos(m * phi_flat)   # m>0
            Y[:, k - m] = np.sqrt(2.0) * Nm * Pm * np.sin(m * phi_flat)   # m<0
    return Y

def _norms(f_vals, w, dphi):
    """Compute ||f||_2 and ||f||_4 over the sphere using quadrature."""
    # f_vals shape (npts,)
    # weight for each point: w_i * dphi
    # Build full weight vector
    nmu = int(np.sqrt(len(w) * len(dphi) / dphi)) if False else None
    return None

def _quadrature_weights(w, dphi, nmu, nphi):
    """Build per-point weights: w_i * dphi."""
    # w has length nmu, dphi scalar
    # Each mu-node repeated nphi times
    wpts = np.tile(w, nphi) * dphi
    return wpts

def _compute_T_and_eval(k, mu, w, phi, dphi):
    """Compute the 4th-order tensor T_{ijkl} = ∫ Y_i Y_j Y_k Y_l dΩ, and return it as a 4D array."""
    nmu = len(mu)
    nphi = len(phi)
    npts = nmu * nphi
    Y = _make_Y(k, mu, phi)  # (npts, 2k+1)
    wpts = np.tile(w, nphi) * dphi  # (npts,)
    # T_{ijkl} = sum_p wpts_p Y[p,i] Y[p,j] Y[p,k] Y[p,l]
    # Build via outer products. n = 2k+1 = 21.
    n = 2 * k + 1
    # Yw = Y * wpts[:, None]
    Yw = Y * wpts[:, None]
    # T = sum_p (Yw_p ⊗ Y_p ⊗ Y_p ⊗ Y_p) -- but that's not right either.
    # T_{ijkl} = sum_p wpts_p * Y[p,i]*Y[p,j]*Y[p,k]*Y[p,l]
    # = sum_p (Yw[p,i] * Y[p,j] * Y[p,k] * Y[p,l])
    # Compute as: for each p, outer product of 4 vectors Y[p,:] weighted.
    # Efficient: T = einsum('pi,pj,pk,pl,p->ijkl', Y, Y, Y, Y, wpts)
    T = np.einsum('pi,pj,pk,pl,p->ijkl', Y, Y, Y, Y, wpts, optimize=True)
    return T, Y, wpts

def _objective(c, T):
    """J(c) = sum_{ijkl} T_{ijkl} c_i c_j c_k c_l."""
    return np.einsum('ijkl,i,j,k,l->', T, c, c, c, c, optimize=True)

def _grad(c, T):
    """Gradient of J w.r.t. c: 4 * sum_{jkl} T_{ijkl} c_j c_k c_l."""
    return 4.0 * np.einsum('ijkl,j,k,l->i', T, c, c, c, optimize=True)

def _hess(c, T, Y, wpts):
    """Hessian H_{pq} = 12 * sum_p wpts_p f_p^2 Y[p,q] ... wait, H_{pq} = 12 ∫ f² Y_p Y_q dΩ where Y_p is the p-th harmonic basis function evaluated at points."""
    # f at each point: f_vals = Y @ c
    f_vals = Y @ c
    # H_{pq} = 12 * sum_points wpts * f_vals^2 * Y[:,p] * Y[:,q]
    # = 12 * (Y * (wpts * f_vals**2)[:,None])^T @ Y
    w_f2 = wpts * (f_vals ** 2)
    H = 12.0 * (Y * w_f2[:, None]).T @ Y
    return H

def _projected_newton(T, Y, wpts, c0, max_iter=200, tol=1e-12, deadline=None):
    """Minimize J(c) s.t. c^T c = 1 using projected Newton on the sphere."""
    n = len(c0)
    c = c0 / np.linalg.norm(c0)
    best_c = c.copy()
    best_J = _objective(c, T)
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        J = _objective(c, T)
        if J < best_J:
            best_J = J
            best_c = c.copy()
        g = _grad(c, T)
        H = _hess(c, T, Y, wpts)
        # Project gradient onto tangent space: g_t = g - (g.c) c
        gc = np.dot(g, c)
        g_t = g - gc * c
        # Newton step on sphere: solve (H - lambda I) d = -g_t with d.c = 0
        # Use Lagrange: minimize 0.5 d^T H d + g_t^T d s.t. d^T c = 0, d^T d = 1 (approx)
        # Simpler: use damped gradient descent with projection, or Newton with line search.
        # Let's do: solve H d = -g_t, then project d onto tangent, normalize, line search.
        try:
            d = np.linalg.solve(H + 1e-12 * np.eye(n), -g_t)
        except np.linalg.LinAlgError:
            d = -g_t
        # Project d onto tangent space
        dc = np.dot(d, c)
        d = d - dc * c
        dn = np.linalg.norm(d)
        if dn < 1e-14:
            break
        d = d / dn
        # Line search
        alpha = 1.0
        for ls in range(30):
            c_new = c + alpha * d
            c_new = c_new / np.linalg.norm(c_new)
            J_new = _objective(c_new, T)
            if J_new < J:
                break
            alpha *= 0.5
        else:
            break
        c = c_new
        if np.linalg.norm(d * alpha) < tol:
            break
    return best_c, best_J

def rudin_problem10() -> np.ndarray:
    """Return 21 real coefficients c_m (m=-10..10) minimizing ||f||_4/||f||_2."""
    t0 = time.time()
    deadline = t0 + 600.0 - 30.0  # 30s safety margin
    
    k = 10
    mu, w, phi, dphi = _build_quadrature()
    nmu = len(mu)
    nphi = len(phi)
    
    T, Y, wpts = _compute_T_and_eval(k, mu, w, phi, dphi)
    
    # Multiple random starts to avoid local minima
    np.random.seed(42)
    best_c = None
    best_J = np.inf
    
    # Start 1: all ones
    c0 = np.ones(2 * k + 1, dtype=np.float64)
    c1, J1 = _projected_newton(T, Y, wpts, c0, max_iter=200, tol=1e-12, deadline=deadline)
    if J1 < best_J:
        best_J = J1
        best_c = c1
    
    # Random starts
    n_starts = 20
    for s in range(n_starts):
        if time.time() > deadline:
            break
        c0 = np.random.randn(2 * k + 1, dtype=np.float64)
        c_s, J_s = _projected_newton(T, Y, wpts, c0, max_iter=150, tol=1e-12, deadline=deadline)
        if J_s < best_J:
            best_J = J_s
            best_c = c_s
    
    # Final polish with tighter tolerance
    if time.time() < deadline - 10:
        c_pol, J_pol = _projected_newton(T, Y, wpts, best_c, max_iter=300, tol=1e-14, deadline=deadline)
        if J_pol < best_J:
            best_J = J_pol
            best_c = c_pol
    
    # Normalize to ensure valid range
    c_out = best_c.copy()
    max_abs = np.max(np.abs(c_out))
    if max_abs < 1e-3:
        c_out = c_out / max_abs * 1e-3
    elif max_abs > 1e5:
        c_out = c_out / max_abs * 1e5
    
    # Ensure finite
    if not np.all(np.isfinite(c_out)):
        c_out = np.ones(2 * k + 1, dtype=np.float64)
    
    return c_out

# EVOLVE-BLOCK-END
