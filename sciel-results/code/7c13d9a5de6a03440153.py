"""Chebyshev collocation approach to minimize uncertainty product upper bound."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = time.time() + 200


def _chebyshev_setup(N):
    """
    Set up Chebyshev collocation points, weights, and derivative matrices.
    We work on [-L, L] with N+1 Chebyshev points.
    
    Returns:
        x: collocation points (N+1,)
        w: quadrature weights (N+1,)
        D1: first derivative matrix (N+1, N+1)
        D2: second derivative matrix (N+1, N+1)
    """
    N = int(N)
    L = 8.0  # domain half-width
    
    # Chebyshev points on [-1, 1]
    j = np.arange(N + 1)
    t = np.cos(np.pi * j / N)
    t = t[::-1]  # ascending order
    x = L * t
    
    # Weights for Chebyshev quadrature
    w = np.ones(N + 1) * np.pi / N
    w[0] *= 0.5
    w[-1] *= 0.5
    w = L * w  # scale to [-L, L]
    
    # Derivative matrices
    # Standard Chebyshev differentiation matrix on [-1, 1]
    c = np.ones(N + 1)
    c[0] = 0.5
    c[-1] = 0.5
    c = c * (-1.0) ** np.arange(N + 1)
    
    X = np.tile(t, (N + 1, 1))
    dX = X - X.T
    dX[np.abs(dX) < 1e-15] = 1.0  # avoid division by zero
    
    # Off-diagonal
    D = (c[:, None] / c[None, :]) / (dX + np.eye(N + 1))
    # Diagonal
    D = D - np.diag(D.sum(axis=1))
    
    # Scale to [-L, L]
    D1 = D / L
    
    # D2 = D1 @ D1
    D2 = D1 @ D1
    
    return x, w, D1, D2


def _build_functional(N, L=8.0):
    """
    Build the quadratic forms for position and momentum variances.
    
    g(x) = P(x) * exp(-x^2/2) where P is a polynomial of degree <= 12
    with coefficients (a0, a4, a8, a12) where a12 is determined by g(0)=0.
    
    We parameterize by (a0, a4, a8) and compute a12 = -(a0 + 3*a4 + 105*a8)/945.
    
    The key insight: g(x) = sum_{n in {0,4,8,12}} a_n H_n(x) exp(-x^2/2)
    Since H_n(x) exp(-x^2/2) are the Hermite functions (up to normalization),
    and the Fourier transform of H_n(x)exp(-x^2/2) is (-i)^n H_n(x)exp(-x^2/2),
    we can work in the Hermite basis.
    
    However, the Chebyshev approach: we discretize and build matrices.
    
    Let's define basis functions phi_n(x) = H_n(x) exp(-x^2/2) for n = 0, 4, 8, 12.
    g(x) = a0*phi_0 + a4*phi_4 + a8*phi_8 + a12*phi_12
    
    A(g)^2 = <x^2> = sum_{i,j} a_i a_j <x phi_i, phi_j> / <g, g>
    A(G)^2 = <xi^2> = sum_{i,j} a_i a_j <xi G_i, G_j> / <G, G>
    
    where G_n = Fourier(phi_n) = (-i)^n phi_n (for the standard Hermite functions with exp(-x^2/2))
    
    Actually, let me be more careful. The Fourier transform convention matters.
    If F(ξ) = ∫ f(x) e^{-2πixξ} dx, then F(H_n exp(-x^2/2)) = (-i)^n H_n(ξ) exp(-ξ^2/2)
    up to a constant factor depending on convention.
    
    For simplicity, let's use the convention where the Fourier transform of
    H_n(x) e^{-x^2/2} is (-i)^n H_n(x) e^{-x^2/2} (this is for the convention
    F(ξ) = (1/√(2π)) ∫ f(x) e^{-ixξ} dx).
    
    So |G|^2 has the same form as |g|^2 but with coefficients multiplied by (-i)^n,
    which for real coefficients just changes signs for odd n. Since we only use even n,
    the signs are all +1. So the momentum variance matrix is the same as position!
    
    Wait, that's not right. Let me reconsider.
    
    Actually for the standard Hermite functions ψ_n(x) = (1/√(2^n n! √π)) H_n(x) e^{-x^2/2},
    the Fourier transform is (-i)^n ψ_n(x).
    
    So if g = sum a_n ψ_n, then G = sum a_n (-i)^n ψ_n.
    For even n, (-i)^n = (-1)^{n/2}.
    
    So the momentum variance involves cross terms with factors (-1)^{n/2}.
    
    Let me just compute everything numerically using the collocation points.
    """
    x, w, D1, D2 = _chebyshev_setup(N)
    
    # Hermite polynomials H_n(x) for n = 0, 4, 8, 12
    def H0(x): return np.ones_like(x)
    def H4(x): return x**4 - 6*x**2 + 3
    def H8(x): return x**8 - 28*x**6 + 210*x**4 - 420*x**2 + 105
    def H12(x): return x**12 - 66*x**10 + 1155*x**8 - 6930*x**6 + 13860*x**4 - 7560*x**2 + 945
    
    # Basis functions: phi_n(x) = H_n(x) * exp(-x^2/2)
    exp_factor = np.exp(-x**2 / 2.0)
    phi0 = H0(x) * exp_factor
    phi4 = H4(x) * exp_factor
    phi8 = H8(x) * exp_factor
    phi12 = H12(x) * exp_factor
    
    # Inner products: <f, g> ≈ sum w_i f(x_i) g(x_i)
    # Position operator: x * f
    # Momentum operator: -i * D1 * f (in the Fourier domain)
    
    # For the position variance:
    # <g, x^2 g> = sum w_i |g(x_i)|^2 x_i^2
    # <g, g> = sum w_i |g(x_i)|^2
    
    # For the momentum variance:
    # G = Fourier(g), and <G, xi^2 G> = <g, (-i D1)^2 g> = <g, -D1^2 g>
    # But this is in the position representation. Actually, the momentum variance
    # is <g, p^2 g> where p = -i d/dx, so p^2 = -d^2/dx^2 = -D2.
    
    # So we need:
    # V_pos = <g, x^2 g> / <g, g>
    # V_mom = <g, -D2 g> / <g, g>
    
    # g = a0*phi0 + a4*phi4 + a8*phi8 + a12*phi12
    # where a12 = -(a0 + 3*a4 + 105*a8)/945
    
    # Let's build the matrices:
    # M_pos[i,j] = <phi_i, x^2 phi_j>
    # M_mom[i,j] = <phi_i, -D2 phi_j>
    # M_0[i,j] = <phi_i, phi_j>
    
    phis = np.array([phi0, phi4, phi8, phi12])  # shape (4, N+1)
    
    # M_0
    M_0 = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            M_0[i, j] = np.sum(w * phis[i] * phis[j])
    
    # M_pos
    x2 = x**2
    M_pos = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            M_pos[i, j] = np.sum(w * x2 * phis[i] * phis[j])
    
    # M_mom: <phi_i, -D2 phi_j>
    # -D2 phi_j is the action of the momentum-squared operator on phi_j
    M_mom = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            D2_phi_j = D2 @ phis[j]
            M_mom[i, j] = -np.sum(w * phis[i] * D2_phi_j)
    
    return M_0, M_pos, M_mom


def _bound_from_matrices(params, M_0, M_pos, M_mom):
    """
    Compute the uncertainty bound from the precomputed matrices.
    params = (a0, a4, a8)
    a12 = -(a0 + 3*a4 + 105*a8)/945
    """
    a0, a4, a8 = params
    a12 = -(a0 + 3.0 * a4 + 105.0 * a8) / 945.0
    a = np.array([a0, a4, a8, a12])
    
    # <g, g> = a^T M_0 a
    norm2 = a @ M_0 @ a
    if norm2 <= 1e-30 or not np.isfinite(norm2):
        return 1e18
    
    # <g, x^2 g> = a^T M_pos a
    pos_var = a @ M_pos @ a
    if pos_var <= 0 or not np.isfinite(pos_var):
        return 1e18
    
    # <g, p^2 g> = a^T M_mom a
    mom_var = a @ M_mom @ a
    if mom_var <= 0 or not np.isfinite(mom_var):
        return 1e18
    
    vf = pos_var / norm2
    vF = mom_var / norm2
    
    if vf <= 0 or vF <= 0:
        return 1e18
    
    return np.sqrt(vf * vF)


def _bound_and_grad_cheb(params, M_0, M_pos, M_mom):
    """
    Compute bound and gradient using finite differences for the gradient
    (since analytic gradients through the matrix products are complex).
    Actually, let's derive the analytic gradient.
    
    bound = sqrt( (a^T M_pos a) * (a^T M_mom a) / (a^T M_0 a)^2 )
    
    Let N = a^T M_0 a, P = a^T M_pos a, Q = a^T M_mom a
    bound = sqrt(P * Q / N^2)
    
    d(bound)/da_c = (1/(2*bound)) * d(P*Q/N^2)/da_c
    
    d(P*Q/N^2)/da_c = (dP*Q + P*dQ)*N^2 - P*Q*2*N*dN / N^4
    = (dP*Q + P*dQ)/N^2 - 2*P*Q*dN/N^3
    
    where dP/da_c = 2 * (M_pos @ a)_c, etc.
    
    But a depends on params through a12.
    da_c/dparams:
    a = [a0, a4, a8, a12]
    a12 = -(a0 + 3*a4 + 105*a8)/945
    
    d a0/d a0 = 1, d a0/d a4 = 0, d a0/d a8 = 0
    d a4/d a0 = 0, d a4/d a4 = 1, d a4/d a8 = 0
    d a8/d a0 = 0, d a8/d a4 = 0, d a8/d a8 = 1
    d a12/d a0 = -1/945, d a12/d a4 = -3/945, d a12/d a8 = -105/945
    
    So da/dparams is a 4x3 matrix.
    """
    a0, a4, a8 = params
    a12 = -(a0 + 3.0 * a4 + 105.0 * a8) / 945.0
    a = np.array([a0, a4, a8, a12])
    
    N = a @ M_0 @ a
    P = a @ M_pos @ a
    Q = a @ M_mom @ a
    
    if N <= 1e-30 or P <= 0 or Q <= 0 or not np.isfinite(N) or not np.isfinite(P) or not np.isfinite(Q):
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    bound = np.sqrt(P * Q / (N * N))
    
    # Gradient
    # da/dparams (4x3)
    da_dparams = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [-1.0/945.0, -3.0/945.0, -105.0/945.0]
    ])
    
    # dN/dparams = 2 * (M_0 @ a)^T @ da_dparams
    M0a = M_0 @ a
    Mposa = M_pos @ a
    Mmoma = M_mom @ a
    
    dN_dparams = 2.0 * (da_dparams @ M0a)  # shape (3,)
    dP_dparams = 2.0 * (da_dparams @ Mposa)
    dQ_dparams = 2.0 * (da_dparams @ Mmoma)
    
    # d(P*Q/N^2)/dparams
    # = (dP*Q + P*dQ)/N^2 - 2*P*Q*dN/N^3
    dPQN2 = (dP_dparams * Q + P * dQ_dparams) / (N * N) - 2.0 * P * Q * dN_dparams / (N * N * N)
    
    grad = dPQN2 / (2.0 * bound)
    
    if not np.all(np.isfinite(grad)):
        grad = np.array([1e18, 1e18, 1e18])
    
    return bound, grad


def construct():
    # Use Chebyshev collocation with moderate N
    N = 200  # number of intervals, N+1 points
    
    try:
        M_0, M_pos, M_mom = _build_functional(N)
    except Exception:
        # Fallback to a simple approach
        return np.array([-1.13, 0.04, 0.000308642])
    
    def _obj(params):
        val, _ = _bound_and_grad_cheb(params, M_0, M_pos, M_mom)
        return val
    
    def _obj_grad(params):
        val, grad = _bound_and_grad_cheb(params, M_0, M_pos, M_mom)
        return val, grad
    
    # Multiple starting points
    starts = [
        np.array([-1.13, 0.04, 0.000308642]),  # seed
        np.array([0.0, 0.0, 0.0]),
        np.array([-1.0, 0.1, 0.001]),
        np.array([-2.0, 0.05, 0.0005]),
        np.array([-0.5, 0.02, 0.0002]),
        np.array([-1.5, 0.08, 0.0008]),
        np.array([1.0, -0.05, 0.0005]),
        np.array([-3.0, 0.15, 0.002]),
        np.array([-0.8, 0.03, 0.0001]),
        np.array([-1.2, 0.05, 0.0004]),
    ]
    
    rng = np.random.default_rng(42)
    for i in range(15):
        starts.append(rng.normal(0, 0.5, 3))
    
    best_val = np.inf
    best_params = None
    
    for start in starts:
        if time.time() > _DEADLINE - 30:
            break
        try:
            res = minimize(_obj_grad, start, method='L-BFGS-B',
                          jac=True, options={'maxiter': 3000, 'ftol': 1e-15, 'gtol': 1e-14})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Polish with SLSQP
    if best_params is not None and time.time() < _DEADLINE - 15:
        try:
            res = minimize(_obj, best_params, method='SLSQP',
                          options={'maxiter': 5000, 'ftol': 1e-15})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Final polish
    if best_params is not None and time.time() < _DEADLINE - 5:
        try:
            res = minimize(_obj_grad, best_params, method='L-BFGS-B',
                          jac=True, options={'maxiter': 5000, 'ftol': 1e-16, 'gtol': 1e-15})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    if best_params is None:
        return np.array([-1.13, 0.04, 0.000308642])
    
    return np.array(best_params)
# EVOLVE-BLOCK-END
