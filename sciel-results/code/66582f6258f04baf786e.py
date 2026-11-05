"""Chebyshev-structured optimization for third autocorrelation inequality.

Strategy:
  * Parameterize h as a constant baseline plus a zero-mean linear combination
    of Chebyshev polynomials of the second kind (imaginary part of (e^{iθ})^k
    structure, i.e., U_k(cos θ) ~ sin((k+1)θ)/sin θ).
  * This gives a smooth, low-dimensional manifold with known spectral properties.
  * Use FFT-based autoconvolution and a soft-max surrogate with analytic gradient.
  * Optimize the small parameter vector (baseline + M Chebyshev coefficients) with
    L-BFGS-B, then polish with coordinate descent on the full 400-dim vector.
  * Multiple restarts with diverse initializations to escape local minima.
"""
N = 400
import time
import numpy as np
from scipy.optimize import minimize
from scipy.special import eval_chebyu


def _autoconv(h):
    """Full autoconvolution via FFT."""
    n = len(h)
    f = np.fft.rfft(h, n=2 * n)
    c = np.fft.irfft(f * f, n=2 * n)
    return c


def _hard(h):
    """Hard objective: abs(2*N*max(conv) / sum(h)^2)."""
    s = h.sum()
    if abs(s) < 1e-9:
        return 1e12
    conv = _autoconv(h)
    return abs(2.0 * N * np.max(conv) / (s * s))


def _chebyshev_basis(n, M):
    """Return an n x M matrix where column k is U_k evaluated on a uniform grid.
    U_k(cos θ) = sin((k+1)θ)/sin θ. We use θ = π*(i+0.5)/n for i=0..n-1
    (Gauss-Chebyshev nodes) for orthogonality.
    """
    # Use uniform grid on [0, π] with offset for better coverage
    theta = np.pi * (np.arange(n) + 0.5) / n
    sin_theta = np.sin(theta)
    basis = np.zeros((n, M))
    for k in range(M):
        # U_k(cos θ) = sin((k+1)*θ) / sin(θ)
        numerator = np.sin((k + 1) * theta)
        # Handle sin_theta near zero (shouldn't happen with offset grid)
        with np.errstate(divide='ignore', invalid='ignore'):
            basis[:, k] = numerator / sin_theta
        # Replace any nan/inf with 0
        basis[~np.isfinite(basis), k] = 0.0
    return basis


def _param_to_h(p, basis, M):
    """Map parameter vector to a 400-length step-height vector.
    p = [baseline, c_1, c_2, ..., c_M]
    h = baseline + sum_k c_k * U_k(grid)
    Then subtract mean of oscillatory part to keep baseline as true DC.
    """
    baseline = p[0]
    coeffs = p[1:M + 1]
    h_osc = basis @ coeffs
    h_osc = h_osc - h_osc.mean()  # ensure zero-mean oscillatory part
    h = baseline + h_osc
    return h


def _soft_max_conv_and_grad(h, beta):
    """Return (soft_max_value, gradient w.r.t. h) using direct computation."""
    n = len(h)
    conv = _autoconv(h)
    m = np.max(conv)
    shifted = beta * (conv - m)
    shifted = np.clip(shifted, -500, 500)
    ez = np.exp(shifted)
    z = ez.sum()
    sm = m + np.log(z) / beta
    w = ez / z

    # Gradient: grad_i = sum_k w_k (h[k-i] + h[i-k])
    # Compute efficiently using FFT
    W = np.fft.rfft(w, n=2 * n)
    H = np.fft.rfft(h, n=2 * n)

    # conv_wh[i] = sum_k w_k h[i-k] (standard convolution)
    conv_wh = np.fft.irfft(W * H, n=2 * n)
    term2 = conv_wh[0:n]  # sum_k w_k h[i-k] for i=0..n-1

    # For term1: sum_k w_k h[k-i]
    # This is the correlation. For real signals, ifft(conj(W) * H) gives
    # sum_k w[k] h[i+k] which is not quite right.
    # Let's use: sum_k w_k h[k-i] = sum_j w_{j+i} h_j
    # This is the cross-correlation of w and h at lag i.
    # corr[i] = sum_j w[j+i] h[j] = ifft(conj(W) * H)[i + offset]?
    # Actually, let's just compute it via the symmetric property.
    # Since autoconvolution is symmetric: conv[k] = conv[2n-2-k]
    # So w is symmetric: w[k] = w[2n-2-k]
    # Therefore sum_k w_k h[k-i] = sum_k w_{2n-2-k} h[k-i]
    # Let j = 2n-2-k, then k = 2n-2-j
    # = sum_j w_j h[2n-2-j-i]
    # This is conv_wh at index (2n-2-i) but we only have 0..2n-1
    # For i in 0..n-1, 2n-2-i is in n-1..2n-2, which is in range.
    term1 = conv_wh[2 * n - 2 - np.arange(n)]

    grad = term1 + term2
    return sm, grad


def _param_obj(p, basis, M, beta):
    """Objective function for parameter optimization."""
    h = _param_to_h(p, basis, M)
    s = h.sum()
    if abs(s) < 1e-9:
        return 1e12
    sm, _ = _soft_max_conv_and_grad(h, beta)
    return 2.0 * N * sm / (s * s)


def _param_obj_grad(p, basis, M, beta):
    """Objective and gradient for parameter optimization."""
    h = _param_to_h(p, basis, M)
    s = h.sum()
    if abs(s) < 1e-9:
        return 1e12, np.zeros_like(p)
    sm, g_sm = _soft_max_conv_and_grad(h, beta)
    val = 2.0 * N * sm / (s * s)
    # Gradient of val w.r.t. h
    g_h = 2.0 * N * (g_sm * s - 2.0 * sm) / (s * s * s)
    # Chain rule: p -> h
    # h = baseline + basis @ coeffs (after mean subtraction)
    # dh/dp[0] = ones (baseline)
    # dh/dp[k] = basis[:, k-1] - basis[:, k-1].mean() for k=1..M
    g_p = np.zeros_like(p)
    g_p[0] = g_h.sum()
    for k in range(1, M + 1):
        col = basis[:, k - 1]
        col_centered = col - col.mean()
        g_p[k] = np.dot(g_h, col_centered)
    return val, g_p


def construct():
    t0 = time.time()
    DEADLINE = 140.0  # wall-clock budget; keep 10s margin before 150s kill

    def time_left():
        return DEADLINE - (time.time() - t0)

    rng = np.random.default_rng(777)

    # Build Chebyshev basis with M modes
    M = 25
    basis = _chebyshev_basis(N, M)

    # Pre-compute centered basis columns for gradient
    basis_centered = np.zeros_like(basis)
    for k in range(M):
        basis_centered[:, k] = basis[:, k] - basis[:, k].mean()

    best_h = np.ones(N, dtype=float)
    best_val = _hard(best_h)

    # ---- Phase 1: Optimize in Chebyshev parameter space ----
    # Try multiple initializations
    num_restarts = 8
    for r in range(num_restarts):
        if time_left() < 40.0:
            break

        # Different initialization strategies
        if r == 0:
            # Start from seed-like: all ones
            p0 = np.concatenate([[1.0], np.zeros(M)])
        elif r == 1:
            # Small random perturbation
            p0 = np.concatenate([[1.0], 0.1 * rng.standard_normal(M)])
        elif r == 2:
            # Emphasize low frequencies
            p0 = np.concatenate([[0.5], np.zeros(M)])
            p0[1:5] = 0.3 * rng.standard_normal(4)
        elif r == 3:
            # Emphasize mid frequencies
            p0 = np.concatenate([[0.5], np.zeros(M)])
            p0[5:15] = 0.3 * rng.standard_normal(10)
        elif r == 4:
            # Higher baseline, small oscillation
            p0 = np.concatenate([[2.0], 0.05 * rng.standard_normal(M)])
        elif r == 5:
            # Negative baseline
            p0 = np.concatenate([[-0.5], 0.2 * rng.standard_normal(M)])
        elif r == 6:
            # Alternating pattern in coefficients
            coeffs = np.zeros(M)
            coeffs[::2] = 0.2
            coeffs[1::2] = -0.1
            p0 = np.concatenate([[0.8], coeffs])
        else:
            # Random
            p0 = np.concatenate([[rng.uniform(0.3, 1.5)], 0.15 * rng.standard_normal(M)])

        # Multi-beta annealing in parameter space
        betas = [10.0, 30.0, 80.0, 200.0]
        h_curr = _param_to_h(p0, basis, M)

        for beta in betas:
            if time_left() < 35.0:
                break
            try:
                def obj(x):
                    return _param_obj(x, basis, M, beta)

                def grad(x):
                    _, g = _param_obj_grad(x, basis, M, beta)
                    return g

                res = minimize(obj, h_curr if False else p0, jac=grad,
                               method='L-BFGS-B',
                               options={'maxiter': 200, 'ftol': 1e-14,
                                        'gtol': 1e-12, 'maxfun': 3000})
                p_opt = res.x
                h_opt = _param_to_h(p_opt, basis, M)
                v = _hard(h_opt)
                if v < best_val:
                    best_val = v
                    best_h = h_opt.copy()
                # Use optimized point for next beta
                p0 = p_opt.copy()
            except Exception:
                pass

    # ---- Phase 2: Full 400-dim polish with soft-max surrogate ----
    h_curr = best_h.copy()
    s = h_curr.sum()
    if abs(s) > 1e-9:
        h_curr = h_curr / s  # normalize sum to 1

    beta_seq = [50.0, 100.0, 200.0, 400.0]
    for beta in beta_seq:
        if time_left() < 20.0:
            break
        try:
            def obj(x):
                sm, _ = _soft_max_conv_and_grad(x, beta)
                ss = x.sum()
                if abs(ss) < 1e-9:
                    return 1e12
                return 2.0 * N * sm / (ss * ss)

            def grad(x):
                sm, g_sm = _soft_max_conv_and_grad(x, beta)
                ss = x.sum()
                if abs(ss) < 1e-9:
                    return np.zeros_like(x)
                return 2.0 * N * (g_sm * ss - 2.0 * sm) / (ss * ss * ss)

            res = minimize(obj, h_curr, jac=grad, method='L-BFGS-B',
                           options={'maxiter': 300, 'ftol': 1e-14,
                                    'gtol': 1e-12, 'maxfun': 5000})
            h_new = res.x
            v = _hard(h_new)
            if v < best_val:
                best_val = v
                best_h = h_new.copy()
                h_curr = h_new.copy()
        except Exception:
            pass

    # ---- Phase 3: Coordinate descent refinement on hard objective ----
    h_curr = best_h.copy()
    s = h_curr.sum()
    if abs(s) > 1e-9:
        h_curr = h_curr / s
    best_local = _hard(h_curr)
    iters = 0
    while time_left() > 10.0 and iters < 40:
        iters += 1
        conv = _autoconv(h_curr)
        kmax = int(np.argmax(conv))
        n = N
        sg = np.zeros(n)
        lag = kmax - (n - 1)
        for i in range(n):
            a = lag - i
            b = i - lag
            if 0 <= a < n:
                sg[i] += h_curr[a]
            if 0 <= b < n:
                sg[i] += h_curr[b]
        ss = h_curr.sum()
        if abs(ss) < 1e-9:
            break
        sg_centered = sg - sg.mean()
        improved = False
        for alpha in [1e-3, 5e-4, 1e-4, 5e-5, 1e-2]:
            cand = h_curr - alpha * sg_centered
            vv = _hard(cand)
            if vv < best_local:
                best_local = vv
                h_curr = cand
                if vv < best_val:
                    best_val = vv
                    best_h = cand.copy()
                improved = True
                break
        if not improved:
            break

    # ---- Final: one more L-BFGS-B from best with high beta ----
    if time_left() > 8.0:
        try:
            beta = 500.0

            def obj(x):
                sm, _ = _soft_max_conv_and_grad(x, beta)
                ss = x.sum()
                if abs(ss) < 1e-9:
                    return 1e12
                return 2.0 * N * sm / (ss * ss)

            def grad(x):
                sm, g_sm = _soft_max_conv_and_grad(x, beta)
                ss = x.sum()
                if abs(ss) < 1e-9:
                    return np.zeros_like(x)
                return 2.0 * N * (g_sm * ss - 2.0 * sm) / (ss * ss * ss)

            res = minimize(obj, best_h, jac=grad, method='L-BFGS-B',
                           options={'maxiter': 400, 'ftol': 1e-14,
                                    'gtol': 1e-12, 'maxfun': 6000})
            v = _hard(res.x)
            if v < best_val:
                best_val = v
                best_h = res.x.copy()
        except Exception:
            pass

    # ---- Sanity / validity ----
    if not np.all(np.isfinite(best_h)):
        best_h = np.ones(N)
        best_val = _hard(best_h)
    if abs(best_h.sum()) < 1e-10:
        best_h = np.ones(N)
        best_val = _hard(best_h)

    return list(best_h)
