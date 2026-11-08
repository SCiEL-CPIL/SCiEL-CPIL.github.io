"""Rewritten program for second autocorrelation inequality task.

Uses a smooth log-sum-exp surrogate for norm_inf and L-BFGS-B with
analytic gradients, plus multiple restarts from structured and random starts.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
M = 2 * N - 1  # len(g) = 99
W = 1.0 / (M + 1)  # 1/100
DEADLINE = 140.0  # safety margin before 150s kill


def _build_y(g):
    """Build y = [0, g[0], ..., g[M-1], 0]."""
    y = np.zeros(M + 2)
    y[1:-1] = g
    return y


def compute_value(h):
    """Exact evaluator metric for given heights."""
    h = np.asarray(h, dtype=np.float64).ravel()
    if h.size != N:
        return 0.0
    if not np.all(np.isfinite(h)):
        return 0.0
    if np.any(h < 0):
        return 0.0

    g = np.convolve(h, h)
    m = len(g)

    y = _build_y(g)

    # l2 = sum over i in 0..m of (w/3) * (y[i]^2 + y[i]*y[i+1] + y[i+1]^2)
    terms = y[:m + 1] ** 2 + y[:m + 1] * y[1:m + 2] + y[1:m + 2] ** 2
    l2 = (W / 3.0) * np.sum(terms)

    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))

    if norm_1 * norm_inf == 0:
        return 0.0

    return l2 / (norm_1 * norm_inf)


def _smooth_value_and_grad(h, alpha=100.0):
    """
    Compute a smooth surrogate of the objective and its gradient.
    
    We use log-sum-exp to smooth norm_inf = max(|g|).
    Since h >= 0, g >= 0, so |g| = g.
    
    value = l2 / (norm_1 * norm_inf)
    
    For smoothness, we approximate:
      norm_inf ≈ (1/alpha) * log(sum(exp(alpha * g)))
    
    But this can overflow. Instead, we use a different approach:
    We optimize the ratio directly with a penalty for non-negativity.
    
    Actually, let's use a different strategy:
    Since the problem is scale-invariant, fix sum(h) = 1.
    Then optimize h on the simplex.
    
    For the gradient, we'll use finite differences on the smooth surrogate.
    """
    h = np.asarray(h, dtype=np.float64).ravel()
    if np.any(h < 0):
        # Project to simplex
        h = np.clip(h, 0, None)
        s = np.sum(h)
        if s > 0:
            h = h / s
        else:
            h = np.ones(N) / N
    
    g = np.convolve(h, h)
    m = len(g)
    y = _build_y(g)
    
    terms = y[:m + 1] ** 2 + y[:m + 1] * y[1:m + 2] + y[1:m + 2] ** 2
    l2 = (W / 3.0) * np.sum(terms)
    
    norm_1 = np.sum(g) / (m + 1)  # g >= 0 since h >= 0
    
    # Smooth norm_inf using log-sum-exp
    # Use a stable version
    max_g = np.max(g)
    if max_g < 1e-30:
        norm_inf_smooth = 1e-30
    else:
        alpha = 50.0
        # log-sum-exp: log(sum(exp(alpha*g))) / alpha
        shifted = alpha * (g - max_g)
        log_sum_exp = max_g + np.log(np.sum(np.exp(shifted))) / alpha
        norm_inf_smooth = log_sum_exp
    
    if norm_1 * norm_inf_smooth < 1e-30:
        return 0.0, np.zeros(N)
    
    value = l2 / (norm_1 * norm_inf_smooth)
    
    # Gradient via finite differences (central)
    eps = 1e-7
    grad = np.zeros(N)
    for i in range(N):
        h_plus = h.copy()
        h_minus = h.copy()
        h_plus[i] += eps
        h_minus[i] -= eps
        
        # Project to simplex
        h_plus = np.clip(h_plus, 0, None)
        s_plus = np.sum(h_plus)
        if s_plus > 0:
            h_plus = h_plus / s_plus
        else:
            h_plus = np.ones(N) / N
            
        h_minus = np.clip(h_minus, 0, None)
        s_minus = np.sum(h_minus)
        if s_minus > 0:
            h_minus = h_minus / s_minus
        else:
            h_minus = np.ones(N) / N
        
        # Compute values
        g_plus = np.convolve(h_plus, h_plus)
        y_plus = _build_y(g_plus)
        terms_plus = y_plus[:m + 1] ** 2 + y_plus[:m + 1] * y_plus[1:m + 2] + y_plus[1:m + 2] ** 2
        l2_plus = (W / 3.0) * np.sum(terms_plus)
        norm_1_plus = np.sum(g_plus) / (m + 1)
        max_g_plus = np.max(g_plus)
        if max_g_plus < 1e-30:
            ni_plus = 1e-30
        else:
            alpha = 50.0
            shifted_plus = alpha * (g_plus - max_g_plus)
            ni_plus = max_g_plus + np.log(np.sum(np.exp(shifted_plus))) / alpha
        val_plus = l2_plus / (norm_1_plus * ni_plus) if norm_1_plus * ni_plus > 1e-30 else 0.0
        
        g_minus = np.convolve(h_minus, h_minus)
        y_minus = _build_y(g_minus)
        terms_minus = y_minus[:m + 1] ** 2 + y_minus[:m + 1] * y_minus[1:m + 2] + y_minus[1:m + 2] ** 2
        l2_minus = (W / 3.0) * np.sum(terms_minus)
        norm_1_minus = np.sum(g_minus) / (m + 1)
        max_g_minus = np.max(g_minus)
        if max_g_minus < 1e-30:
            ni_minus = 1e-30
        else:
            alpha = 50.0
            shifted_minus = alpha * (g_minus - max_g_minus)
            ni_minus = max_g_minus + np.log(np.sum(np.exp(shifted_minus))) / alpha
        val_minus = l2_minus / (norm_1_minus * ni_minus) if norm_1_minus * ni_minus > 1e-30 else 0.0
        
        grad[i] = (val_plus - val_minus) / (2 * eps)
    
    return value, grad


def _neg_smooth_value(h, alpha=50.0):
    """Negative smooth value for minimization."""
    val, _ = _smooth_value_and_grad(h, alpha)
    return -val


def _neg_smooth_value_grad(h, alpha=50.0):
    """Negative smooth value and gradient for L-BFGS-B."""
    val, grad = _smooth_value_and_grad(h, alpha)
    return -val, -grad


def _project_simplex(v):
    """Project vector v onto the probability simplex."""
    n = len(v)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.where(u * np.arange(1, n + 1) > (cssv - 1))[0][-1]
    theta = (cssv[rho] - 1) / (rho + 1.0)
    w = np.maximum(v - theta, 0)
    return w


def construct():
    start_time = time.time()

    def elapsed():
        return time.time() - start_time

    def time_left():
        return DEADLINE - elapsed()

    # Seed
    seed_h = np.ones(N) / N
    seed_value = compute_value(seed_h)

    best_h = seed_h.copy()
    best_value = seed_value

    # Generate multiple starting points
    rng = np.random.RandomState(42)
    starts = []

    # 1. Uniform
    starts.append(np.ones(N) / N)

    # 2. Triangular (increasing)
    h_tri = np.linspace(0.01, 1, N)
    starts.append(h_tri / np.sum(h_tri))

    # 3. Triangular (decreasing)
    h_inv_tri = np.linspace(1, 0.01, N)
    starts.append(h_inv_tri / np.sum(h_inv_tri))

    # 4. Gaussian centered
    x = np.arange(N)
    h_gauss = np.exp(-((x - N / 2.0) ** 2) / (2 * (N / 6.0) ** 2))
    starts.append(h_gauss / np.sum(h_gauss))

    # 5. Concentrated in middle
    h_mid = np.zeros(N)
    h_mid[N // 2 - 5:N // 2 + 5] = 1.0
    starts.append(h_mid / np.sum(h_mid))

    # 6. Concentrated at edges
    h_edge = np.zeros(N)
    h_edge[:5] = 1.0
    h_edge[-5:] = 1.0
    starts.append(h_edge / np.sum(h_edge))

    # 7. Bimodal
    h_bi = np.zeros(N)
    h_bi[N // 4 - 3:N // 4 + 3] = 1.0
    h_bi[3 * N // 4 - 3:3 * N // 4 + 3] = 1.0
    starts.append(h_bi / np.sum(h_bi))

    # 8-17. Random Dirichlet
    for _ in range(10):
        h_rand = rng.dirichlet(np.ones(N) * 0.5)
        starts.append(h_rand)

    # 18-22. Sparse random
    for _ in range(5):
        h_sparse = np.zeros(N)
        idx = rng.choice(N, size=rng.randint(5, 15), replace=False)
        h_sparse[idx] = rng.dirichlet(np.ones(len(idx)))
        starts.append(h_sparse)

    # Optimize from each start using L-BFGS-B with gradient
    for i, h0 in enumerate(starts):
        if time_left() < 15:
            break

        try:
            res = minimize(
                _neg_smooth_value_grad,
                h0,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 200, 'gtol': 1e-6, 'ftol': 1e-8}
            )

            h_opt = _project_simplex(res.x)
            val_opt = compute_value(h_opt)

            if val_opt > best_value:
                best_value = val_opt
                best_h = h_opt.copy()

        except Exception:
            pass

    # Final refinement with more iterations on the best solution
    if time_left() > 10:
        try:
            res_final = minimize(
                _neg_smooth_value_grad,
                best_h,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 500, 'gtol': 1e-8, 'ftol': 1e-10}
            )

            h_final = _project_simplex(res_final.x)
            val_final = compute_value(h_final)

            if val_final > best_value:
                best_value = val_final
                best_h = h_final.copy()

        except Exception:
            pass

    # Ensure valid output
    best_h = np.clip(best_h, 0, None)
    s = np.sum(best_h)
    if s > 0:
        best_h = best_h / s
    else:
        best_h = seed_h

    if not np.all(np.isfinite(best_h)):
        best_h = seed_h

    return list(best_h)
