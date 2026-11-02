"""Tuned Fourier-invariant Hermite-Gaussian test function.

Minimizes the upper bound on A(f)*A(Fourier(f)) = max_{lambda>=0} (1-g(lambda))/g(lambda)
over three free Hermite coefficients (degrees 0, 4, 8). The degree-12 weight is
fixed so that g(0) = 0.
"""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np
from scipy.optimize import minimize, brentq

_SEED = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
_START = time.time()
_DEADLINE = 210.0
_MARGIN = 30.0


def _remaining():
    return _DEADLINE - _MARGIN - (time.time() - _START)


def _hermite_phys(n, x):
    """Physicists' Hermite polynomial H_n(x) via recurrence. Vectorized."""
    x = np.asarray(x, dtype=float)
    if n == 0:
        return np.ones_like(x)
    if n == 1:
        return 2.0 * x
    h0 = np.ones_like(x)
    h1 = 2.0 * x
    for k in range(2, n + 1):
        h2 = 2.0 * x * h1 - 2.0 * (k - 1) * h0
        h0, h1 = h1, h2
    return h1


def _g_of_lambda(a, lam):
    """g(lambda) = (a0 H0 + a4 H4 + a8 H8 + a12 H12)(lambda) * exp(-lambda^2/2),
    with a12 chosen so g(0) = 0."""
    a0, a4, a8 = a
    H0_0 = 1.0
    H4_0 = 12.0
    H8_0 = 1680.0
    H12_0 = 798336.0
    a12 = -(a0 * H0_0 + a4 * H4_0 + a8 * H8_0) / H12_0
    
    lam = np.asarray(lam, dtype=float)
    h0 = np.ones_like(lam)
    h4 = _hermite_phys(4, lam)
    h8 = _hermite_phys(8, lam)
    h12 = _hermite_phys(12, lam)
    poly = a0 * h0 + a4 * h4 + a8 * h8 + a12 * h12
    return poly * np.exp(-0.5 * lam * lam)


def _g_prime_of_lambda(a, lam):
    """Derivative of g(lambda) with respect to lambda."""
    a0, a4, a8 = a
    H0_0 = 1.0
    H4_0 = 12.0
    H8_0 = 1680.0
    H12_0 = 798336.0
    a12 = -(a0 * H0_0 + a4 * H4_0 + a8 * H8_0) / H12_0
    
    lam = np.asarray(lam, dtype=float)
    h0 = np.ones_like(lam)
    h4 = _hermite_phys(4, lam)
    h8 = _hermite_phys(8, lam)
    h12 = _hermite_phys(12, lam)
    poly = a0 * h0 + a4 * h4 + a8 * h8 + a12 * h12
    
    h0_prime = np.zeros_like(lam)
    h4_prime = 2 * 4 * _hermite_phys(3, lam)
    h8_prime = 2 * 8 * _hermite_phys(7, lam)
    h12_prime = 2 * 12 * _hermite_phys(11, lam)
    poly_prime = a0 * h0_prime + a4 * h4_prime + a8 * h8_prime + a12 * h12_prime
    
    return (poly_prime - lam * poly) * np.exp(-0.5 * lam * lam)


def _find_critical_points(a, lam_min=0.005, lam_max=40.0, n_grid=5000):
    """Find critical points of g(lambda) where g'(lambda)=0."""
    grid = np.linspace(lam_min, lam_max, n_grid)
    g_prime = _g_prime_of_lambda(a, grid)
    
    signs = np.sign(g_prime)
    # Handle zeros
    signs[signs == 0] = 1
    sign_changes = np.where(np.diff(signs) != 0)[0]
    
    critical_points = []
    for idx in sign_changes:
        a_left = grid[idx]
        a_right = grid[idx + 1]
        try:
            root = brentq(lambda x: _g_prime_of_lambda(a, np.array([x]))[0], a_left, a_right, xtol=1e-12, rtol=1e-12)
            if lam_min <= root <= lam_max:
                critical_points.append(root)
        except Exception:
            pass
    
    return critical_points


def _bound_exact(a, grid=None):
    """Exact max of (1-g)/g over lambda>=0 on a dense grid with critical point analysis."""
    if grid is None:
        # Build a very dense grid
        grid = np.concatenate([
            np.linspace(0.0001, 35.0, 300000),
            np.geomspace(1e-5, 50.0, 20000),
        ])
        grid = np.sort(grid)
    
    g = _g_of_lambda(a, grid)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = (1.0 - g) / g
    ratio = np.where(g > 1e-14, ratio, 1e18)
    ratio = np.where(np.isfinite(ratio), ratio, 1e18)
    ratio = np.maximum(ratio, 0.0)
    
    idx = int(np.argmax(ratio))
    val = float(ratio[idx])
    x0 = float(grid[idx])
    
    # Evaluate at critical points of g
    try:
        crit_points = _find_critical_points(a)
        for cp in crit_points:
            g_cp = _g_of_lambda(a, np.array([cp]))[0]
            if g_cp > 1e-14:
                r_cp = (1.0 - g_cp) / g_cp
                if np.isfinite(r_cp) and r_cp > val:
                    val = float(r_cp)
                    x0 = float(cp)
    except Exception:
        pass
    
    # Local refinement around argmax
    if 0.0001 < x0 < 50.0:
        width = max(0.005, min(0.3, x0 * 0.05))
        local = np.linspace(max(0.0001, x0 - width), x0 + width, 20000)
        gl = _g_of_lambda(a, local)
        with np.errstate(divide="ignore", invalid="ignore"):
            rl = (1.0 - gl) / gl
        rl = np.where(gl > 1e-14, rl, 1e18)
        rl = np.where(np.isfinite(rl), rl, 1e18)
        rl = np.maximum(rl, 0.0)
        val = max(val, float(np.max(rl)))
    
    return val


def _bound_smooth(a, grid=None, temperature=0.003):
    """Smooth log-sum-exp surrogate for max (1-g)/g."""
    if grid is None:
        grid = np.concatenate([
            np.linspace(0.0005, 35.0, 150000),
            np.geomspace(1e-4, 45.0, 10000),
        ])
        grid = np.sort(grid)
    g = _g_of_lambda(a, grid)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = (1.0 - g) / g
    ratio = np.where(g > 1e-14, ratio, 1e18)
    ratio = np.where(np.isfinite(ratio), ratio, 1e18)
    ratio = np.clip(ratio, 1e-12, 1e15)
    m = float(np.max(ratio))
    s = np.exp((ratio - m) / temperature)
    softmax = m + temperature * math.log(float(np.mean(s)))
    return softmax


def _cma_es_minimize(objective, x0, sigma0=0.05, max_iter=200, popsize=None, seed=42):
    """Simple CMA-ES implementation for 3D optimization."""
    rng = np.random.default_rng(seed)
    n = len(x0)
    x0 = np.asarray(x0, dtype=float)
    if popsize is None:
        popsize = max(4, int(4 + 3 * math.log(n)))
    
    # CMA-ES parameters
    cc = (4 + n / 4) / (n + 4 + 2)
    cs = (n + 1.3) / (n + 3 + 2 * cc)
    ds = 1 + 2 * max(0, math.sqrt((cc - 1) / (cc + 1)) - 1) + cs
    alpha_mu = 0.5
    mu = popsize // 2
    weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    weights /= np.sum(weights)
    mu_eff = 1.0 / np.sum(weights ** 2)
    c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
    c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
    c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
    c_mu = min(1 - c_1, 2 * (alpha_mu * mu_eff - 2) / ((n + 2) ** 2 + mu_eff))
    chi_n = math.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
    
    # State variables
    x_mean = x0.copy()
    sigma = sigma0
    B = np.eye(n)
    D = np.ones(n)
    p_sigma = np.zeros(n)
    p_c = np.zeros(n)
    
    best_x = x0.copy()
    best_f = objective(x0)
    
    for iteration in range(max_iter):
        if _remaining() < 5.0:
            break
        
        # Generate population
        pop = []
        for i in range(popsize):
            z = rng.standard_normal(n)
            y = B @ (D * z)
            x_i = x_mean + sigma * y
            pop.append((x_i, y, z))
        
        # Evaluate population
        fvals = []
        for x_i, _, _ in pop:
            try:
                f = objective(x_i)
                if not np.isfinite(f):
                    f = 1e10
            except Exception:
                f = 1e10
            fvals.append(f)
        
        # Sort by fitness
        order = np.argsort(fvals)
        pop_sorted = [pop[i] for i in order]
        fvals_sorted = [fvals[i] for i in order]
        
        # Update best
        if fvals_sorted[0] < best_f:
            best_f = fvals_sorted[0]
            best_x = pop_sorted[0][0].copy()
        
        # Compute weighted mean of top mu
        x_new = np.zeros(n)
        y_new = np.zeros(n)
        for i in range(mu):
            x_new += weights[i] * pop_sorted[i][0]
            y_new += weights[i] * pop_sorted[i][1]
        
        # Update step size
        p_sigma = (1 - c_sigma) * p_sigma + math.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (B @ (D * y_new))
        
        h_sigma = 1.0 if np.sqrt(np.sum(p_sigma ** 2) / (1 - (1 - c_sigma) ** (2 * (iteration + 1)))) / chi_n < 1.4 + 2 / (n + 1) else 0.0
        sigma *= math.exp(c_sigma / chi_n * (np.sqrt(np.sum(p_sigma ** 2)) / chi_n - 1.0))
        sigma = max(sigma, 1e-10)
        sigma = min(sigma, 1.0)
        
        # Update covariance matrix
        p_c = (1 - c_c) * p_c + h_sigma * math.sqrt(c_c * (2 - c_c) * mu_eff) * y_new
        C = B @ (D ** 2) @ B.T
        C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(p_c, p_c) + (1 - h_sigma) * c_c * (2 - c_c) * C)
        for i in range(mu):
            y_i = pop_sorted[i][1]
            C += c_mu * weights[i] * np.outer(y_i, y_i)
        
        # Eigen decomposition for B and D
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(C)
            eigenvalues = np.maximum(eigenvalues, 1e-12)
            D = np.sqrt(eigenvalues)
            B = eigenvectors
        except Exception:
            B = np.eye(n)
            D = np.ones(n) * sigma
            sigma = sigma0 * 0.5
        
        x_mean = x_new
        
        # Check convergence
        if sigma < 1e-8:
            break
    
    return best_x, best_f


def construct():
    global _START
    _START = time.time()
    best_a = _SEED.copy()
    best_val = _bound_exact(_SEED)
    
    # Phase 1: Multi-start L-BFGS-B on smooth surrogate
    rng = np.random.default_rng(42)
    starts = [_SEED.copy()]
    for scale in (0.1, 0.2, 0.4, 0.8, 1.5, 3.0, 5.0):
        starts.append(_SEED + scale * 0.005 * rng.standard_normal(3))
    for _ in range(20):
        starts.append(rng.standard_normal(3) * 0.05)
    # Structured starts based on problem structure
    starts.append(np.array([-1.0, 0.035, 0.00025]))
    starts.append(np.array([-1.2, 0.045, 0.00035]))
    starts.append(np.array([-0.8, 0.025, 0.0002]))
    starts.append(np.array([-1.5, 0.06, 0.0005]))
    starts.append(np.array([-0.5, 0.02, 0.00015]))
    starts.append(np.array([-1.1, 0.04, 0.0003]))
    starts.append(np.array([-1.3, 0.05, 0.0004]))
    starts.append(np.array([-0.9, 0.03, 0.00018]))
    
    for s in starts:
        if _remaining() < 5.0:
            break
        try:
            res = minimize(
                _bound_smooth, s, method="L-BFGS-B",
                options={"maxiter": 500, "ftol": 1e-14, "gtol": 1e-12},
            )
        except Exception:
            continue
        cand = np.asarray(res.x, dtype=float)
        if not np.all(np.isfinite(cand)):
            continue
        v = _bound_exact(cand)
        if v < best_val:
            best_val = v
            best_a = cand.copy()
    
    # Phase 2: Polish with tighter smooth surrogate
    if _remaining() > 5.0:
        try:
            res = minimize(
                _bound_smooth, best_a, method="L-BFGS-B",
                options={"maxiter": 800, "ftol": 1e-15, "gtol": 1e-14},
            )
            cand = np.asarray(res.x, dtype=float)
            if np.all(np.isfinite(cand)):
                v = _bound_exact(cand)
                if v < best_val:
                    best_val = v
                    best_a = cand.copy()
        except Exception:
            pass
    
    # Phase 3: CMA-ES global search on smooth surrogate with multiple initializations
    if _remaining() > 15.0:
        cma_starts = [
            _SEED.copy(),
            best_a.copy(),
            np.array([-1.0, 0.035, 0.00025]),
            np.array([-1.2, 0.045, 0.00035]),
            np.array([-0.8, 0.025, 0.0002]),
            np.array([-1.5, 0.06, 0.0005]),
            np.array([-0.5, 0.02, 0.00015]),
            np.array([-1.1, 0.04, 0.0003]),
            np.array([-1.3, 0.05, 0.0004]),
            np.array([-0.9, 0.03, 0.00018]),
        ]
        cma_sigmas = [0.03, 0.02, 0.05, 0.04, 0.06, 0.08, 0.04, 0.03, 0.05, 0.04]
        cma_seeds = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
        
        for start, sigma, seed in zip(cma_starts, cma_sigmas, cma_seeds):
            if _remaining() < 8.0:
                break
            try:
                cma_x, cma_f = _cma_es_minimize(
                    _bound_smooth, start, sigma0=sigma,
                    max_iter=200, popsize=14, seed=seed
                )
                if np.all(np.isfinite(cma_x)):
                    v = _bound_exact(cma_x)
                    if v < best_val:
                        best_val = v
                        best_a = cma_x.copy()
            except Exception:
                pass
    
    # Phase 4: CMA-ES on exact objective for fine-tuning
    if _remaining() > 12.0:
        cma_seeds_exact = [999, 1000, 1001, 1002]
        for seed in cma_seeds_exact:
            if _remaining() < 5.0:
                break
            try:
                cma_x, cma_f = _cma_es_minimize(
                    _bound_exact, best_a, sigma0=0.006,
                    max_iter=100, popsize=10, seed=seed
                )
                if np.all(np.isfinite(cma_x)):
                    v = _bound_exact(cma_x)
                    if v < best_val:
                        best_val = v
                        best_a = cma_x.copy()
            except Exception:
                pass
    
    # Phase 5: Direct optimization on exact objective
    if _remaining() > 6.0:
        try:
            res = minimize(
                _bound_exact, best_a, method="Nelder-Mead",
                options={"maxiter": 3000, "xatol": 1e-13, "fatol": 1e-15, "adaptive": True},
            )
            cand = np.asarray(res.x, dtype=float)
            if np.all(np.isfinite(cand)):
                v = _bound_exact(cand)
                if v < best_val:
                    best_val = v
                    best_a = cand.copy()
        except Exception:
            pass
    
    # Phase 6: Fine grid polish around best point
    if _remaining() > 4.0:
        step = 0.0002
        for i in (-4, -3, -2, -1, 0, 1, 2, 3, 4):
            for j in (-4, -3, -2, -1, 0, 1, 2, 3, 4):
                for k in (-4, -3, -2, -1, 0, 1, 2, 3, 4):
                    if _remaining() < 2.0:
                        break
                    cand = best_a + step * np.array([i, j, k], dtype=float)
                    v = _bound_exact(cand)
                    if v < best_val:
                        best_val = v
                        best_a = cand.copy()
    
    # Phase 7: Very fine local search
    if _remaining() > 2.5:
        step = 0.00005
        for i in (-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5):
            for j in (-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5):
                for k in (-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5):
                    if _remaining() < 1.0:
                        break
                    cand = best_a + step * np.array([i, j, k], dtype=float)
                    v = _bound_exact(cand)
                    if v < best_val:
                        best_val = v
                        best_a = cand.copy()
    
    # Phase 8: Final ultra-fine polish
    if _remaining() > 1.5:
        step = 0.00001
        for i in (-3, -2, -1, 0, 1, 2, 3):
            for j in (-3, -2, -1, 0, 1, 2, 3):
                for k in (-3, -2, -1, 0, 1, 2, 3):
                    if _remaining() < 0.5:
                        break
                    cand = best_a + step * np.array([i, j, k], dtype=float)
                    v = _bound_exact(cand)
                    if v < best_val:
                        best_val = v
                        best_a = cand.copy()
    
    return np.array(best_a, dtype=float)

# EVOLVE-BLOCK-END
