import numpy as np
from scipy.optimize import minimize
import time
import math

def _cma_es_minimize(objective, x0, sigma0, max_evals, seed=42):
    """
    Simple CMA-ES implementation for 3D optimization.
    """
    rng = np.random.default_rng(seed)
    n = len(x0)
    best_x = x0.copy()
    best_f = objective(x0)
    mean = x0.copy()
    sigma = sigma0
    C = np.eye(n)
    weights = np.exp(-np.arange(1, 11) / (2 * n))
    weights /= weights.sum()
    mu = int((len(weights) - 1) / 2)
    mu_eff = 1 / (weights.sum() ** 2 - 1)
    c_sigma = (mu_eff + 2) / (n + 4 + mu_eff)
    c_c = (4 + mu_eff / n) / (n + 4 + 3 * mu_eff)
    c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
    c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / (n + 2)) / ((n + 2) ** 2 + mu_eff))
    chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
    p_sigma = np.zeros(n)
    p_c = np.zeros(n)
    evals = 1
    popsize = 10
    
    while evals < max_evals:
        # Generate population
        zs = [rng.standard_normal(n) for _ in range(popsize)]
        xs = [mean + sigma * (zs[i] @ C) for i in range(popsize)]
        fs = [objective(x) for x in xs]
        evals += popsize
        
        # Sort
        order = np.argsort(fs)
        xs = [xs[i] for i in order]
        fs = [fs[i] for i in order]
        zs = [zs[i] for i in order]
        
        # Update best
        if fs[0] < best_f:
            best_f = fs[0]
            best_x = xs[0].copy()
        
        # Weighted mean
        m_new = np.zeros(n)
        for i in range(mu):
            m_new += weights[i] * xs[i]
        
        # Update covariance
        v = (m_new - mean) / sigma
        p_c = (1 - c_c) * p_c + c_c * v
        p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * v / chi_n
        C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(p_c, p_c) + (1 - (1 - c_sigma) ** 2) / chi_n ** 2 * C)
        for i in range(mu):
            C += c_mu * weights[i] * np.outer(zs[i], zs[i])
        
        mean = m_new
        sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))
        
        # Check for convergence
        if np.all(np.abs(mean - best_x) < 1e-10) and sigma < 1e-10:
            break
    
    return best_x, best_f

# Precompute grid and Hermite polynomials for speed
# The function g(x) decays like x^12 * e^{-x^2/2}.
# The maximum is likely within |x| < 10.
# Let's use a dense grid.
N_GRID = 4000
X_GRID = np.linspace(-12, 12, N_GRID)
Y_GRID = X_GRID ** 2
EXP_GRID = np.exp(-Y_GRID / 2.0)

# Hermite polynomials evaluated on grid
# H0
H0_GRID = np.ones_like(X_GRID)
# H4
H4_GRID = 16 * Y_GRID**2 - 48 * Y_GRID + 12
# H8
Y2 = Y_GRID**2
Y3 = Y2 * Y_GRID
Y4 = Y2 * Y2
H8_GRID = 256 * Y4 - 4480 * Y3 + 26880 * Y2 - 6720 * Y_GRID + 1680
# H12
Y5 = Y4 * Y_GRID
Y6 = Y3 * Y3
H12_GRID = 4096 * Y6 - 245760 * Y5 + 5529600 * Y4 - 45158400 * Y3 + 164430720 * Y2 - 222772500 * Y_GRID + 47900160

def _compute_bound(c0, c4, c8):
    """
    Compute the upper bound on A(f)*A(Fourier(f)) for given coefficients.
    Returns the raw value (lower is better).
    """
    # Calculate c12 to satisfy g(0) = 0
    # g(0) = c0*H0(0) + c4*H4(0) + c8*H8(0) + c12*H12(0) = 0
    # H0(0)=1, H4(0)=12, H8(0)=1680, H12(0)=47900160
    c12 = -(c0 + 12.0 * c4 + 1680.0 * c8) / 47900160.0
    
    # Construct g(x) on the grid
    g_vals = c0 * H0_GRID + c4 * H4_GRID + c8 * H8_GRID + c12 * H12_GRID
    g_vals *= EXP_GRID
    
    # Calculate sup |x * g(x)|
    xg = X_GRID * g_vals
    max_abs = np.max(np.abs(xg))
    
    # Bound is sqrt(2*pi) * (sup |x g(x)|)^2
    return math.sqrt(2.0 * math.pi) * max_abs * max_abs

def _objective(coeffs):
    """
    Objective function for optimization.
    """
    try:
        return _compute_bound(coeffs[0], coeffs[1], coeffs[2])
    except Exception:
        return 1e10

def construct():
    """
    Find optimal coefficients by numerical optimization.
    Returns an array of three coefficients [c0, c4, c8].
    """
    start_time = time.time()
    deadline = 175.0  # Safety margin before 210s limit
    
    # Initial guess from current best
    x0 = np.array([-1.13, 0.04, 0.000308642])
    best_val = _objective(x0)
    best_x = x0.copy()
    
    # Phase 1: Multi-start CMA-ES global search
    # Use diverse initializations with different sigma values
    cma_starts = [
        (np.array([-1.13, 0.04, 0.000308642]), 0.5, 42),
        (np.array([-1.0, 0.05, 0.0004]), 0.3, 123),
        (np.array([-1.2, 0.03, 0.00025]), 0.3, 456),
        (np.array([-0.9, 0.06, 0.0005]), 0.2, 789),
        (np.array([-1.3, 0.02, 0.0002]), 0.2, 321),
        (np.array([-1.05, 0.045, 0.00035]), 0.1, 654),
    ]
    
    # Allocate evaluation budget across CMA-ES runs
    remaining_time = deadline - (time.time() - start_time)
    time_per_run = remaining_time / len(cma_starts) * 0.7  # 70% for CMA-ES
    
    for i, (init_x, sigma, seed) in enumerate(cma_starts):
        if time.time() > deadline - 30:
            break
        try:
            # Estimate max evals based on time budget
            max_evals = max(200, int(time_per_run * 500))  # ~500 evals/sec estimate
            cma_x, cma_f = _cma_es_minimize(
                _objective, init_x, sigma, max_evals, seed=seed
            )
            if cma_f < best_val:
                best_val = cma_f
                best_x = cma_x.copy()
        except Exception:
            pass
    
    # Phase 2: Local refinement with multiple starting points
    start_points = [
        best_x.copy(),
        np.array([-1.13, 0.04, 0.000308642]),
        np.array([-1.0, 0.05, 0.0004]),
        np.array([-1.2, 0.03, 0.00025]),
        np.array([-0.9, 0.06, 0.0005]),
        np.array([-1.3, 0.02, 0.0002]),
    ]
    
    for sp in start_points:
        if time.time() > deadline - 20:
            break
        try:
            # Use Nelder-Mead for local search
            res = minimize(
                _objective, sp, method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            pass
    
    # Phase 3: Final high-precision polish
    if time.time() < deadline - 5:
        try:
            res = minimize(
                _objective, best_x, method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-15, 'fatol': 1e-17, 'adaptive': True}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            pass
    
    return np.array(best_x)
