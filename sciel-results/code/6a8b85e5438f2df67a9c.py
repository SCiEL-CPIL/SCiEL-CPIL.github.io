# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _clip_and_normalize(P):
    """Clip negative values to 0 and normalize to sum 1."""
    P = np.clip(P, 0.0, None)
    total = P.sum()
    if total <= 1e-9:
        return np.ones((10, 10)) / 100.0
    return P / total


def _compute_ratio(P):
    """Compute D/M ratio for a given 10x10 distribution P[y, x]."""
    n = 10
    xs = np.arange(n)
    ys = np.arange(n)
    X, Y = np.meshgrid(xs, ys)  # X[y,x] = x, Y[y,x] = y
    
    # D: H of x - y (slope -1)
    vals_d = X - Y  # values from -9 to 9
    counts_d = np.zeros(19)
    for v in range(19):
        mask = (vals_d == v - 9)
        counts_d[v] = P[mask].sum()
    H_d = 0.0
    for c in counts_d:
        if c > 0:
            H_d -= c * np.log2(c)
    
    # M: max of H for slopes 0, 1, 2
    max_h = 0.0
    for r in [0, 1, 2]:
        vals = X + r * Y
        min_val = 0
        max_val = (n - 1) + r * (n - 1)
        counts = np.zeros(max_val + 1)
        for v in range(max_val + 1):
            mask = (vals == v)
            counts[v] = P[mask].sum()
        h = 0.0
        for c in counts:
            if c > 0:
                h -= c * np.log2(c)
        max_h = max(max_h, h)
    
    if max_h <= 1e-9:
        return 0.0
    return H_d / max_h


def _gaussian_P(params):
    """
    Build P from 3-param Gaussian ansatz: P(x,y) ∝ exp(-αx² - βy² - γxy).
    params = [α, β, γ]
    """
    n = 10
    alpha, beta, gamma = params
    xs = np.arange(n)
    ys = np.arange(n)
    X, Y = np.meshgrid(xs, ys)
    log_P = -alpha * X**2 - beta * Y**2 - gamma * X * Y
    # Shift to avoid underflow
    log_P -= log_P.max()
    P = np.exp(log_P)
    P = _clip_and_normalize(P)
    return P


def _neg_ratio_gaussian(params):
    """Negative ratio for minimization of Gaussian params."""
    P = _gaussian_P(params)
    return -_compute_ratio(P)


def _optimize_gaussian(deadline):
    """
    Optimize the 3-parameter Gaussian ansatz.
    Returns best P and ratio found.
    """
    start_time = time.time()
    
    # Multiple initializations in 3D space
    inits = [
        [0.1, 0.1, 0.0],
        [0.05, 0.05, 0.0],
        [0.2, 0.2, 0.1],
        [0.2, 0.2, -0.1],
        [0.1, 0.1, 0.5],
        [0.1, 0.1, -0.5],
        [0.3, 0.3, 0.2],
        [0.3, 0.3, -0.2],
        [0.05, 0.05, 0.3],
        [0.05, 0.05, -0.3],
        [0.15, 0.15, 0.15],
        [0.15, 0.15, -0.15],
        [0.25, 0.25, 0.05],
        [0.25, 0.25, -0.05],
        [0.1, 0.2, 0.0],
        [0.2, 0.1, 0.0],
        [0.1, 0.2, 0.1],
        [0.1, 0.2, -0.1],
        [0.2, 0.1, 0.1],
        [0.2, 0.1, -0.1],
    ]
    
    best_P = None
    best_ratio = 0.0
    
    for init in inits:
        if time.time() - start_time > deadline - 30:
            break
        
        try:
            result = minimize(
                _neg_ratio_gaussian,
                np.array(init),
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            
            P_opt = _gaussian_P(result.x)
            ratio_opt = _compute_ratio(P_opt)
            
            if ratio_opt > best_ratio:
                best_ratio = ratio_opt
                best_P = P_opt.copy()
        except Exception:
            pass
    
    if best_P is None:
        return np.ones((10, 10)) / 100.0, 0.0
    
    return best_P, best_ratio


def _compute_ratio_from_d(d):
    """Compute D/M ratio from anti-diagonal weights d."""
    n = 10
    P = np.zeros((n, n))
    xs = np.arange(n)
    ys = np.arange(n)
    X, Y = np.meshgrid(xs, ys)
    vals_d = X - Y
    
    diag_counts = np.zeros(19)
    for k in range(19):
        diag_counts[k] = np.sum(vals_d == k - 9)
    
    d_norm = d / d.sum()
    
    for k in range(19):
        if d_norm[k] > 0:
            mask = (vals_d == k - 9)
            P[mask] = d_norm[k] / diag_counts[k]
    
    total = P.sum()
    if total <= 1e-9:
        return 0.0
    P = P / total
    
    H_d = 0.0
    for k in range(19):
        if d_norm[k] > 0:
            H_d -= d_norm[k] * np.log2(d_norm[k])
    
    max_h = 0.0
    for r in [0, 1, 2]:
        vals = X + r * Y
        min_val = 0
        max_val = (n - 1) + r * (n - 1)
        counts = np.zeros(max_val + 1)
        for v in range(max_val + 1):
            mask = (vals == v)
            counts[v] = P[mask].sum()
        h = 0.0
        for c in counts:
            if c > 0:
                h -= c * np.log2(c)
        max_h = max(max_h, h)
    
    if max_h <= 1e-9:
        return 0.0
    return H_d / max_h


def _optimize_anti_diag(d0, deadline):
    """Optimize anti-diagonal weights using Nelder-Mead."""
    start_time = time.time()
    
    # Initial guess
    d = np.ones(19) / 19.0
    best_d = d.copy()
    best_ratio = _compute_ratio_from_d(d)
    
    # Try multiple initializations
    inits = [
        np.ones(19) / 19.0,
        np.array([1,2,3,4,5,6,7,8,9,10,9,8,7,6,5,4,3,2,1], dtype=float),
        np.array([1,1,1,1,1,10,1,1,1,1,1,1,1,1,1,1,1,1,1], dtype=float),
        np.array([10,9,8,7,6,5,4,3,2,1,2,3,4,5,6,7,8,9,10], dtype=float),
        np.array([1,0,0,0,0,5,0,0,0,0,0,0,0,0,0,0,0,0,1], dtype=float),
    ]
    
    for init in inits:
        if time.time() - start_time > deadline - 60:
            break
        
        d_init = init / init.sum()
        
        try:
            result = minimize(
                lambda d: -_compute_ratio_from_d(d),
                d_init,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            
            d_opt = result.x
            d_opt = np.clip(d_opt, 1e-10, None)
            d_opt = d_opt / d_opt.sum()
            
            ratio_opt = _compute_ratio_from_d(d_opt)
            if ratio_opt > best_ratio:
                best_ratio = ratio_opt
                best_d = d_opt.copy()
        except Exception:
            pass
    
    return best_d, best_ratio


def _d_to_P(d):
    """Convert anti-diagonal weights to full P matrix."""
    n = 10
    P = np.zeros((n, n))
    xs = np.arange(n)
    ys = np.arange(n)
    X, Y = np.meshgrid(xs, ys)
    vals_d = X - Y
    
    diag_counts = np.zeros(19)
    for k in range(19):
        diag_counts[k] = np.sum(vals_d == k - 9)
    
    d_norm = d / d.sum()
    
    for k in range(19):
        if d_norm[k] > 0 and diag_counts[k] > 0:
            mask = (vals_d == k - 9)
            P[mask] = d_norm[k] / diag_counts[k]
    
    total = P.sum()
    if total > 1e-9:
        P = P / total
    else:
        P = np.ones((n, n)) / 100.0
    
    return P


def _full_optimize_from_P(P0, deadline):
    """Do full 100-parameter optimization starting from P0, using coordinate descent."""
    n = 10
    start_time = time.time()
    P = P0.copy()
    best_ratio = _compute_ratio(P)
    best_P = P.copy()
    
    # Coordinate descent with line search
    for iteration in range(500):
        if time.time() - start_time > deadline - 30:
            break
        
        improved = False
        for i in range(n):
            for j in range(n):
                # Try to optimize P[i,j] by adjusting it
                old_val = P[i, j]
                
                # Try a few values
                for scale in [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]:
                    new_val = old_val * scale if old_val > 0 else scale * 0.01
                    P_test = P.copy()
                    P_test[i, j] = new_val
                    P_test = _clip_and_normalize(P_test)
                    r_test = _compute_ratio(P_test)
                    
                    if r_test > best_ratio + 1e-10:
                        P = P_test
                        best_ratio = r_test
                        best_P = P.copy()
                        improved = True
                        break
        
        if not improved:
            break
    
    return best_P, best_ratio


def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Arithmetic Kakeya conjecture (entropy formulation). Construct a joint probability
    distribution P[y, x] of two discrete random variables X, Y supported on an n x n grid
    (here n = 10) so as to maximize the entropy projection ratio

        H(pi_{r_inf}) / max_i H(pi_{r_i}),

    where pi_r(x, y) = x + r * y projects the grid along slope r, H is the Shannon entropy
    (in bits) of the resulting projected distribution, the finite slopes are
    r_1, ..., r_k = 0.0, 1.0, 2.0 and the distinguished slope is r_inf = -1.0.

    Uses Gaussian ansatz, anti-diagonal weights, and full refinement.

    Returns:
        P: np.ndarray of shape (10, 10), non-negative and summing to 1, where P[y, x] is the
            joint probability that X = x and Y = y.
    """
    n = 10
    start_time = time.time()
    deadline = 580.0  # seconds, with safety margin
    
    best_P = np.ones((n, n)) / 100.0
    best_ratio = _compute_ratio(best_P)
    
    # Phase 1: Gaussian ansatz optimization (3 params, fast)
    P_gauss, ratio_gauss = _optimize_gaussian(deadline)
    if ratio_gauss > best_ratio:
        best_ratio = ratio_gauss
        best_P = P_gauss.copy()
    
    # Phase 2: Optimize anti-diagonal weights (19 params)
    d_best, ratio_diag = _optimize_anti_diag(np.ones(19) / 19.0, deadline)
    P_diag = _d_to_P(d_best)
    if ratio_diag > best_ratio:
        best_ratio = ratio_diag
        best_P = P_diag.copy()
    
    # Phase 3: Full 100-parameter refinement from the best structured solution
    P_refined, ratio_refined = _full_optimize_from_P(best_P, deadline)
    if ratio_refined > best_ratio:
        best_P = P_refined
        best_ratio = ratio_refined
    
    # Phase 4: Try more random restarts with anti-diagonal structure
    np.random.seed(42)
    for restart in range(10):
        if time.time() - start_time > deadline - 60:
            break
        
        # Random anti-diagonal initialization
        d_rand = np.random.dirichlet(np.ones(19) * 2)
        d_opt, ratio_opt = _optimize_anti_diag(d_rand, deadline)
        P_opt = _d_to_P(d_opt)
        
        if ratio_opt > best_ratio:
            best_ratio = ratio_opt
            best_P = P_opt
        
        # Quick refinement
        P_ref, ratio_ref = _full_optimize_from_P(best_P, deadline)
        if ratio_ref > best_ratio:
            best_ratio = ratio_ref
            best_P = P_ref
    
    # Phase 5: Try more Gaussian restarts with random inits
    np.random.seed(123)
    for restart in range(15):
        if time.time() - start_time > deadline - 30:
            break
        
        # Random Gaussian initialization
        init = np.random.uniform(0.01, 0.5, size=3)
        init[2] *= np.random.choice([-1, 1])
        
        try:
            result = minimize(
                _neg_ratio_gaussian,
                init,
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            
            P_opt = _gaussian_P(result.x)
            ratio_opt = _compute_ratio(P_opt)
            
            if ratio_opt > best_ratio:
                best_ratio = ratio_opt
                best_P = P_opt.copy()
        except Exception:
            pass
    
    # Final validation
    best_P = _clip_and_normalize(best_P)
    return best_P


# EVOLVE-BLOCK-END
