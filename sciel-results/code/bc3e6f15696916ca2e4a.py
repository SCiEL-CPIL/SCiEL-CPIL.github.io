# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_N = 10
_DEADLINE = 630.0
_START = time.time()


def _remaining():
    return _DEADLINE - (time.time() - _START)


def _shannon_bits(p):
    p = np.asarray(p, dtype=np.float64)
    p = p[p > 1e-18]
    if p.size == 0:
        return 0.0
    return float(-(p * np.log2(p)).sum())


def _compute_entropies(P):
    """Compute D, H0, H1, H2 from probability matrix P (10x10)."""
    n = P.shape[0]
    X = np.arange(n)[None, :]
    Y = np.arange(n)[:, None]
    
    # x - y: diagonals
    diff_vals = (X - Y).astype(int)  # range -(n-1) to (n-1)
    diff_min = -(n - 1)
    num_diffs = 2 * n - 1
    diff_probs = np.zeros(num_diffs)
    for d in range(num_diffs):
        mask = (diff_vals == (d + diff_min))
        diff_probs[d] = P[mask].sum()
    D = _shannon_bits(diff_probs)
    
    # x: columns
    x_probs = P.sum(axis=0)  # shape (n,)
    H0 = _shannon_bits(x_probs)
    
    # x + y
    sum_vals = (X + Y).astype(int)  # 0 to 2n-2
    num_sums = 2 * n - 1
    sum_probs = np.zeros(num_sums)
    for s in range(num_sums):
        mask = (sum_vals == s)
        sum_probs[s] = P[mask].sum()
    H1 = _shannon_bits(sum_probs)
    
    # x + 2y
    x2y_vals = (X + 2.0 * Y).astype(int)  # 0 to 3n-2
    num_x2y = 3 * n - 1
    x2y_probs = np.zeros(num_x2y)
    for v in range(num_x2y):
        mask = (x2y_vals == v)
        x2y_probs[v] = P[mask].sum()
    H2 = _shannon_bits(x2y_probs)
    
    M = max(H0, H1, H2)
    return D, H0, H1, H2, M


def _objective_flat(P):
    P = np.clip(P, 0.0, None)
    s = P.sum()
    if s <= 1e-12:
        return 0.0
    P = P / s
    D, H0, H1, H2, M = _compute_entropies(P)
    if M <= 1e-9:
        return 0.0
    return D / M


def _analytic_solution(n=_N):
    """Per-cell weight 1 / (number of cells sharing its x+2y value)."""
    X = np.arange(n)[None, :]
    Y = np.arange(n)[:, None]
    v = (X + 2.0 * Y).astype(int)
    counts = np.bincount(v.ravel(), minlength=3 * n - 1)
    counts = np.maximum(counts, 1)
    P = 1.0 / counts[v]
    P = P / P.sum()
    return P


def _param_to_P(theta, n=_N):
    """Convert parameter vector to probability matrix via softmax."""
    theta = theta.reshape(n, n)
    theta_max = theta.max()
    log_P = theta - theta_max
    P = np.exp(log_P)
    Z = P.sum()
    if Z <= 1e-12:
        return np.ones((n, n)) / (n * n)
    P = P / Z
    return P


def _objective_from_theta(theta, n=_N):
    P = _param_to_P(theta, n)
    return _objective_flat(P)


def _grad_objective_from_theta(theta, n=_N):
    """Finite-difference gradient of D/M w.r.t. theta."""
    eps = 1e-6
    grad = np.zeros_like(theta)
    for i in range(theta.size):
        tp = theta.copy()
        tm = theta.copy()
        tp.flat[i] += eps
        tm.flat[i] -= eps
        fp = _objective_from_theta(tp, n)
        fm = _objective_from_theta(tm, n)
        grad.flat[i] = (fp - fm) / (2.0 * eps)
    return grad


def _nelder_mead_optimize(theta0, n=_N, maxiter=500, tol=1e-10):
    """Nelder-Mead optimization on theta parameters."""
    try:
        res = minimize(
            lambda t: -_objective_from_theta(t, n),
            theta0,
            method='Nelder-Mead',
            options={'maxiter': maxiter, 'fatol': tol, 'xatol': tol,
                     'adaptive': True}
        )
        if hasattr(res, 'x') and res.x is not None:
            return res.x
    except Exception:
        pass
    return theta0


def _differential_evolution_optimize(n=_N, budget=120.0, popsize=20, maxiter=80):
    """Differential evolution over theta space."""
    try:
        from scipy.optimize import differential_evolution
    except Exception:
        return None
    
    # Bounds for theta: use reasonable range
    bounds = [(-5.0, 5.0)] * (n * n)
    
    t_end = time.time() + budget
    
    def de_objective(theta_flat):
        if time.time() > t_end:
            return 0.0
        return -_objective_from_theta(theta_flat, n)
    
    try:
        res = differential_evolution(
            de_objective,
            bounds,
            maxiter=maxiter,
            popsize=popsize,
            tol=1e-8,
            seed=42,
            polish=True
        )
        if hasattr(res, 'x') and res.x is not None:
            return res.x
    except Exception:
        pass
    return None


def _generate_diverse_starts(n=_N, rng=None):
    """Generate diverse starting points for optimization."""
    if rng is None:
        rng = np.random.RandomState(42)
    
    starts = []
    
    # 1. Analytic solution
    P0 = _analytic_solution(n)
    starts.append(np.log(P0 + 1e-15).ravel())
    
    # 2. Uniform
    P_uni = np.ones((n, n)) / (n * n)
    starts.append(np.log(P_uni + 1e-15).ravel())
    
    # 3. Dirichlet-like: random positive weights
    for _ in range(30):
        w = rng.dirichlet(np.ones(n * n) * 0.5)
        starts.append(np.log(w + 1e-15))
    
    # 4. Alpha-scaled analytic: P ~ (analytic)^alpha
    P0 = _analytic_solution(n)
    for alpha in [0.1, 0.25, 0.5, 0.75, 1.5, 2.0, 3.0, 5.0]:
        P_alpha = P0 ** alpha
        P_alpha = P_alpha / P_alpha.sum()
        starts.append(np.log(P_alpha + 1e-15).ravel())
    
    # 5. Edge-weighted patterns
    X = np.arange(n)[None, :]
    Y = np.arange(n)[:, None]
    
    # Concentrate on corners
    for corner in [(0,0), (0,n-1), (n-1,0), (n-1,n-1)]:
        P_c = np.zeros((n, n))
        P_c[corner] = 1.0
        # Add some spread
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                cy, cx = corner[0]+dy, corner[1]+dx
                if 0 <= cy < n and 0 <= cx < n:
                    P_c[cy, cx] += 0.5
        P_c = P_c / P_c.sum()
        starts.append(np.log(P_c + 1e-15).ravel())
    
    # 6. Cross/diagonal patterns
    # Emphasize main diagonal (x=y, so x-y=0)
    P_diag = np.zeros((n, n))
    for i in range(n):
        P_diag[i, i] = 1.0
    P_diag = P_diag / P_diag.sum()
    starts.append(np.log(P_diag + 1e-15).ravel())
    
    # Anti-diagonal
    P_adiag = np.zeros((n, n))
    for i in range(n):
        P_adiag[i, n-1-i] = 1.0
    P_adiag = P_adiag / P_adiag.sum()
    starts.append(np.log(P_adiag + 1e-15).ravel())
    
    # 7. Random smooth patterns
    for _ in range(10):
        # Low-frequency random
        fx = rng.randn(n)
        fy = rng.randn(n)
        P_rand = np.outer(fy, fx)
        P_rand = np.abs(P_rand) + 0.1
        P_rand = P_rand / P_rand.sum()
        starts.append(np.log(P_rand + 1e-15).ravel())
    
    return starts


def _optimize(n=_N, budget=450.0):
    """Main optimization routine."""
    rng = np.random.RandomState(42)
    
    # Start with analytic solution
    P0 = _analytic_solution(n)
    best_P = P0.copy()
    best_val = _objective_flat(P0)
    
    t_end = time.time() + budget
    
    # Phase 1: Nelder-Mead from diverse starts
    starts = _generate_diverse_starts(n, rng)
    
    for idx, theta0 in enumerate(starts):
        if time.time() > t_end - 60.0:
            break
        
        # Adaptive maxiter based on remaining time
        time_per_start = max(2.0, min(15.0, (t_end - time.time()) / max(1, len(starts) - idx)))
        maxiter = int(max(100, min(500, time_per_start * 30)))
        
        theta_opt = _nelder_mead_optimize(theta0, n, maxiter=maxiter, tol=1e-10)
        P_opt = _param_to_P(theta_opt, n)
        val = _objective_flat(P_opt)
        
        if val > best_val + 1e-9:
            best_val = val
            best_P = P_opt.copy()
    
    # Phase 2: Differential evolution if time allows
    if time.time() < t_end - 120.0:
        de_budget = min(120.0, t_end - time.time() - 60.0)
        if de_budget > 30.0:
            theta_de = _differential_evolution_optimize(n, budget=de_budget, popsize=20, maxiter=80)
            if theta_de is not None:
                P_de = _param_to_P(theta_de, n)
                val_de = _objective_flat(P_de)
                if val_de > best_val + 1e-9:
                    best_val = val_de
                    best_P = P_de.copy()
    
    # Phase 3: Final Nelder-Mead refinement from best
    if time.time() < t_end - 30.0:
        theta_best = np.log(best_P + 1e-15).ravel()
        theta_refined = _nelder_mead_optimize(theta_best, n, maxiter=300, tol=1e-11)
        P_refined = _param_to_P(theta_refined, n)
        val_ref = _objective_flat(P_refined)
        if val_ref > best_val + 1e-9:
            best_val = val_ref
            best_P = P_refined.copy()
    
    return best_P, best_val


def arithmetic_kakeya_distribution() -> np.ndarray:
    n = 10
    
    # Guaranteed valid baseline
    P = _analytic_solution(n)
    best_val = _objective_flat(P)
    
    # Attempt optimization if there's enough time
    if _remaining() > 100.0:
        try:
            budget = min(450.0, _remaining() - 100.0)
            P2, val2 = _optimize(n, budget=budget)
            if P2 is not None and val2 > best_val + 1e-9:
                P = P2
                best_val = val2
        except Exception:
            pass
    
    # Enforce the public contract.
    P = np.clip(P, 0.0, None)
    s = P.sum()
    if s <= 1e-9:
        P = np.ones((n, n), dtype=np.float64) / (n * n)
    else:
        P = P / s
    P = P.astype(np.float64)
    if not np.all(np.isfinite(P)) or P.shape != (n, n) or P.sum() <= 1e-9:
        P = np.ones((n, n), dtype=np.float64) / (n * n)
    return P


# EVOLVE-BLOCK-END
