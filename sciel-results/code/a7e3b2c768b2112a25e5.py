"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.special import roots_laguerre

# Global deadline guard
_T0 = time.time()
_DEADLINE = 630.0  # seconds wall-clock

def _elapsed():
    return time.time() - _T0

def _time_left():
    return _DEADLINE - _elapsed() - 30.0  # 30s safety margin


def _laguerre_coeffs(m):
    """Return coefficients of L_m(x) (highest degree first)."""
    L_prev = np.array([1.0])  # L_0
    L_curr = np.array([1.0, -1.0])  # L_1
    if m == 0:
        return L_prev
    if m == 1:
        return L_curr
    for n in range(1, m):
        xLn = np.concatenate([L_curr, [0.0]])
        factor_Ln = (2*n + 1) * L_curr
        pad = len(xLn) - len(factor_Ln)
        if pad > 0:
            factor_Ln = np.concatenate([np.zeros(pad), factor_Ln])
        nLprev = n * L_prev
        pad2 = len(xLn) - len(nLprev)
        if pad2 > 0:
            nLprev = np.concatenate([np.zeros(pad2), nLprev])
        L_next = (factor_Ln - xLn - nLprev) / (n + 1)
        L_prev = L_curr
        L_curr = L_next
    return L_curr


def _poly_eval_all(coeffs, xs):
    """Evaluate polynomial at multiple points."""
    return np.polyval(coeffs, xs)


def _compute_C4(roots, m=12, n_dim=1):
    """
    Compute the upper bound C4 for the Fourier sign uncertainty.
    """
    r = np.array(roots, dtype=float)
    if len(r) != m:
        return float('inf')
    if np.any(r <= 0):
        return float('inf')
    rs = np.sort(r)
    if np.any(np.diff(rs) < 1e-10):
        return float('inf')  # roots must be distinct

    # Compute L_m at each root
    Lm_coeffs = _laguerre_coeffs(m)
    Lm_vals = _poly_eval_all(Lm_coeffs, r)

    prod_r = np.prod(r)
    sum_term = np.sum(r * Lm_vals**2)

    if sum_term < 1e-15:
        return float('inf')

    C4 = prod_r / sum_term
    return C4


def _score(roots):
    """Compute the score (upper bound on C4). Lower is better."""
    try:
        r = np.array(roots, dtype=float)
        if len(r) != 12:
            return float('inf')
        if np.any(r <= 0):
            return float('inf')
        rs = np.sort(r)
        if np.any(np.diff(rs) < 1e-8):
            return float('inf')
        return _compute_C4(r)
    except Exception:
        return float('inf')


def _cost_geometric(params):
    """Cost function for geometric parameterization. params = [log_r0, log_q]."""
    log_r0, log_q = params
    r0 = np.exp(log_r0)
    q = np.exp(log_q)
    if r0 <= 0 or q <= 0:
        return 1e10
    roots = r0 * (q ** np.arange(12))
    if np.any(roots <= 0):
        return 1e10
    rs = np.sort(roots)
    if np.any(np.diff(rs) < 1e-8):
        return 1e10
    if np.max(roots) > 1e6:
        return 1e10
    return _score(roots)


def _cost_direct(roots):
    """Cost function for direct root optimization."""
    try:
        r = np.array(roots, dtype=float)
        if len(r) != 12:
            return 1e10
        if np.any(r <= 0):
            return 1e10
        rs = np.sort(r)
        if np.any(np.diff(rs) < 1e-8):
            return 1e10
        if np.max(r) > 1e6:
            return 1e10
        return _score(r)
    except Exception:
        return 1e10


def construct():
    """
    Construct 12 positive double-root locations for the Laguerre combination (m=12, n=1).
    Returns a list of 12 positive root locations.
    """
    # Seed solution: geometric with r0=4, q=2
    seed_roots = [4.0 * (2 ** i) for i in range(12)]
    best_score = _score(seed_roots)
    best_roots = seed_roots[:]

    if _time_left() < 5.0:
        return best_roots

    # Strategy 1: Global optimization over geometric parameters using differential evolution
    # Then local polish with L-BFGS-B
    if _time_left() > 60.0:
        try:
            # Define bounds for log_r0 and log_q
            # r0 in [0.1, 100], q in [0.5, 5]
            bounds = [(np.log(0.1), np.log(100.0)), (np.log(0.5), np.log(5.0))]
            
            de_result = differential_evolution(
                _cost_geometric,
                bounds=bounds,
                seed=42,
                maxiter=100,
                tol=1e-10,
                popsize=15,
                init='latinhypercube'
            )
            
            if de_result.fun < best_score:
                log_r0_opt, log_q_opt = de_result.x
                r0_opt = np.exp(log_r0_opt)
                q_opt = np.exp(log_q_opt)
                roots_opt = r0_opt * (q_opt ** np.arange(12))
                
                if np.all(roots_opt > 0) and np.all(np.diff(np.sort(roots_opt)) > 1e-8):
                    s = _score(roots_opt)
                    if s < best_score:
                        best_score = s
                        best_roots = roots_opt.tolist()
        except Exception:
            pass

    # Strategy 2: Local polish with L-BFGS-B from multiple starting points
    if _time_left() > 30.0:
        # Starting points: seed, DE result, and perturbations
        start_points = []
        
        # Seed
        start_points.append((np.log(4.0), np.log(2.0)))
        
        # Perturbations around seed
        for dr0 in [0.5, 0.8, 1.2, 1.5, 2.0, 3.0]:
            for dq in [0.5, 0.7, 0.9, 1.1, 1.3, 1.5]:
                start_points.append((np.log(4.0) + np.log(dr0), np.log(2.0) + np.log(dq)))
        
        # Broader exploration
        for r0_val in [1.0, 2.0, 5.0, 10.0, 20.0]:
            for q_val in [1.01, 1.1, 1.3, 1.5, 1.8]:
                start_points.append((np.log(r0_val), np.log(q_val)))
        
        # Add current best if it's geometric
        logs_best = np.log(best_roots)
        A = np.vstack([np.ones(12), np.arange(12)]).T
        coeffs, residuals, rank, sv = np.linalg.lstsq(A, logs_best, rcond=None)
        pred = A @ coeffs
        if np.max(np.abs(pred - logs_best)) < 0.5:
            start_points.append(tuple(coeffs))
        
        # Shuffle with fixed seed
        rng = np.random.RandomState(42)
        rng.shuffle(start_points)
        
        for idx, (lr0, lq) in enumerate(start_points):
            if _time_left() < 10.0:
                break
            
            try:
                result = minimize(
                    _cost_geometric,
                    x0=np.array([lr0, lq]),
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                
                if result.success or result.fun < best_score:
                    lr0_opt, lq_opt = result.x
                    r0_opt = np.exp(lr0_opt)
                    q_opt = np.exp(lq_opt)
                    roots_opt = r0_opt * (q_opt ** np.arange(12))
                    
                    if np.all(roots_opt > 0) and np.all(np.diff(np.sort(roots_opt)) > 1e-8):
                        s = _score(roots_opt)
                        if s < best_score:
                            best_score = s
                            best_roots = roots_opt.tolist()
            except Exception:
                continue

    # Strategy 3: Direct root optimization (non-geometric) as fallback
    if _time_left() > 20.0:
        try:
            # Start from current best and perturb
            r_best = np.array(best_roots)
            
            # Try Nelder-Mead on direct roots
            result = minimize(
                _cost_direct,
                x0=r_best,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            
            if result.fun < best_score:
                r_opt = result.x
                if np.all(r_opt > 0) and np.all(np.diff(np.sort(r_opt)) > 1e-8):
                    s = _score(r_opt)
                    if s < best_score:
                        best_score = s
                        best_roots = r_opt.tolist()
        except Exception:
            pass

    # Final refinement: try to fit geometric to best and polish
    if _time_left() > 10.0:
        try:
            logs = np.log(best_roots)
            A = np.vstack([np.ones(12), np.arange(12)]).T
            coeffs, residuals, rank, sv = np.linalg.lstsq(A, logs, rcond=None)
            lr0_fit, lq_fit = coeffs
            pred = A @ coeffs
            if np.max(np.abs(pred - logs)) < 0.5:
                result = minimize(
                    _cost_geometric,
                    x0=np.array([lr0_fit, lq_fit]),
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12}
                )
                if result.fun < best_score:
                    lr0_opt, lq_opt = result.x
                    r0_opt = np.exp(lr0_opt)
                    q_opt = np.exp(lq_opt)
                    roots_opt = r0_opt * (q_opt ** np.arange(12))
                    if np.all(roots_opt > 0) and np.all(np.diff(np.sort(roots_opt)) > 1e-8):
                        s = _score(roots_opt)
                        if s < best_score:
                            best_score = s
                            best_roots = roots_opt.tolist()
        except Exception:
            pass

    # Ensure we return a valid list of 12 positive floats
    if len(best_roots) != 12 or np.any(np.array(best_roots) <= 0):
        return [4.0 * (2 ** i) for i in range(12)]
    
    return [float(r) for r in best_roots]
# EVOLVE-BLOCK-END
