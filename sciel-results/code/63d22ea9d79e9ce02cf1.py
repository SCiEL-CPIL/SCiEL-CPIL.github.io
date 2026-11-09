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


def _compute_entropies_vec(P, n):
    """Vectorized entropy computation using precomputed index arrays."""
    # Precomputed mapping arrays (computed once at module level or cached)
    diff_idx, sum_idx, x2y_idx = _get_index_maps(n)
    
    num_diffs = 2 * n - 1
    num_sums = 2 * n - 1
    num_x2y = 3 * n - 1
    
    diff_probs = np.zeros(num_diffs)
    np.add.at(diff_probs, diff_idx, P.ravel())
    D = _shannon_bits(diff_probs)
    
    x_probs = P.sum(axis=0)
    H0 = _shannon_bits(x_probs)
    
    sum_probs = np.zeros(num_sums)
    np.add.at(sum_probs, sum_idx, P.ravel())
    H1 = _shannon_bits(sum_probs)
    
    x2y_probs = np.zeros(num_x2y)
    np.add.at(x2y_probs, x2y_idx, P.ravel())
    H2 = _shannon_bits(x2y_probs)
    
    M = max(H0, H1, H2)
    return D, H0, H1, H2, M


# Cache for index maps
_index_cache = {}

def _get_index_maps(n):
    if n in _index_cache:
        return _index_cache[n]
    X = np.arange(n)[None, :]
    Y = np.arange(n)[:, None]
    diff_idx = (X - Y + (n - 1)).ravel().astype(np.int64)
    sum_idx = (X + Y).ravel().astype(np.int64)
    x2y_idx = (X + 2 * Y).ravel().astype(np.int64)
    _index_cache[n] = (diff_idx, sum_idx, x2y_idx)
    return diff_idx, sum_idx, x2y_idx


def _objective_flat(P):
    P = np.clip(P, 0.0, None)
    s = P.sum()
    if s <= 1e-12:
        return 0.0
    P = P / s
    n = P.shape[0]
    D, H0, H1, H2, M = _compute_entropies_vec(P, n)
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


def _coordinate_ascent(P, n, max_rounds=200, tol=1e-12):
    """
    Direct coordinate ascent on the probability simplex.
    For each coordinate i, move mass from the best donor j to i
    to maximize D/M. This avoids the expensive finite-difference
    gradient in theta-space.
    """
    P = P.copy()
    P = P / P.sum()
    best_val = _objective_flat(P)
    n_coords = n * n
    flat = P.ravel()
    
    for round_idx in range(max_rounds):
        improved = False
        for i in range(n_coords):
            # Try moving a fraction of mass from each other coordinate to i
            if flat[i] < 1e-15:
                # Need to bring mass in
                best_gain = 0.0
                best_j = -1
                best_frac = 0.0
                for j in range(n_coords):
                    if j == i or flat[j] < 1e-12:
                        continue
                    # Try fractions: move 10%, 30%, 50% of j's mass to i
                    for frac in (0.1, 0.3, 0.5):
                        amount = flat[j] * frac
                        P_new = flat.copy()
                        P_new[j] -= amount
                        P_new[i] += amount
                        P_new = np.maximum(P_new, 0)
                        s = P_new.sum()
                        if s < 1e-12:
                            continue
                        P_new = P_new / s
                        val = _objective_flat(P_new)
                        gain = val - best_val
                        if gain > best_gain:
                            best_gain = gain
                            best_j = j
                            best_frac = frac
                if best_j >= 0 and best_gain > tol:
                    amount = flat[best_j] * best_frac
                    flat[best_j] -= amount
                    flat[i] += amount
                    flat = np.maximum(flat, 0)
                    s = flat.sum()
                    if s > 1e-12:
                        flat = flat / s
                    best_val += best_gain
                    improved = True
            else:
                # Try sending mass from i to other coordinates
                best_gain = 0.0
                best_j = -1
                best_frac = 0.0
                for j in range(n_coords):
                    if j == i:
                        continue
                    for frac in (0.1, 0.3, 0.5):
                        amount = flat[i] * frac
                        P_new = flat.copy()
                        P_new[i] -= amount
                        P_new[j] += amount
                        P_new = np.maximum(P_new, 0)
                        s = P_new.sum()
                        if s < 1e-12:
                            continue
                        P_new = P_new / s
                        val = _objective_flat(P_new)
                        gain = val - best_val
                        if gain > best_gain:
                            best_gain = gain
                            best_j = j
                            best_frac = frac
                if best_j >= 0 and best_gain > tol:
                    amount = flat[i] * best_frac
                    flat[i] -= amount
                    flat[best_j] += amount
                    flat = np.maximum(flat, 0)
                    s = flat.sum()
                    if s > 1e-12:
                        flat = flat / s
                    best_val += best_gain
                    improved = True
        
        if not improved:
            break
    
    P = flat.reshape(n, n)
    s = P.sum()
    if s > 1e-12:
        P = P / s
    return P, best_val


def _multiplicative_perturb(P, rng, n, num_trials=50):
    """
    Generate candidates via multiplicative perturbations: P_new ~ P * exp(noise).
    This explores the neighborhood in a different way than softmax shifts.
    """
    best_P = P.copy()
    best_val = _objective_flat(P)
    flat = P.ravel()
    
    for _ in range(num_trials):
        # Random multiplicative perturbation
        scale = rng.uniform(0.3, 1.5)
        noise = rng.randn(n * n) * scale
        P_new = P * np.exp(noise)
        s = P_new.sum()
        if s < 1e-12:
            continue
        P_new = P_new / s
        val = _objective_flat(P_new)
        if val > best_val + 1e-10:
            best_val = val
            best_P = P_new.copy()
    
    return best_P, best_val


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
    for _ in range(20):
        w = rng.dirichlet(np.ones(n * n) * 0.5)
        starts.append(np.log(w + 1e-15))
    
    # 4. Alpha-scaled analytic: P ~ (analytic)^alpha
    P0 = _analytic_solution(n)
    for alpha in [0.1, 0.25, 0.5, 0.75, 1.5, 2.0, 3.0, 5.0]:
        P_alpha = P0 ** alpha
        P_alpha = P_alpha / P_alpha.sum()
        starts.append(np.log(P_alpha + 1e-15).ravel())
    
    # 5. Diagonal and anti-diagonal patterns
    P_diag = np.zeros((n, n))
    for i in range(n):
        P_diag[i, i] = 1.0
    P_diag = P_diag / P_diag.sum()
    starts.append(np.log(P_diag + 1e-15).ravel())
    
    P_adiag = np.zeros((n, n))
    for i in range(n):
        P_adiag[i, n-1-i] = 1.0
    P_adiag = P_adiag / P_adiag.sum()
    starts.append(np.log(P_adiag + 1e-15).ravel())
    
    # 6. Random smooth patterns
    for _ in range(8):
        fx = rng.randn(n)
        fy = rng.randn(n)
        P_rand = np.outer(fy, fx)
        P_rand = np.abs(P_rand) + 0.1
        P_rand = P_rand / P_rand.sum()
        starts.append(np.log(P_rand + 1e-15).ravel())
    
    return starts


def _optimize(n=_N, budget=450.0):
    """Main optimization routine with coordinate ascent + multiplicative perturbations."""
    rng = np.random.RandomState(42)
    
    # Start with analytic solution
    P0 = _analytic_solution(n)
    best_P = P0.copy()
    best_val = _objective_flat(P0)
    
    t_end = time.time() + budget
    
    # Phase 1: Coordinate ascent from diverse starts
    starts = _generate_diverse_starts(n, rng)
    
    for idx, theta0 in enumerate(starts):
        if time.time() > t_end - 60.0:
            break
        
        P_start = _param_to_P(theta0, n)
        
        # Coordinate ascent
        max_rounds = 50 if (time.time() > t_end - 120.0) else 100
        P_ca, val_ca = _coordinate_ascent(P_start, n, max_rounds=max_rounds, tol=1e-11)
        if val_ca > best_val + 1e-9:
            best_val = val_ca
            best_P = P_ca.copy()
        
        # Nelder-Mead refinement on the coordinate-ascent result
        if time.time() < t_end - 30.0:
            theta_ca = np.log(P_ca + 1e-15).ravel()
            theta_nm = _nelder_mead_optimize(theta_ca, n, maxiter=200, tol=1e-10)
            P_nm = _param_to_P(theta_nm, n)
            val_nm = _objective_flat(P_nm)
            if val_nm > best_val + 1e-9:
                best_val = val_nm
                best_P = P_nm.copy()
    
    # Phase 2: Multiplicative perturbations from current best
    if time.time() < t_end - 30.0:
        P_mp, val_mp = _multiplicative_perturb(best_P, rng, n, num_trials=100)
        if val_mp > best_val + 1e-9:
            best_val = val_mp
            best_P = P_mp.copy()
        
        # Coordinate ascent from multiplicative perturbation result
        P_ca2, val_ca2 = _coordinate_ascent(P_mp, n, max_rounds=80, tol=1e-11)
        if val_ca2 > best_val + 1e-9:
            best_val = val_ca2
            best_P = P_ca2.copy()
    
    # Phase 3: Final coordinate ascent + Nelder-Mead from best
    if time.time() < t_end - 20.0:
        P_final, val_final = _coordinate_ascent(best_P, n, max_rounds=100, tol=1e-11)
        if val_final > best_val + 1e-9:
            best_val = val_final
            best_P = P_final.copy()
        
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
            budget = min(500.0, _remaining() - 50.0)
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
