# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_N = 10
_VALS = np.arange(_N)
_YS = np.repeat(_VALS, _N)
_XS = np.tile(_VALS, _N)
_XY = _XS + _YS
_X2Y = _XS + 2 * _YS
_XM1Y = _XS - _YS  # range -9..9
_IDX_XM1Y = _XM1Y + 9 # 0..18

# Global state for annealing
_current_alpha = 1.0

def _entropy(p):
    """Compute Shannon entropy in bits, ignoring zero probabilities."""
    s = p.sum()
    if s <= 1e-30:
        return 0.0
    q = p / s
    q = q[q > 0.0]
    if len(q) == 0:
        return 0.0
    return float(-(q * np.log2(q)).sum())

def _compute_value(P):
    """Compute the ratio D/M for a given distribution P using hard max."""
    return _compute_value_smooth(P, np.inf)

def _compute_value_smooth(P, alpha):
    """Compute the ratio D/M_smooth for a given distribution P."""
    p = P.ravel()
    total = p.sum()
    if total <= 1e-30:
        return 0.0

    # D: distribution of x - y (slope -1), 19 bins
    d_bins = np.bincount(_IDX_XM1Y, weights=p, minlength=19)
    hd = _entropy(d_bins)

    # H_0: distribution of x, 10 bins
    h0_bins = np.bincount(_XS, weights=p, minlength=10)
    h0 = _entropy(h0_bins)

    # H_1: distribution of x + y, 19 bins
    h1_bins = np.bincount(_XY, weights=p, minlength=19)
    h1 = _entropy(h1_bins)

    # H_2: distribution of x + 2y, 28 bins
    h2_bins = np.bincount(_X2Y, weights=p, minlength=28)
    h2 = _entropy(h2_bins)

    if alpha == np.inf:
        m = max(h0, h1, h2)
    else:
        # Soft max: (1/alpha) * log(sum(exp(alpha * Hi)))
        # Scale by alpha to avoid overflow
        h_vals = np.array([h0, h1, h2])
        max_h = np.max(h_vals)
        # If max_h is 0, result is 0
        if max_h < 1e-15:
            m = 0.0
        else:
            # log-sum-exp trick
            exp_terms = np.exp(alpha * (h_vals - max_h))
            log_sum = np.log(np.sum(exp_terms))
            m = (max_h + log_sum / alpha)
            
    if m <= 1e-9:
        return 0.0
    return hd / m

def _build_from_theta(theta):
    """Convert log-probability parameters to a probability distribution."""
    theta_shifted = theta - np.max(theta)
    p = np.exp(theta_shifted)
    s = p.sum()
    if s <= 1e-30:
        return np.ones(_N * _N) / (_N * _N)
    p = p / s
    return p

def _objective_smooth(theta):
    """Negative of the value we want to maximize, using current alpha."""
    p = _build_from_theta(theta)
    P = p.reshape(_N, _N)
    val = _compute_value_smooth(P, _current_alpha)
    return -val

def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Arithmetic Kakeya entropy ratio on a 10x10 grid.
    Uses annealing of the soft-max temperature to handle the non-smooth max term.
    """
    global _current_alpha
    deadline = time.time() + 550
    n_cells = _N * _N
    
    best_P = np.ones((_N, _N), dtype=np.float64) / n_cells
    best_val = _compute_value(best_P)
    
    rng = np.random.RandomState(42)
    
    # Initial best theta
    best_theta = np.zeros(n_cells)
    
    # Annealing schedule
    alphas = [1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0]
    
    for alpha in alphas:
        if time.time() > deadline - 30:
            break
            
        _current_alpha = alpha
        
        # Try a few restarts for each alpha, or just refine the best so far?
        # Refining the best so far is efficient.
        # But if we are stuck, a restart might help.
        # Let's do 2 restarts per alpha: one from best_theta, one random.
        
        for restart in range(2):
            if time.time() > deadline - 10:
                break
                
            if restart == 0:
                theta0 = best_theta.copy()
            else:
                # Random restart
                theta0 = rng.randn(n_cells) * 1.0
                
            result = minimize(
                _objective_smooth, theta0, method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            p_opt = _build_from_theta(result.x)
            P_opt = p_opt.reshape(_N, _N)
            
            # Evaluate with HARD max to see true score
            val_opt = _compute_value(P_opt)
            
            if val_opt > best_val:
                best_val = val_opt
                best_P = P_opt.copy()
                best_theta = result.x.copy()
                
    # Final polish with very high alpha (effectively hard max)
    if time.time() < deadline - 10:
        _current_alpha = 1000.0
        result = minimize(
            _objective_smooth, best_theta, method='L-BFGS-B',
            options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12}
        )
        p_opt = _build_from_theta(result.x)
        P_opt = p_opt.reshape(_N, _N)
        val_opt = _compute_value(P_opt)
        if val_opt > best_val:
            best_val = val_opt
            best_P = P_opt.copy()

    # Final cleanup
    best_P = np.clip(best_P, 0, None)
    s = best_P.sum()
    if s <= 1e-9:
        best_P = np.ones((_N, _N), dtype=np.float64) / n_cells
    else:
        best_P /= s
    
    return best_P

# EVOLVE-BLOCK-END
