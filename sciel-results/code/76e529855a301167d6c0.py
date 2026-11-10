import numpy as np
import time
from scipy.optimize import minimize

# Precompute the value arrays for the four slopes
_N = 10
_y_idx, _x_idx = np.indices((_N, _N))
_x_flat = _x_idx.flatten()
_y_flat = _y_idx.flatten()

# For each slope, precompute the bin index for each cell
# D: x - y, range -9..9 -> shift by 9 to get 0..18
_d_vals = (_x_flat - _y_flat + _N - 1).astype(np.int64)
# H0: x, range 0..9
_h0_vals = _x_flat.astype(np.int64)
# H1: x + y, range 0..18
_h1_vals = (_x_flat + _y_flat).astype(np.int64)
# H2: x + 2y, range 0..27
_h2_vals = (_x_flat + 2 * _y_flat).astype(np.int64)

# Number of bins for each
_N_D = _d_vals.max() + 1
_N_H0 = _h0_vals.max() + 1
_N_H1 = _h1_vals.max() + 1
_N_H2 = _h2_vals.max() + 1

# Precompute pooling matrices (sparse-style, but dense for 100 cells)
# Each matrix maps flat P to pooled probabilities
def _make_pooling_matrix(vals, n_bins):
    M = np.zeros((n_bins, _N * _N), dtype=np.float64)
    for i in range(_N * _N):
        M[vals[i], i] = 1.0
    return M

_M_D = _make_pooling_matrix(_d_vals, _N_D)
_M_H0 = _make_pooling_matrix(_h0_vals, _N_H0)
_M_H1 = _make_pooling_matrix(_h1_vals, _N_H1)
_M_H2 = _make_pooling_matrix(_h2_vals, _N_H2)

def _entropy_from_counts(counts):
    """Compute Shannon entropy in bits from a count array."""
    probs = counts[counts > 1e-15]
    if len(probs) == 0:
        return 0.0
    return -np.sum(probs * np.log2(probs))

def _score_fast_impl(w):
    """Fast score computation assuming w is a flat array of non-negative weights summing to 1."""
    d_probs = _M_D @ w
    h0_probs = _M_H0 @ w
    h1_probs = _M_H1 @ w
    h2_probs = _M_H2 @ w
    
    D = _entropy_from_counts(d_probs)
    H0 = _entropy_from_counts(h0_probs)
    H1 = _entropy_from_counts(h1_probs)
    H2 = _entropy_from_counts(h2_probs)
    
    M = max(H0, H1, H2)
    
    if M <= 1e-9:
        return 0.0
        
    return D / M

def _compute_ratio(P):
    P = np.clip(P, 0, None)
    s = P.sum()
    if s <= 1e-15:
        return 0.0
    w = P.flatten() / s
    return _score_fast_impl(w)

def _smooth_max_entropy(H0, H1, H2, temperature=1.0):
    """Soft max using log-sum-exp for smoothness."""
    scaled = np.array([H0, H1, H2]) / temperature
    max_scaled = np.max(scaled)
    soft_max = (max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))) * temperature
    return soft_max

def _negative_ratio_smooth(params, temperature=0.5):
    """Smooth objective using soft-max for M."""
    # Convert from unconstrained params to probabilities via softmax
    log_p = params
    max_log_p = np.max(log_p)
    log_probs = log_p - max_log_p
    exp_probs = np.exp(log_probs)
    probs = exp_probs / np.sum(exp_probs)
    
    d_probs = _M_D @ probs
    h0_probs = _M_H0 @ probs
    h1_probs = _M_H1 @ probs
    h2_probs = _M_H2 @ probs
    
    D = _entropy_from_counts(d_probs)
    H0 = _entropy_from_counts(h0_probs)
    H1 = _entropy_from_counts(h1_probs)
    H2 = _entropy_from_counts(h2_probs)
    
    M = _smooth_max_entropy(H0, H1, H2, temperature)
    
    if M <= 1e-9:
        return 0.0
        
    return -D / M

def arithmetic_kakeya_distribution() -> np.ndarray:
    n = _N
    P = np.ones((n, n), dtype=np.float64) / (n * n)
    
    start_time = time.time()
    timeout = 630
    safety_margin = 15
    
    best_score = _compute_ratio(P)
    best_P = P.copy()
    
    np.random.seed(42)
    
    def _normalize(P):
        P = np.clip(P, 0, None)
        s = P.sum()
        if s <= 1e-15:
            return None
        return P / s
    
    def _score_fast(P_flat):
        """Fast score computation assuming P_flat is already normalized and non-negative."""
        return _score_fast_impl(P_flat)
    
    def _log_softmax_params_to_prob(params):
        """Convert unconstrained params to probability via log-sum-exp."""
        log_p = params
        max_log_p = np.max(log_p)
        log_probs = log_p - max_log_p
        exp_probs = np.exp(log_probs)
        probs = exp_probs / np.sum(exp_probs)
        return probs
    
    def _negative_ratio_from_params(params):
        """Objective for scipy: minimize negative ratio."""
        probs = _log_softmax_params_to_prob(params)
        return -_score_fast(probs)
    
    # Multi-start with different initializations
    num_restarts = 25
    time_per_restart = (timeout - safety_margin - 30) / num_restarts  # reserve 30s for final polish
    
    for r in range(num_restarts):
        if time.time() - start_time > timeout - safety_margin:
            break
            
        # Initialize current P
        if r == 0:
            current_P = P.copy()
        elif r == 1:
            # Concentrate on x=0
            current_P = np.zeros((n, n))
            current_P[0, :] = 1.0 / n
        elif r == 2:
            # Concentrate on a few x values
            current_P = np.zeros((n, n))
            current_P[0, :] = 0.5 / n
            current_P[9, :] = 0.5 / n
        elif r == 3:
            # Diagonal pattern
            current_P = np.zeros((n, n))
            np.fill_diagonal(current_P, 1.0 / n)
        elif r == 4:
            # Anti-diagonal
            current_P = np.zeros((n, n))
            for i in range(n):
                current_P[i, n-1-i] = 1.0 / n
        elif r == 5:
            # Concentrate on extreme x+2y values
            current_P = np.zeros((n, n))
            current_P[0, 0] = 0.5
            current_P[9, 9] = 0.5
        elif r == 6:
            # Spread with boost on extremes
            current_P = np.ones((n, n)) / (n * n)
            current_P[0, 0] *= 3
            current_P[9, 9] *= 3
            current_P = _normalize(current_P)
        else:
            # Perturb best found
            current_P = best_P + np.random.normal(0, 0.1, (n, n))
            current_P = _normalize(current_P)
            if current_P is None:
                current_P = P.copy()
                
        current_score = _compute_ratio(current_P)
        
        # Phase 1: Hill climbing with adaptive step size
        step_size = 0.3
        local_time_limit = time_per_restart * 0.4
        local_start = time.time()
        no_improve_count = 0
        max_no_improve = 150
        
        while time.time() - local_start < local_time_limit:
            # Try a few random moves per iteration
            for _ in range(8):
                if time.time() - local_start >= local_time_limit:
                    break
                    
                i, j = np.random.randint(0, n, 2)
                delta = np.random.uniform(-step_size, step_size)
                
                new_P = current_P.copy()
                new_P[i, j] += delta
                
                if new_P[i, j] < -1e-9:
                    continue
                    
                new_P = np.clip(new_P, 0, None)
                s = new_P.sum()
                if s <= 1e-9:
                    continue
                new_P = new_P / s
                
                new_score = _score_fast(new_P.flatten())
                
                if new_score > current_score:
                    current_P = new_P
                    current_score = new_score
                    no_improve_count = 0
                    if current_score > best_score:
                        best_score = current_score
                        best_P = current_P.copy()
                    step_size *= 0.98
                else:
                    no_improve_count += 1
                    if no_improve_count > max_no_improve:
                        # Restart local search from a perturbation of best
                        current_P = best_P + np.random.normal(0, 0.15, (n, n))
                        current_P = _normalize(current_P)
                        if current_P is None:
                            current_P = P.copy()
                        current_score = _compute_ratio(current_P)
                        no_improve_count = 0
                        step_size = 0.3
                        break
                        
        # Phase 2: Scipy optimization with smooth objective
        if time.time() - start_time < timeout - safety_margin - 5:
            try:
                # Convert current_P to log-space params
                log_params = np.log(current_P.flatten() + 1e-15)
                
                # Use multiple temperatures for smooth max
                for temp in [0.3, 0.5, 1.0]:
                    if time.time() - start_time > timeout - safety_margin - 5:
                        break
                    
                    result = minimize(
                        _negative_ratio_smooth,
                        log_params,
                        args=(temp,),
                        method='L-BFGS-B',
                        options={'maxiter': 100, 'ftol': 1e-8}
                    )
                    
                    if result.success:
                        opt_probs = _log_softmax_params_to_prob(result.x)
                        opt_P = opt_probs.reshape(n, n)
                        opt_score = _score_fast(opt_P.flatten())
                        
                        if opt_score > best_score:
                            best_score = opt_score
                            best_P = opt_P.copy()
            except Exception:
                pass
    
    # Final polish: high-precision local optimization on best_P
    if time.time() - start_time < timeout - safety_margin:
        try:
            log_params = np.log(best_P.flatten() + 1e-15)
            
            for temp in [0.2, 0.3, 0.5]:
                if time.time() - start_time > timeout - safety_margin - 2:
                    break
                
                result = minimize(
                    _negative_ratio_smooth,
                    log_params,
                    args=(temp,),
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-10}
                )
                
                if result.success:
                    opt_probs = _log_softmax_params_to_prob(result.x)
                    opt_P = opt_probs.reshape(n, n)
                    opt_score = _score_fast(opt_P.flatten())
                    
                    if opt_score > best_score:
                        best_score = opt_score
                        best_P = opt_P.copy()
        except Exception:
            pass
    
    # Ensure validity: clip and normalize
    best_P = np.clip(best_P, 0, None)
    s = best_P.sum()
    if s > 1e-15:
        best_P = best_P / s
    else:
        best_P = np.ones((n, n)) / (n * n)
    
    return best_P
