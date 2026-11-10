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

# Precompute pooling matrices for faster computation
# M_D: shape (100, 19), M_H0: (100, 10), M_H1: (100, 19), M_H2: (100, 28)
def _make_pooling_matrix(vals, n_bins):
    M = np.zeros((100, n_bins), dtype=np.float64)
    for i, v in enumerate(vals):
        M[i, v] = 1.0
    return M

_M_D = _make_pooling_matrix(_d_vals, _N_D)
_M_H0 = _make_pooling_matrix(_h0_vals, _N_H0)
_M_H1 = _make_pooling_matrix(_h1_vals, _N_H1)
_M_H2 = _make_pooling_matrix(_h2_vals, _N_H2)

# Precompute feature matrix for Maximum Entropy parameterization
# Features: x (10), x+y (19), x+2y (28)
_N_FEAT = _N_H0 + _N_H1 + _N_H2
_Phi = np.zeros((100, _N_FEAT), dtype=np.float64)
# x features
for i in range(100):
    _Phi[i, _h0_vals[i]] = 1.0
# x+y features
for i in range(100):
    _Phi[i, _N_H0 + _h1_vals[i]] = 1.0
# x+2y features
for i in range(100):
    _Phi[i, _N_H0 + _N_H1 + _h2_vals[i]] = 1.0

def _entropy_from_probs(probs):
    """Compute Shannon entropy in bits from probability array."""
    probs = probs[probs > 1e-15]
    if len(probs) == 0:
        return 0.0
    return -np.sum(probs * np.log2(probs))

def _compute_ratio(P):
    P = np.clip(P, 0, None)
    s = P.sum()
    if s <= 1e-15:
        return 0.0
    w = P.flatten()
    
    d_probs = _M_D.T @ w
    h0_probs = _M_H0.T @ w
    h1_probs = _M_H1.T @ w
    h2_probs = _M_H2.T @ w
    
    D = _entropy_from_probs(d_probs)
    H0 = _entropy_from_probs(h0_probs)
    H1 = _entropy_from_probs(h1_probs)
    H2 = _entropy_from_probs(h2_probs)
    
    M = max(H0, H1, H2)
    
    if M <= 1e-9:
        return 0.0
        
    return D / M

def arithmetic_kakeya_distribution() -> np.ndarray:
    n = _N
    P = np.ones((n, n), dtype=np.float64) / (n * n)
    
    start_time = time.time()
    timeout = 630
    safety_margin = 20
    
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
        """Fast score computation using precomputed pooling matrices."""
        d_probs = _M_D.T @ P_flat
        h0_probs = _M_H0.T @ P_flat
        h1_probs = _M_H1.T @ P_flat
        h2_probs = _M_H2.T @ P_flat
        
        D = _entropy_from_probs(d_probs)
        H0 = _entropy_from_probs(h0_probs)
        H1 = _entropy_from_probs(h1_probs)
        H2 = _entropy_from_probs(h2_probs)
        
        M = max(H0, H1, H2)
        
        if M <= 1e-9:
            return 0.0
            
        return D / M
    
    def _log_softmax_params_to_prob(params):
        """Convert unconstrained params to probability via log-sum-exp."""
        log_p = params
        max_log_p = np.max(log_p)
        log_probs = log_p - max_log_p
        exp_probs = np.exp(log_probs)
        probs = exp_probs / np.sum(exp_probs)
        return probs
    
    def _me_params_to_prob(params):
        """Convert Maximum Entropy params to probability."""
        log_p = _Phi @ params
        max_log_p = np.max(log_p)
        log_probs = log_p - max_log_p
        exp_probs = np.exp(log_probs)
        probs = exp_probs / np.sum(exp_probs)
        return probs

    def _negative_ratio_smooth(params, temperature=0.5):
        """Smooth objective using soft-max for M."""
        probs = _log_softmax_params_to_prob(params)
        d_probs = _M_D.T @ probs
        h0_probs = _M_H0.T @ probs
        h1_probs = _M_H1.T @ probs
        h2_probs = _M_H2.T @ probs
        
        D = _entropy_from_probs(d_probs)
        H0 = _entropy_from_probs(h0_probs)
        H1 = _entropy_from_probs(h1_probs)
        H2 = _entropy_from_probs(h2_probs)
        
        # Soft max using log-sum-exp
        scaled = np.array([H0, H1, H2]) / temperature
        max_scaled = np.max(scaled)
        soft_max = (max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))) * temperature
        
        if soft_max <= 1e-9:
            return 0.0
            
        return -D / soft_max
    
    def _negative_ratio_me_smooth(params, temperature=0.5):
        """Smooth objective for Maximum Entropy parameterization."""
        probs = _me_params_to_prob(params)
        d_probs = _M_D.T @ probs
        h0_probs = _M_H0.T @ probs
        h1_probs = _M_H1.T @ probs
        h2_probs = _M_H2.T @ probs
        
        D = _entropy_from_probs(d_probs)
        H0 = _entropy_from_probs(h0_probs)
        H1 = _entropy_from_probs(h1_probs)
        H2 = _entropy_from_probs(h2_probs)
        
        scaled = np.array([H0, H1, H2]) / temperature
        max_scaled = np.max(scaled)
        soft_max = (max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))) * temperature
        
        if soft_max <= 1e-9:
            return 0.0
            
        return -D / soft_max

    # Multi-start with different initializations
    num_restarts = 25
    time_per_restart = (timeout - safety_margin - 80) / num_restarts  # reserve 80s for polish
    
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
            # Corner concentration
            current_P = np.zeros((n, n))
            current_P[0, 0] = 0.25
            current_P[0, 9] = 0.25
            current_P[9, 0] = 0.25
            current_P[9, 9] = 0.25
        elif r == 6:
            # Edge concentration
            current_P = np.zeros((n, n))
            current_P[0, :] = 0.25 / n
            current_P[9, :] = 0.25 / n
            current_P[:, 0] = 0.25 / n
            current_P[:, 9] = 0.25 / n
        else:
            # Perturb best found
            current_P = best_P + np.random.normal(0, 0.1, (n, n))
            current_P = _normalize(current_P)
            if current_P is None:
                current_P = P.copy()
                
        current_score = _compute_ratio(current_P)
        
        # Phase 1: Hill climbing with adaptive step size
        step_size = 0.3
        local_time_limit = time_per_restart * 0.35
        local_start = time.time()
        no_improve_count = 0
        max_no_improve = 200
        
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
                    step_size *= 0.97
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
                        
        # Phase 2: Smooth optimization using scipy L-BFGS-B on log-softmax params
        if time.time() - start_time < timeout - safety_margin - 40:
            # Convert current_P to log-softmax parameters
            current_P_clipped = np.clip(current_P, 1e-15, None)
            current_P_norm = current_P_clipped / current_P_clipped.sum()
            log_params = np.log(current_P_norm.flatten())
            
            # Try smooth objective with decreasing temperature
            for temp in [1.0, 0.5, 0.2, 0.1, 0.05]:
                if time.time() - start_time > timeout - safety_margin - 30:
                    break
                    
                try:
                    result = minimize(
                        _negative_ratio_smooth,
                        log_params,
                        args=(temp,),
                        method='L-BFGS-B',
                        options={'maxiter': 800, 'ftol': 1e-14, 'gtol': 1e-10}
                    )
                    
                    if result.success or result.fun < -current_score:
                        opt_params = result.x
                        opt_P = _log_softmax_params_to_prob(opt_params).reshape(n, n)
                        opt_score = _compute_ratio(opt_P)
                        
                        if opt_score > current_score:
                            current_P = opt_P
                            current_score = opt_score
                            if current_score > best_score:
                                best_score = current_score
                                best_P = current_P.copy()
                except Exception:
                    pass
        
        # Phase 2.5: Maximum Entropy Parameterization Optimization
        if time.time() - start_time < timeout - safety_margin - 30:
            # Initialize ME params to zero (uniform) or fit to current_P?
            # Fitting is complex. Start from 0 (uniform) and optimize.
            # Or start from a perturbation of 0.
            me_params = np.zeros(_N_FEAT)
            
            for temp in [1.0, 0.5, 0.2, 0.1]:
                if time.time() - start_time > timeout - safety_margin - 20:
                    break
                try:
                    result = minimize(
                        _negative_ratio_me_smooth,
                        me_params,
                        args=(temp,),
                        method='L-BFGS-B',
                        options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
                    )
                    if result.success or result.fun < -current_score:
                        opt_P = _me_params_to_prob(result.x).reshape(n, n)
                        opt_score = _compute_ratio(opt_P)
                        if opt_score > current_score:
                            current_P = opt_P
                            current_score = opt_score
                            if current_score > best_score:
                                best_score = current_score
                                best_P = current_P.copy()
                            # Update me_params to continue from here
                            me_params = result.x
                except Exception:
                    pass

        # Phase 3: Fine-grained hill climbing polish
        if time.time() - start_time < timeout - safety_margin - 15:
            step_size = 0.05
            polish_time_limit = min(15, time_per_restart * 0.25)
            polish_start = time.time()
            
            while time.time() - polish_start < polish_time_limit:
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
                    if current_score > best_score:
                        best_score = current_score
                        best_P = current_P.copy()
                    step_size *= 0.95
                else:
                    step_size *= 0.98
                    if step_size < 0.001:
                        break
                        
    # Final polish: very fine-grained search on best_P
    if time.time() - start_time < timeout - safety_margin:
        final_P = best_P.copy()
        final_score = best_score
        step_size = 0.02
        final_start = time.time()
        final_limit = timeout - safety_margin - (time.time() - start_time)
        
        if final_limit > 10:
            no_improve = 0
            while time.time() - final_start < final_limit:
                i, j = np.random.randint(0, n, 2)
                delta = np.random.uniform(-step_size, step_size)
                
                new_P = final_P.copy()
                new_P[i, j] += delta
                
                if new_P[i, j] < -1e-9:
                    continue
                    
                new_P = np.clip(new_P, 0, None)
                s = new_P.sum()
                if s <= 1e-9:
                    continue
                new_P = new_P / s
                
                new_score = _score_fast(new_P.flatten())
                
                if new_score > final_score:
                    final_P = new_P
                    final_score = new_score
                    no_improve = 0
                    step_size *= 0.98
                else:
                    no_improve += 1
                    step_size *= 0.97
                    if no_improve > 150 or step_size < 0.0005:
                        break
            
            if final_score > best_score:
                best_P = final_P
                best_score = final_score
    
    return best_P
