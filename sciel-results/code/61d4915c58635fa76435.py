import numpy as np
import time
from scipy.optimize import minimize

N = 50
DELTA = 0.5 / N

def _score_raw(h):
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    l2_sq = np.sum(h**2)
    h_conv = np.convolve(h, h, mode='full')
    autoconv_max = DELTA * np.max(h_conv)
    if autoconv_max == 0:
        return 0.0
    return (l2_sq / (l1 * autoconv_max))

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin for 150s limit
    
    best_h = np.ones(N) / N
    best_score = _score_raw(best_h)
    
    # Strategy: Use simulated annealing on a log-parameterized representation
    # h_i = exp(theta_i) for active indices, 0 for inactive
    # This naturally enforces non-negativity and allows sparse solutions
    
    def _score_from_theta(theta, active_mask):
        h = np.zeros(N)
        h[active_mask] = np.exp(theta)
        l1 = np.sum(h)
        if l1 == 0:
            return 0.0
        l2_sq = np.sum(h**2)
        h_conv = np.convolve(h, h, mode='full')
        autoconv_max = DELTA * np.max(h_conv)
        if autoconv_max == 0:
            return 0.0
        return (l2_sq / (l1 * autoconv_max))
    
    # Find a good Sidon-like set of indices
    def find_good_sidon_set(n_points, n_candidates=200):
        """Find a subset of indices with minimal autoconvolution peak"""
        best_set = None
        best_peak = np.inf
        
        # Try random subsets of various sizes
        for size in range(3, 15):
            for _ in range(n_candidates):
                candidate = np.random.choice(N, size, replace=False)
                h = np.zeros(N)
                h[candidate] = 1.0
                h_conv = np.convolve(h, h, mode='full')
                peak = np.max(h_conv)
                if peak < best_peak:
                    best_peak = peak
                    best_set = candidate.copy()
        
        return best_set, best_peak
    
    # Simulated annealing on theta parameters
    def simulated_annealing(initial_theta, active_mask, n_iter=5000, T_start=1.0, T_end=0.01):
        current_theta = initial_theta.copy()
        current_score = _score_from_theta(current_theta, active_mask)
        best_theta = current_theta.copy()
        best_score_local = current_score
        
        n_active = np.sum(active_mask)
        if n_active == 0:
            return current_theta, current_score
        
        for i in range(n_iter):
            if time.time() > deadline - 5:
                break
            
            T = T_start * (T_end / T_start) ** (i / n_iter)
            
            # Pick a random active index to perturb
            idx = np.random.randint(n_active)
            old_val = current_theta[idx]
            current_theta[idx] = old_val + np.random.normal(0, 0.5)
            
            new_score = _score_from_theta(current_theta, active_mask)
            
            if new_score > current_score or np.random.random() < np.exp((new_score - current_score) / max(T, 1e-10)):
                current_score = new_score
                if new_score > best_score_local:
                    best_score_local = new_score
                    best_theta = current_theta.copy()
            else:
                current_theta[idx] = old_val
        
        return best_theta, best_score_local
    
    # Try multiple active set sizes
    for k in [4, 5, 6, 7, 8, 10, 12, 15, 20]:
        if time.time() > deadline - 10:
            break
        
        # Find good index sets for this size
        for trial in range(20):
            if time.time() > deadline - 10:
                break
            
            # Random subset
            active_idx = np.random.choice(N, k, replace=False)
            active_mask = np.zeros(N, dtype=bool)
            active_mask[active_idx] = True
            
            # Initialize theta
            initial_theta = np.random.normal(0, 0.5, k)
            
            theta, score = simulated_annealing(initial_theta, active_mask, n_iter=3000, T_start=0.5, T_end=0.01)
            
            h = np.zeros(N)
            h[active_idx] = np.exp(theta)
            h = np.maximum(0, h)
            if np.sum(h) > 0:
                h /= np.sum(h)
                s = _score_raw(h)
                if s > best_score:
                    best_score = s
                    best_h = h.copy()
        
        # Also try the best Sidon-like set for this size
        sidon_set, _ = find_good_sidon_set(k, n_candidates=50)
        if sidon_set is not None and len(sidon_set) == k:
            active_mask = np.zeros(N, dtype=bool)
            active_mask[sidon_set] = True
            initial_theta = np.random.normal(0, 0.3, k)
            
            theta, score = simulated_annealing(initial_theta, active_mask, n_iter=5000, T_start=0.3, T_end=0.005)
            
            h = np.zeros(N)
            h[sidon_set] = np.exp(theta)
            h = np.maximum(0, h)
            if np.sum(h) > 0:
                h /= np.sum(h)
                s = _score_raw(h)
                if s > best_score:
                    best_score = s
                    best_h = h.copy()
    
    # Final SLSQP polish on the best found
    if time.time() < deadline - 5:
        h_start = best_h.copy()
        
        def neg_obj(h):
            l1 = np.sum(h)
            if l1 == 0:
                return 0
            l2_sq = np.sum(h**2)
            h_conv = np.convolve(h, h, mode='full')
            max_conv = np.max(h_conv)
            if max_conv == 0:
                return 0
            return -(l2_sq / max_conv)
        
        constraints = [
            {'type': 'eq', 'fun': lambda h: np.sum(h) - 1.0},
            {'type': 'ineq', 'fun': lambda h: h}
        ]
        bounds = [(0.0, 1.0)] * N
        
        try:
            res = minimize(
                neg_obj,
                h_start,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 500, 'ftol': 1e-12}
            )
            if res.success:
                h_candidate = np.maximum(0, res.x)
                if np.sum(h_candidate) > 0:
                    h_candidate /= np.sum(h_candidate)
                    s_candidate = _score_raw(h_candidate)
                    if s_candidate > best_score:
                        best_score = s_candidate
                        best_h = h_candidate
        except Exception:
            pass
    
    # Ensure nonnegativity and normalization
    best_h = np.maximum(0, best_h)
    if np.sum(best_h) == 0:
        best_h = np.ones(N) / N
    else:
        best_h /= np.sum(best_h)
    
    return list(best_h)
