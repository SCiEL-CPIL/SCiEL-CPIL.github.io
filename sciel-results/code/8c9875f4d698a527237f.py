import numpy as np
import time
from scipy.optimize import minimize as scipy_minimize

N = 50

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a sequence of 50 nonnegative heights.
    Strategy: Local optimization over weight ratios for 2-point and 3-point clusters,
    with scipy L-BFGS-B polish on the best candidates.
    """
    t0 = time.time()
    deadline = 110.0  # Safety margin well under 120s
    
    def compute_score(h):
        """Compute the ratio: L2^2 / (L1 * sup_norm_autoconv)."""
        h = np.asarray(h, dtype=float)
        l1 = float(np.sum(h))
        if l1 < 1e-15:
            return 0.0
        l2_sq = float(np.sum(h * h))
        conv = np.convolve(h, h, mode='full')
        conv_inf = float(np.max(conv))
        if conv_inf < 1e-15:
            return 0.0
        return l2_sq / (l1 * conv_inf)
    
    def neg_score(h):
        """Negative score for minimization."""
        return -compute_score(h)
    
    def local_optimize(h_init, max_iter=300, tol=1e-12):
        """Coordinate descent with adaptive steps, then scipy polish."""
        h = np.array(h_init, dtype=float)
        l1 = np.sum(h)
        if l1 < 1e-15:
            return h, 0.0
        h = h / l1  # normalize to L1=1
        
        eps = 1e-14
        h_safe = np.maximum(h, eps)
        h_safe = h_safe / np.sum(h_safe)
        
        best_h = h_safe.copy()
        best_s = compute_score(best_h)
        
        # Adaptive coordinate descent
        for it in range(max_iter):
            if time.time() - t0 > deadline - 5.0:  # Leave time for polish
                break
            improved = False
            for i in range(N):
                if best_h[i] < 1e-10:
                    continue
                for j in range(N):
                    if j == i or best_h[j] < 1e-10:
                        continue
                    # Adaptive step sizes
                    steps = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
                    best_delta_s = 0.0
                    best_trial = None
                    for frac in steps:
                        delta = best_h[j] * frac
                        if delta >= best_h[j]:
                            continue
                        h_trial = best_h.copy()
                        h_trial[i] += delta
                        h_trial[j] -= delta
                        s_trial = compute_score(h_trial)
                        if s_trial > best_s + best_delta_s:
                            best_delta_s = s_trial - best_s
                            best_trial = h_trial.copy()
                    if best_trial is not None:
                        best_h = best_trial
                        best_s += best_delta_s
                        improved = True
            if not improved:
                break
        
        # Scipy polish on the active support
        active = best_h > 1e-10
        if np.sum(active) >= 2:
            active_idx = np.where(active)[0]
            h_active = best_h[active_idx].copy()
            
            def neg_score_active(x):
                h_full = np.zeros(N)
                h_full[active_idx] = x
                h_full = np.maximum(h_full, 0)
                s = np.sum(h_full)
                if s < 1e-15:
                    return 0.0
                h_full = h_full / s
                return -compute_score(h_full)
            
            x0 = h_active / np.sum(h_active)
            bounds = [(0, 1)] * len(active_idx)
            
            try:
                result = scipy_minimize(
                    neg_score_active, x0, method='L-BFGS-B',
                    bounds=bounds, options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-12}
                )
                h_polished = np.zeros(N)
                h_polished[active_idx] = np.maximum(result.x, 0)
                s_polished = np.sum(h_polished)
                if s_polished > 1e-15:
                    h_polished = h_polished / s_polished
                    s_polished_val = compute_score(h_polished)
                    if s_polished_val > best_s:
                        best_h = h_polished
                        best_s = s_polished_val
            except Exception:
                pass
        
        return best_h, best_s
    
    best_score = 0.0
    best_h = None
    
    # Generate diverse initial configurations
    candidates = []
    
    # 2-point clusters with various weight ratios - focus on promising region
    for r in np.logspace(-2, 2, 200):
        if time.time() - t0 > deadline - 30.0:
            break
        a = r
        b = 1.0
        total = a + b
        h = np.zeros(N)
        h[0] = a / total
        h[1] = b / total
        candidates.append(h)
        
        # Also try at different positions
        for pos in [0, 5, 10, 20, 25]:
            if pos + 1 < N:
                h2 = np.zeros(N)
                h2[pos] = a / total
                h2[pos+1] = b / total
                candidates.append(h2)
    
    # 3-point clusters
    for r1 in np.logspace(-2, 1, 50):
        if time.time() - t0 > deadline - 30.0:
            break
        for r2 in np.logspace(-2, 1, 50):
            if time.time() - t0 > deadline - 30.0:
                break
            a = r1
            b = 1.0
            c = r2
            total = a + b + c
            h = np.zeros(N)
            h[0] = a / total
            h[1] = b / total
            h[2] = c / total
            candidates.append(h)
    
    # 4-point clusters
    for r1 in np.logspace(-1.5, 1, 25):
        if time.time() - t0 > deadline - 30.0:
            break
        for r2 in np.logspace(-1.5, 1, 25):
            if time.time() - t0 > deadline - 30.0:
                break
            a = r1
            b = 1.0
            c = 1.0
            d = r2
            total = a + b + c + d
            h = np.zeros(N)
            h[0] = a / total
            h[1] = b / total
            h[2] = c / total
            h[3] = d / total
            candidates.append(h)
    
    # Evaluate all candidates and keep top ones for local optimization
    scored_candidates = []
    for h in candidates:
        s = compute_score(h)
        scored_candidates.append((s, h))
    
    scored_candidates.sort(key=lambda x: -x[0])
    
    # Local optimize the top candidates
    top_k = min(30, len(scored_candidates))
    for s_init, h_init in scored_candidates[:top_k]:
        if time.time() - t0 > deadline - 10.0:
            break
        h_opt, s_opt = local_optimize(h_init, max_iter=200)
        if s_opt > best_score:
            best_score = s_opt
            best_h = h_opt.copy()
    
    if best_h is None:
        # Fallback: known good 2-point 2:1
        best_h = [0.0] * N
        best_h[0] = 2.0 / 3.0
        best_h[1] = 1.0 / 3.0
    else:
        best_h = best_h.tolist()
    
    return best_h
# EVOLVE-BLOCK-END
