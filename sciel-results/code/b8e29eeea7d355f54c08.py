"""Rewritten program: multi-phase direct optimization of 50 heights."""
import numpy as np
from scipy.optimize import minimize
import time

N = 50

def compute_ratio(h):
    """Compute the objective ratio for a nonnegative height vector."""
    h = np.asarray(h, dtype=np.float64)
    h = np.clip(h, 0, None)
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    l2_sq = np.sum(h ** 2)
    conv = np.convolve(h, h, mode='full')
    sup_norm = np.max(conv)
    if sup_norm < 1e-15:
        return 0.0
    return l2_sq / (l1 * sup_norm)

def neg_ratio_smooth(h, alpha=100.0):
    """Smooth surrogate for negative ratio using log-sum-exp for max."""
    h = np.clip(h, 0, None)
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    l2_sq = np.sum(h ** 2)
    conv = np.convolve(h, h, mode='full')
    # Log-sum-exp for smooth max
    m = np.max(conv)
    if m < 1e-15:
        return 0.0
    lse = m + np.log(np.sum(np.exp((conv - m) * alpha)) / alpha)
    if lse < 1e-15:
        return 0.0
    return -(l2_sq / (l1 * lse))

def generate_initial_candidates():
    """Generate diverse initial candidates for optimization."""
    candidates = []
    
    # 1. Sparse: single spike
    for i in range(N):
        h = np.zeros(N)
        h[i] = 1.0
        candidates.append(h.copy())
    
    # 2. Sparse: two spikes
    for i in range(N):
        for j in range(i + 1, N):
            h = np.zeros(N)
            h[i] = 1.0
            h[j] = 1.0
            candidates.append(h.copy())
    
    # 3. Concentrated: top-k ones
    for k in [1, 2, 3, 5, 10, 25]:
        for start in range(0, N - k + 1, max(1, (N - k) // 10)):
            h = np.zeros(N)
            h[start:start + k] = 1.0
            candidates.append(h.copy())
    
    # 4. Gaussian-like
    for center in [N // 2, N // 4, 3 * N // 4]:
        for width in [2, 5, 10, 20]:
            x = np.arange(N)
            h = np.exp(-0.5 * ((x - center) / width) ** 2)
            candidates.append(h.copy())
    
    # 5. Triangular
    for center in [N // 2, N // 4, 3 * N // 4]:
        for width in [5, 10, 20]:
            h = np.maximum(0, 1 - np.abs(np.arange(N) - center) / width)
            candidates.append(h.copy())
    
    # 6. Cosine bump
    for center in [N // 2, N // 4, 3 * N // 4]:
        for width in [5, 10, 20]:
            x = np.arange(N)
            arg = np.pi * (x - center) / (2 * width)
            mask = np.abs(arg) <= np.pi / 2
            h = np.zeros(N)
            h[mask] = np.cos(arg[mask])
            candidates.append(h.copy())
    
    # 7. Random nonnegative
    np.random.seed(42)
    for _ in range(20):
        h = np.random.exponential(1.0, N)
        candidates.append(h.copy())
    
    # 8. Alternating patterns
    for period in [2, 3, 4, 5]:
        h = np.zeros(N)
        for i in range(N):
            if i % period == 0:
                h[i] = 1.0
        candidates.append(h.copy())
    
    # 9. Edge-concentrated
    for k in [1, 2, 5]:
        h = np.zeros(N)
        h[:k] = 1.0
        candidates.append(h.copy())
        h = np.zeros(N)
        h[-k:] = 1.0
        candidates.append(h.copy())
    
    return candidates

def optimize_from_start(h0, deadline):
    """Optimize from a starting point using multi-phase approach."""
    best_h = h0.copy()
    best_score = compute_ratio(best_h)
    
    elapsed = lambda: time.time() - _start_time
    
    # Phase 1: SLSQP for broad search
    try:
        bounds = [(0, 100) for _ in range(N)]
        res = minimize(
            neg_ratio_smooth, best_h,
            method='SLSQP', bounds=bounds,
            options={'maxiter': 50, 'ftol': 1e-8}
        )
        if elapsed() < deadline - 5.0:
            h_cand = np.clip(res.x, 0, None)
            score_cand = compute_ratio(h_cand)
            if score_cand > best_score:
                best_score = score_cand
                best_h = h_cand
    except Exception:
        pass
    
    # Phase 2: L-BFGS-B refinement
    if elapsed() < deadline - 3.0:
        try:
            res = minimize(
                neg_ratio_smooth, best_h,
                method='L-BFGS-B', bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            h_cand = np.clip(res.x, 0, None)
            score_cand = compute_ratio(h_cand)
            if score_cand > best_score:
                best_score = score_cand
                best_h = h_cand
        except Exception:
            pass
    
    # Phase 3: Final polish with tight tolerance
    if elapsed() < deadline - 1.0:
        try:
            res = minimize(
                neg_ratio_smooth, best_h,
                method='L-BFGS-B', bounds=bounds,
                options={'maxiter': 50, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            h_cand = np.clip(res.x, 0, None)
            score_cand = compute_ratio(h_cand)
            if score_cand > best_score:
                best_score = score_cand
                best_h = h_cand
        except Exception:
            pass
    
    return best_h, best_score

_start_time = None

def construct():
    global _start_time
    _start_time = time.time()
    deadline = _start_time + 110.0  # Safety margin
    
    # Generate candidates and evaluate
    candidates = generate_initial_candidates()
    
    best_h = np.ones(N)
    best_score = compute_ratio(best_h)
    
    # Quick evaluation of all candidates
    for h in candidates:
        score = compute_ratio(h)
        if score > best_score:
            best_score = score
            best_h = h.copy()
        if time.time() > deadline - 80.0:
            break
    
    # Select top candidates for optimization
    # Sort by score, take top few
    scored = [(compute_ratio(h), h) for h in candidates[:50]]
    scored.sort(key=lambda x: x[0], reverse=True)
    
    top_candidates = [h for _, h in scored[:10]]
    if best_h not in top_candidates:
        top_candidates.append(best_h)
    
    # Optimize from each top candidate
    for h0 in top_candidates:
        if time.time() > deadline - 10.0:
            break
        h_opt, score_opt = optimize_from_start(h0.copy(), deadline)
        if score_opt > best_score:
            best_score = score_opt
            best_h = h_opt
    
    # Final fallback
    if best_score < 0.5:
        # Try a concentrated center
        h = np.zeros(N)
        h[N // 2] = 1.0
        best_h = h
        best_score = compute_ratio(best_h)
    
    return [float(x) for x in best_h]
