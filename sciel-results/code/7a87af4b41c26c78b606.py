"""Rewritten program for Erdos minimum overlap task."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_TIME_LIMIT = 140.0

def _time_remaining():
    return _TIME_LIMIT - (time.time() - _START_TIME)

def _compute_min_corr(h):
    """Compute min over shifts of the cross-correlation."""
    n = len(h)
    min_corr = float('inf')
    for k in range(1, n):
        corr = np.dot(h[:n-k], h[k:]) * (2.0 / n)
        if corr < min_corr:
            min_corr = corr
    return min_corr

def _compute_all_corrs(h):
    """Compute all cross-correlations."""
    n = len(h)
    corrs = np.zeros(n-1)
    for k in range(1, n):
        corrs[k-1] = np.dot(h[:n-k], h[k:]) * (2.0 / n)
    return corrs

def _project(v, target_sum):
    """Project onto {x: 0<=x<=1, sum=target_sum}."""
    n = len(v)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho_cands = np.nonzero(u * np.arange(1, n + 1) > (cssv - target_sum))[0]
    if len(rho_cands) == 0:
        rho = n - 1
    else:
        rho = rho_cands[-1]
    theta = (cssv[rho] - target_sum) / (rho + 1.0)
    w = np.maximum(v - theta, 0)
    w = np.clip(w, 0, 1)
    
    for _ in range(50):
        diff = target_sum - np.sum(w)
        if abs(diff) < 1e-14:
            break
        if diff > 0:
            mask = w < 1.0
            cnt = np.sum(mask)
            if cnt == 0:
                break
            w[mask] += diff / cnt
        else:
            mask = w > 0.0
            cnt = np.sum(mask)
            if cnt == 0:
                break
            w[mask] += diff / cnt
        w = np.clip(w, 0, 1)
    return w

def _max_overlap_subgrad(h, c_target, lr=0.05, max_iter=200):
    """
    Projected gradient ascent to maximize min correlation.
    Uses subgradient of min over k of corr_k(h).
    """
    n = len(h)
    target_sum = n / 2.0
    h = h.copy()
    best_h = h.copy()
    best_min_corr = _compute_min_corr(h)
    
    for it in range(max_iter):
        if _time_remaining() < 5.0:
            break
        
        # Compute all correlations
        corrs = _compute_all_corrs(h)
        min_idx = np.argmin(corrs)
        min_corr = corrs[min_idx]
        
        # Subgradient direction: increase the minimum correlation
        k_star = min_idx + 1
        grad = np.zeros(n)
        for j in range(n - k_star):
            grad[j] += (2.0 / n) * h[j + k_star]
            grad[j + k_star] += (2.0 / n) * h[j]
        
        # Step
        h_new = h + lr * grad
        h_new = _project(h_new, target_sum)
        
        new_min_corr = _compute_min_corr(h_new)
        if new_min_corr > best_min_corr:
            best_min_corr = new_min_corr
            best_h = h_new.copy()
        
        h = h_new
        
        # Adaptive learning rate
        if new_min_corr < min_corr - 0.001:
            lr *= 0.8
        elif it > 50:
            lr = min(lr * 1.02, 0.2)
    
    return best_h, best_min_corr

def _local_search(h, max_iter=5000):
    """
    Local search: randomly perturb pairs of coordinates and accept if min corr improves.
    """
    n = len(h)
    target_sum = n / 2.0
    h = h.copy()
    best_h = h.copy()
    best_min_corr = _compute_min_corr(h)
    rng = np.random.default_rng(42)
    
    for it in range(max_iter):
        if _time_remaining() < 5.0:
            break
        
        # Pick two random indices to swap perturb
        i, j = rng.integers(0, n, size=2)
        if i == j:
            continue
        
        # Perturbation magnitude
        delta = rng.uniform(-0.05, 0.05)
        
        h[i] += delta
        h[j] -= delta  # Keep sum constant
        
        # Project to [0,1]
        if h[i] < 0:
            h[j] -= h[i]
            h[i] = 0
        elif h[i] > 1:
            h[j] += (h[i] - 1)
            h[i] = 1
        if h[j] < 0:
            h[i] -= h[j]
            h[j] = 0
        elif h[j] > 1:
            h[i] += (h[j] - 1)
            h[j] = 1
        
        # Ensure sum is correct
        diff = target_sum - np.sum(h)
        if abs(diff) > 1e-10:
            mask = (h > 0) & (h < 1)
            if np.sum(mask) > 0:
                h[mask] += diff / np.sum(mask)
            h = np.clip(h, 0, 1)
        
        new_min_corr = _compute_min_corr(h)
        if new_min_corr > best_min_corr:
            best_min_corr = new_min_corr
            best_h = h.copy()
        else:
            # Revert
            h[i] -= delta
            h[j] += delta
            # Re-project
            h = _project(h, target_sum)
    
    return best_h, best_min_corr

def construct():
    n = N
    target_sum = n / 2.0
    
    best_x = np.full(n, 0.5)
    best_obj = 1.0 - _compute_min_corr(best_x)
    
    np.random.seed(42)
    
    # Generate diverse restarts
    restarts = [np.full(n, 0.5)]
    
    # Linear ramps
    restarts.append(np.linspace(0, 1, n))
    restarts.append(np.linspace(1, 0, n))
    
    # Sinusoidal patterns
    x = np.linspace(0, 2*np.pi, n)
    for freq in [1, 2, 3, 4, 5, 7, 10]:
        for amp in [0.2, 0.3, 0.4, 0.5]:
            restarts.append(0.5 + amp*np.sin(freq*x))
            restarts.append(0.5 + amp*np.cos(freq*x))
    
    # Random smooth via Fourier
    for _ in range(20):
        coeffs = np.random.randn(1, 15, 2) * 0.05
        t = np.linspace(0, 2*np.pi, n)
        sig = 0.5
        for j in range(1, 15):
            for k in range(2):
                sig += coeffs[0, j, k] * np.sin(j * t + k*np.pi/2)
        restarts.append(np.clip(sig, 0, 1))
    
    # Block patterns
    for k in range(1, n//2):
        restarts.append(np.concatenate([np.zeros(k), np.ones(n-k)*0.5]))
        restarts.append(np.concatenate([np.ones(k)*0.5, np.zeros(n-k)]))
    
    # Alternating
    alt1 = np.array([0.0 if i % 2 == 0 else 1.0 for i in range(n)])
    alt2 = np.array([1.0 if i % 2 == 0 else 0.0 for i in range(n)])
    restarts.append(alt1)
    restarts.append(alt2)
    
    # Random
    for _ in range(15):
        restarts.append(np.random.uniform(0, 1, n))
    
    # Phase 1: Subgradient ascent from multiple restarts
    for x0 in restarts:
        if _time_remaining() < 10.0:
            break
        x0 = _project(x0, target_sum)
        
        try:
            h_new, min_corr = _max_overlap_subgrad(x0, 0.0, lr=0.05, max_iter=300)
            obj = 1.0 - min_corr
            if obj < best_obj:
                best_obj = obj
                best_x = h_new
        except Exception:
            pass
    
    # Phase 2: Local search refinement
    if _time_remaining() > 30.0:
        for _ in range(3):
            if _time_remaining() < 10.0:
                break
            try:
                h_new, min_corr = _local_search(best_x, max_iter=3000)
                obj = 1.0 - min_corr
                if obj < best_obj:
                    best_obj = obj
                    best_x = h_new
            except Exception:
                pass
    
    # Phase 3: More subgradient with smaller steps
    if _time_remaining() > 20.0:
        for lr in [0.02, 0.01, 0.005]:
            if _time_remaining() < 5.0:
                break
            try:
                h_new, min_corr = _max_overlap_subgrad(best_x, 0.0, lr=lr, max_iter=500)
                obj = 1.0 - min_corr
                if obj < best_obj:
                    best_obj = obj
                    best_x = h_new
            except Exception:
                pass
    
    # Phase 4: Final local search
    if _time_remaining() > 15.0:
        try:
            h_new, min_corr = _local_search(best_x, max_iter=5000)
            obj = 1.0 - min_corr
            if obj < best_obj:
                best_obj = obj
                best_x = h_new
        except Exception:
            pass
    
    # Exact sum fix
    best_x = np.clip(best_x, 0, 1)
    diff = target_sum - np.sum(best_x)
    if abs(diff) > 1e-13:
        mask = (best_x > 0) & (best_x < 1)
        if np.sum(mask) > 0:
            best_x[mask] += diff / np.sum(mask)
            best_x = np.clip(best_x, 0, 1)
    
    return best_x
# EVOLVE-BLOCK-END
