"""Erdos minimum overlap - optimize step heights to minimize max overlap."""
import numpy as np
from scipy.optimize import minimize
import time

N = 100

def compute_overlap_max(h, n_eval=200):
    """Compute the maximum overlap of h with shifted 1-h."""
    step_w = 2.0 / N
    shifts = np.linspace(-2, 2, n_eval)
    max_ov = 0.0
    for s in shifts:
        x_lo = max(0.0, -s)
        x_hi = min(2.0, 2.0 - s)
        if x_lo >= x_hi:
            continue
        total = 0.0
        for i in range(N):
            x0 = 2.0 * i / N
            x1 = 2.0 * (i + 1) / N
            a = max(x0, x_lo)
            b = min(x1, x_hi)
            if a >= b:
                continue
            x_mid = (a + b) / 2.0
            xs_mid = x_mid + s
            if xs_mid < 0 or xs_mid >= 2.0:
                continue
            j = int(xs_mid / step_w)
            if j >= N:
                j = N - 1
            if j < 0:
                j = 0
            total += h[i] * (1.0 - h[j]) * (b - a)
        ov = total
        if ov > max_ov:
            max_ov = ov
    return max_ov


def _smooth_max(vals, temperature=0.01):
    """Compute smooth maximum using log-sum-exp."""
    vals = np.asarray(vals)
    max_val = np.max(vals)
    if max_val < -1000:
        return max_val
    shifted = vals - max_val
    log_sum = np.log(np.sum(np.exp(shifted / temperature))) * temperature + max_val
    return log_sum


def _overlap_at_shift(h, s, n_eval=50):
    """Compute overlap at a specific shift s."""
    step_w = 2.0 / N
    x_lo = max(0.0, -s)
    x_hi = min(2.0, 2.0 - s)
    if x_lo >= x_hi:
        return 0.0
    total = 0.0
    for i in range(N):
        x0 = 2.0 * i / N
        x1 = 2.0 * (i + 1) / N
        a = max(x0, x_lo)
        b = min(x1, x_hi)
        if a >= b:
            continue
        x_mid = (a + b) / 2.0
        xs_mid = x_mid + s
        if xs_mid < 0 or xs_mid >= 2.0:
            continue
        j = int(xs_mid / step_w)
        if j >= N:
            j = N - 1
        if j < 0:
            j = 0
        total += h[i] * (1.0 - h[j]) * (b - a)
    return total


def _smooth_objective(z, temperature=0.01):
    """Smooth objective: smooth max of overlaps at multiple shifts."""
    # Reparameterize: z -> h via sigmoid, scaled to [0,1]
    h = 1.0 / (1.0 + np.exp(-z))
    # Evaluate overlaps at several shifts
    shifts = np.linspace(-1.5, 1.5, 20)
    overlaps = np.array([_overlap_at_shift(h, s) for s in shifts])
    return _smooth_max(overlaps, temperature)


def _normalize(h):
    """Normalize h to satisfy sum constraint: sum(h) = N/2."""
    diff = N / 2.0 - np.sum(h)
    h = h + diff / N
    h = np.clip(h, 0, 1)
    # Second pass if needed
    diff = N / 2.0 - np.sum(h)
    if abs(diff) > 1e-8:
        mask = h > 0.01 if diff > 0 else h < 0.99
        if np.any(mask):
            h[mask] += diff / np.sum(mask)
            h = np.clip(h, 0, 1)
    return h


def construct():
    t_start = time.time()
    deadline = t_start + 110.0
    
    np.random.seed(42)
    
    best_h = np.full(N, 0.5)
    best_score = compute_overlap_max(best_h)
    
    # Phase 1: Gradient-based optimization with smooth max surrogate
    # Use reparameterization: z -> h = sigmoid(z)
    # Start with z = 0 (gives h = 0.5)
    
    z_init = np.zeros(N)
    
    # Use L-BFGS-B to minimize the smooth objective
    try:
        result = minimize(
            _smooth_objective, z_init,
            method='L-BFGS-B',
            options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-6}
        )
        h_opt = 1.0 / (1.0 + np.exp(-result.x))
        h_opt = _normalize(h_opt)
        score = compute_overlap_max(h_opt)
        if score < best_score:
            best_score = score
            best_h = h_opt.copy()
    except Exception:
        pass
    
    # Phase 2: Local polish with direct optimization on h
    # Use projected gradient descent with smooth max
    
    if time.time() < deadline - 20:
        h_current = best_h.copy()
        lr = 0.01
        for step in range(500):
            if time.time() > deadline - 10:
                break
            # Compute overlaps at shifts
            shifts = np.linspace(-1.5, 1.5, 30)
            overlaps = np.array([_overlap_at_shift(h_current, s) for s in shifts])
            max_idx = np.argmax(overlaps)
            
            # Compute gradient numerically
            grad = np.zeros(N)
            eps = 1e-4
            for i in range(N):
                h_plus = h_current.copy()
                h_plus[i] = np.clip(h_plus[i] + eps, 0, 1)
                h_plus = _normalize(h_plus)
                ov_plus = _overlap_at_shift(h_plus, shifts[max_idx])
                
                h_minus = h_current.copy()
                h_minus[i] = np.clip(h_minus[i] - eps, 0, 1)
                h_minus = _normalize(h_minus)
                ov_minus = _overlap_at_shift(h_minus, shifts[max_idx])
                
                grad[i] = (ov_plus - ov_minus) / (2 * eps)
            
            # Gradient step
            h_new = h_current - lr * grad
            h_new = _normalize(h_new)
            
            new_score = compute_overlap_max(h_new)
            if new_score < best_score:
                best_score = new_score
                best_h = h_new.copy()
                h_current = h_new.copy()
                # Reduce learning rate slightly
                lr *= 0.95
            else:
                # Increase temperature or try different shift
                lr *= 0.8
    
    # Phase 3: Final refinement with scipy minimize on h directly
    if time.time() < deadline - 15:
        def objective_direct(h_flat):
            h = _normalize(h_flat)
            return compute_overlap_max(h)
        
        try:
            result = minimize(
                objective_direct, best_h.copy(),
                method='SLSQP',
                bounds=[(0.0, 1.0)] * N,
                constraints=[{'type': 'eq', 'fun': lambda h: np.sum(h) - N/2.0}],
                options={'maxiter': 100, 'ftol': 1e-6}
            )
            h_opt = _normalize(result.x)
            score = compute_overlap_max(h_opt)
            if score < best_score:
                best_score = score
                best_h = h_opt.copy()
        except Exception:
            pass
    
    # Final: ensure exact sum
    best_h = _normalize(best_h)
    
    return best_h
