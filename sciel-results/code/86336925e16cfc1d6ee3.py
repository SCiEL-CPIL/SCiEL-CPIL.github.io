"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h.
Strategy: Parametric sinusoidal initialization + smooth max optimization.
We initialize h as a clipped sinusoid (naturally bounded in [0,1]) and use
smooth max (log-sum-exp) objective with projected gradient descent. Multiple
diverse initializations explore different basins of attraction.
"""
import time
import numpy as np
from scipy.optimize import minimize

N = 100
_DEADLINE = time.time() + 145  # Safety margin before 150s kill

def _linear_overlap_values(h):
    """
    Compute the linear overlap values C_k for k=0..N-1.
    C_k = (2/n) * (n/2 - r_k) where r_k is the linear autocorrelation.
    """
    n = len(h)
    n_fft = 2 * n
    h_pad = np.zeros(n_fft)
    h_pad[:n] = h
    H = np.fft.rfft(h_pad, n=n_fft)
    corr = np.fft.irfft(H * np.conj(H), n=n_fft)
    r = corr[:n]
    cs = (2.0 / n) * (n / 2.0 - r)
    return cs

def _score(h):
    """
    Compute the maximum linear overlap.
    """
    h = np.asarray(h, dtype=np.float64)
    cs = _linear_overlap_values(h)
    return float(np.max(cs))

def _smooth_max_and_grad(h, beta=100.0):
    """
    Compute smooth max of overlap values and its gradient.
    Uses log-sum-exp for numerical stability.
    """
    cs = _linear_overlap_values(h)
    m = np.max(cs)
    exp_terms = np.exp(beta * (cs - m))
    sum_exp = np.sum(exp_terms)
    w = exp_terms / sum_exp
    val = m + np.log(sum_exp) / beta
    
    # Gradient: d/dh_j sum_k w_k * C_k
    # C_k = (2/n) * (n/2 - r_k), where r_k = sum_j h_j * h_{j+k}
    # dC_k/dh_j = -(2/n) * (h_{j+k} + h_{j-k}) for appropriate indices
    grad = np.zeros(N)
    for k in range(N):
        if w[k] < 1e-15:
            continue
        factor = -(2.0 / N) * w[k]
        if k == 0:
            grad += factor * 2.0 * h
        else:
            grad[:N-k] += factor * h[k:]
            grad[k:] += factor * h[:N-k]
    
    return val, grad

def _fix_sum(h):
    """
    Adjust h to have sum exactly N/2, keeping values in [0, 1].
    """
    h = np.array(h, dtype=np.float64, copy=True)
    target = N / 2.0
    diff = target - np.sum(h)
    h += diff / N
    h = np.clip(h, 0.0, 1.0)
    
    # If clipping changed the sum, do a second pass
    diff = target - np.sum(h)
    if abs(diff) > 1e-12:
        mask = (h > 0) & (h < 1)
        if np.any(mask):
            h[mask] += diff / np.sum(mask)
            h = np.clip(h, 0.0, 1.0)
            
    # Final tiny correction on one element to ensure exact sum in floating point
    diff = target - np.sum(h)
    if abs(diff) > 1e-15:
        if diff > 0:
            idx = np.argmin(h)
        else:
            idx = np.argmax(h)
        h[idx] += diff
        h = np.clip(h, 0.0, 1.0)
        
    return h

def _parametric_h(frequency, amplitude, phase, offset=0.5):
    """
    Generate h as a clipped sinusoid: h_i = clip(offset + amplitude * sin(2*pi*frequency*i/N + phase), 0, 1)
    Then adjust to have sum = N/2.
    """
    i = np.arange(N)
    h = offset + amplitude * np.sin(2 * np.pi * frequency * i / N + phase)
    h = np.clip(h, 0.0, 1.0)
    h = _fix_sum(h)
    return h

def _projected_gradient_descent(h0, max_iter=200, beta=100.0, lr=0.01):
    """
    Projected gradient descent on smooth max objective.
    """
    h = h0.copy()
    best_h = h.copy()
    best_score = _score(h)
    
    for it in range(max_iter):
        if time.time() > _DEADLINE - 10:
            break
        
        val, grad = _smooth_max_and_grad(h, beta)
        
        # Projected gradient step
        h_new = h - lr * grad
        h_new = np.clip(h_new, 0.0, 1.0)
        h_new = _fix_sum(h_new)
        
        s = _score(h_new)
        if s < best_score:
            best_score = s
            best_h = h_new.copy()
        
        h = h_new
        
        # Adaptive learning rate
        if it % 20 == 0:
            lr *= 0.9
    
    return best_h, best_score

def _slsqp_polish(h0, max_iter=100):
    """
    SLSQP polish step to refine the solution.
    """
    n = N
    target_sum = n / 2.0
    
    def obj_func(h):
        cs = _linear_overlap_values(h)
        return float(np.max(cs))
    
    def jac_func(h):
        cs = _linear_overlap_values(h)
        k_max = int(np.argmax(cs))
        grad = np.zeros(n)
        factor = -(2.0 / n)
        if k_max == 0:
            grad += factor * 2.0 * h
        else:
            grad[:n-k_max] += factor * h[k_max:]
            grad[k_max:] += factor * h[:n-k_max]
        return grad
    
    cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - target_sum,
             'jac': lambda x: np.ones(n)}]
    bounds = [(0.0, 1.0)] * n
    
    try:
        res = minimize(obj_func, h0, jac=jac_func, method='SLSQP',
                       bounds=bounds, constraints=cons,
                       options={'maxiter': max_iter, 'ftol': 1e-9})
        return res.x
    except Exception:
        return h0

def construct():
    """Construct step heights minimizing maximum overlap with shifted complement."""
    t0 = time.time()
    np.random.seed(42)
    
    best_h = np.full(N, 0.5)
    best_h = _fix_sum(best_h)
    best_score = _score(best_h)
    
    # Generate diverse initializations
    initializations = []
    
    # 1. Constant
    initializations.append(np.full(N, 0.5))
    
    # 2. Sinusoidal with various parameters
    for freq in [1, 2, 3, 4, 5]:
        for amp in [0.3, 0.5, 0.7]:
            for phase in [0, np.pi/2, np.pi, 3*np.pi/2]:
                h = _parametric_h(freq, amp, phase)
                initializations.append(h)
    
    # 3. Random
    for _ in range(5):
        h = np.random.uniform(0, 1, N)
        h = _fix_sum(h)
        initializations.append(h)
    
    # 4. Alternating patterns
    h_alt1 = np.array([0.3, 0.7] * (N // 2))
    initializations.append(_fix_sum(h_alt1))
    h_alt2 = np.array([0.2, 0.8] * (N // 2))
    initializations.append(_fix_sum(h_alt2))
    
    # 5. Block patterns
    h_block1 = np.concatenate([np.full(25, 0.3), np.full(25, 0.7), np.full(25, 0.3), np.full(25, 0.7)])
    initializations.append(_fix_sum(h_block1))
    h_block2 = np.concatenate([np.full(50, 0.3), np.full(50, 0.7)])
    initializations.append(_fix_sum(h_block2))
    
    # Run projected gradient descent from each initialization
    for i, h_init in enumerate(initializations):
        if time.time() > _DEADLINE - 15:
            break
        
        h_result, score = _projected_gradient_descent(h_init, max_iter=150, beta=80.0, lr=0.02)
        
        if score < best_score:
            best_score = score
            best_h = h_result.copy()
    
    # SLSQP polish on the best result
    if time.time() < _DEADLINE - 10:
        h_polished = _slsqp_polish(best_h.copy(), max_iter=100)
        s_polished = _score(h_polished)
        if s_polished < best_score:
            best_score = s_polished
            best_h = h_polished
    
    # Final sum fix
    best_h = _fix_sum(best_h)
    return best_h
