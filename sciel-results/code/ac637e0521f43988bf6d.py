"""Rational generating function approach for Erdos minimum overlap."""
import numpy as np
from scipy.optimize import minimize

N = 100

def _compute_h_from_rational(coeffs, order):
    """
    Generate step heights from a rational generating function.
    coeffs: [a0, a1, ..., a_{order-1}, b0, b1, ..., b_{order-1}]
    The generating function is A(z)/B(z) where A(z) = sum a_i z^i, B(z) = sum b_i z^i
    We use this to create a structured autocorrelation.
    """
    a = coeffs[:order]
    b = coeffs[order:2*order]
    
    # Generate impulse response of the rational filter
    # h[n] = sum_{k=0}^{min(n,order-1)} a[k] * delta[n-k] - sum_{k=1}^{min(n,order-1)} b[k] * h[n-k]
    # But we want a periodic sequence, so use a different approach
    
    # Use the rational function to create a smooth periodic signal
    # h[n] = real part of sum_j c_j * exp(2*pi*i*f_j*n/N)
    # where f_j are chosen to give good autocorrelation properties
    
    # Alternative: use the coefficients to define a low-pass filter response
    # and modulate it periodically
    
    # Simple approach: use the coefficients to create a smooth periodic function
    # h[n] = 0.5 + sum_{k=1}^{order-1} (a[k] - 0.5*b[k]) * cos(2*pi*k*n/N)
    # This ensures h is periodic and smooth
    
    n = np.arange(N)
    h = np.zeros(N)
    h[:] = 0.5
    
    for k in range(1, order):
        # Use a_k and b_k to determine the amplitude of the k-th harmonic
        amp = a[k] - 0.5 * b[k] if k < len(a) else 0
        h += amp * np.cos(2 * np.pi * k * n / N)
    
    # Also add a linear term from a_0 and b_0 to adjust the mean
    h += (a[0] - 0.5 * b[0]) * 0.1  # small adjustment
    
    # Clip to [0,1]
    h = np.clip(h, 0.0, 1.0)
    
    # Project to exact sum N/2
    s = np.sum(h)
    resid = N / 2.0 - s
    h = h + resid / N
    h = np.clip(h, 0.0, 1.0)
    
    # Final exact fix
    s = np.sum(h)
    resid = N / 2.0 - s
    if abs(resid) > 1e-12:
        if resid > 0:
            idx = np.argmax(1.0 - h)
        else:
            idx = np.argmax(h)
        room = 1.0 - h[idx] if resid > 0 else h[idx]
        add = min(resid, room) if resid > 0 else max(resid, -room)
        h[idx] += add
    
    s = np.sum(h)
    h[-1] += (N / 2.0 - s)
    h = np.clip(h, 0.0, 1.0)
    
    s = np.sum(h)
    h[-1] += (N / 2.0 - s)
    
    return h

def _overlap(h, k):
    return np.dot(h, np.roll(h, k))

def _max_overlap(h):
    ks = np.arange(1, N)
    return np.max([_overlap(h, k) for k in ks])

def construct():
    import time
    t0 = time.time()
    DEADLINE = 135.0  # safety margin before 150s kill
    
    best_h = np.full(N, 0.5)
    best_val = _max_overlap(best_h)
    
    # Try different orders of the rational function
    orders = [2, 3, 4, 5]
    
    for order in orders:
        if time.time() > DEADLINE - 10.0:
            break
        
        # Initialize coefficients
        # a[0] and b[0] control the DC component
        # a[k], b[k] for k>=1 control the k-th harmonic
        rng = np.random.default_rng(order)
        
        # Try multiple random initializations
        for init in range(20):
            if time.time() > DEADLINE - 5.0:
                break
            
            # Random initialization
            a = rng.uniform(0.3, 0.7, order)
            b = rng.uniform(0.3, 0.7, order)
            coeffs0 = np.concatenate([a, b])
            
            def objective(coeffs):
                h = _compute_h_from_rational(coeffs, order)
                ks = np.arange(1, N)
                ov = np.array([_overlap(h, k) for k in ks])
                # log-sum-exp approximation of max
                m = np.max(ov)
                lse = m + np.log(np.sum(np.exp(ov - m)))
                return lse
            
            def objective_grad(coeffs):
                h = _compute_h_from_rational(coeffs, order)
                ks = np.arange(1, N)
                ov = np.array([_overlap(h, k) for k in ks])
                m = np.max(ov)
                e = np.exp(ov - m)
                w = e / np.sum(e)
                
                # Numerical gradient (finite differences)
                eps = 1e-7
                grad = np.zeros_like(coeffs)
                for i in range(len(coeffs)):
                    coeffs_plus = coeffs.copy()
                    coeffs_plus[i] += eps
                    h_plus = _compute_h_from_rational(coeffs_plus, order)
                    ov_plus = np.array([_overlap(h_plus, k) for k in ks])
                    m_plus = np.max(ov_plus)
                    lse_plus = m_plus + np.log(np.sum(np.exp(ov_plus - m_plus)))
                    
                    coeffs_minus = coeffs.copy()
                    coeffs_minus[i] -= eps
                    h_minus = _compute_h_from_rational(coeffs_minus, order)
                    ov_minus = np.array([_overlap(h_minus, k) for k in ks])
                    m_minus = np.max(ov_minus)
                    lse_minus = m_minus + np.log(np.sum(np.exp(ov_minus - m_minus)))
                    
                    grad[i] = (lse_plus - lse_minus) / (2 * eps)
                
                return grad
            
            # Optimize
            res = minimize(objective, coeffs0, jac=objective_grad,
                           method='L-BFGS-B',
                           options={'maxiter': 100, 'maxfun': 300, 'ftol': 1e-10, 'gtol': 1e-8})
            
            h = _compute_h_from_rational(res.x, order)
            val = _max_overlap(h)
            
            if val < best_val:
                best_val = val
                best_h = h.copy()
    
    # Final exact-sum correction
    s = np.sum(best_h)
    resid = N / 2.0 - s
    best_h = best_h + resid / N
    best_h = np.clip(best_h, 0.0, 1.0)
    
    s = np.sum(best_h)
    resid = N / 2.0 - s
    if abs(resid) > 1e-12:
        if resid > 0:
            idx = np.argmax(1.0 - best_h)
        else:
            idx = np.argmax(best_h)
        room = 1.0 - best_h[idx] if resid > 0 else best_h[idx]
        add = min(resid, room) if resid > 0 else max(resid, -room)
        best_h[idx] += add
    
    s = np.sum(best_h)
    best_h[-1] += (N / 2.0 - s)
    best_h = np.clip(best_h, 0.0, 1.0)
    
    s = np.sum(best_h)
    best_h[-1] += (N / 2.0 - s)
    
    return best_h
