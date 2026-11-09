"""Minimize peak of autoconvolution for nonnegative step function on [-1/4, 1/4]."""
import numpy as np
from scipy.optimize import minimize
import time

N = 600
DELTA_T = 0.5 / N  # interval width: (1/4 - (-1/4)) / 600 = 0.5/600

def _autoconvolution_peaks(h):
    """Compute the autoconvolution values at all discrete lags for a step function.
    
    h: array of N heights
    Returns array of length 2N-1: autoconv at lags -(N-1)*dt to (N-1)*dt
    
    For step functions, autoconv at lag k*dt is:
    sum_{i} h[i] * h[i+k] * dt  (for k >= 0, with appropriate indexing)
    Actually, for a step function on equal intervals, the autoconvolution at
    offset m (in interval units) is:
    (1/dt) * sum_i h[i] * h[i+m] * dt^2 = dt * sum_i h[i]*h[i+m]
    Wait, let me be more careful.
    
    f(x) = h[i] for x in [x_i, x_{i+1})
    (f*f)(t) = integral f(x) f(t-x) dx
    
    For t = m * dt (m integer), the integral can be computed exactly.
    For m >= 0: (f*f)(m*dt) = dt * sum_{i=0}^{N-1-m} h[i] * h[i+m]
    For m < 0: (f*f)(m*dt) = (f*f)(|m|*dt) by symmetry
    
    But the peak might not be exactly at integer lags! However, for a step function,
    the autoconvolution is piecewise linear, so the maximum on each interval 
    [m*dt, (m+1)*dt] occurs at one of the endpoints. So we just need to check
    integer lags m = -(N-1), ..., 0, ..., N-1.
    """
    h = np.asarray(h, dtype=np.float64)
    n = len(h)
    # Autoconv at lag m (in interval units), m = 0, 1, ..., n-1
    # (f*f)(m*dt) = dt * sum_{i=0}^{n-1-m} h[i] * h[i+m]
    lags = np.arange(n)
    conv = np.array([np.dot(h[:n-m], h[m:]) for m in lags]) * DELTA_T
    # By symmetry, full autoconv is symmetric, so peaks are at m=0..n-1
    return conv

def _score(h):
    """Compute the normalized peak: max(autoconv) / (integral f)^2"""
    h = np.asarray(h, dtype=np.float64)
    conv = _autoconvolution_peaks(h)
    peak = np.max(conv)
    integral = np.sum(h) * DELTA_T
    if integral < 1e-15:
        return 1e15
    return peak / (integral * integral)

def construct():
    """Return sequence of N nonnegative step heights minimizing normalized autoconv peak."""
    deadline = time.time() + 110  # safety margin before 120s limit
    best_h = np.ones(N)
    best_score = _score(best_h)
    
    # Log-parameterization: h[i] = exp(log_h[i]) to ensure nonnegativity
    # But this can cause numerical issues. Alternative: use h[i] = softplus(z[i]) or just project.
    # Let's use direct optimization with projection to nonnegative orthant.
    
    def objective(z):
        """z: unconstrained variables, h = max(z, 0) or we use a smooth nonneg constraint"""
        # Use h = softplus(z) = log(1 + exp(z)) for smoothness, but clip for stability
        z_clipped = np.clip(z, -50, 50)
        h = np.log1p(np.exp(z_clipped))
        # Normalize: the score is scale-invariant! peak/(integral)^2 is invariant under scaling.
        # So we can fix the norm. Let's normalize h to have sum(h)*dt = 1, i.e., integral = 1.
        s = np.sum(h) * DELTA_T
        if s < 1e-15:
            return 1e15
        h_norm = h / s  # now integral = 1
        conv = _autoconvolution_peaks(h_norm)
        peak = np.max(conv)
        return peak
    
    def gradient(z):
        z_clipped = np.clip(z, -50, 50)
        h = np.log1p(np.exp(z_clipped))
        s = np.sum(h) * DELTA_T
        if s < 1e-15:
            return np.zeros_like(z)
        h_norm = h / s
        
        # Gradient of peak w.r.t. h_norm, then chain rule
        conv = _autoconvolution_peaks(h_norm)
        # Soft max for smoothness
        beta = 50.0
        m = np.max(conv)
        conv_shifted = (conv - m) * beta
        conv_clipped = np.clip(conv_shifted, -50, 50)
        logsumexp = np.log(np.sum(np.exp(conv_clipped)))
        weights = np.exp(conv_clipped - logsumexp)  # softmax weights
        # d(peak_smooth)/d(conv[j]) = weights[j] * beta
        # conv[m] = dt * sum_i h_norm[i] * h_norm[i+m]
        # d(conv[m])/d(h_norm[k]) = dt * (h_norm[k+m] if k+m < N else 0) + dt * (h_norm[k-m] if k-m >= 0 else 0)
        # Actually: conv[m] = dt * sum_{i=0}^{N-1-m} h_norm[i] * h_norm[i+m]
        # d(conv[m])/d(h_norm[k]) = dt * (h_norm[k+m] if k+m < N else 0) + dt * (h_norm[k-m] if k-m >= 0 and m > 0 else 0)
        # Wait, for m=0: conv[0] = dt * sum_i h_norm[i]^2, d/d(h_norm[k]) = 2*dt*h_norm[k]
        # For m>0: conv[m] = dt * sum_{i=0}^{N-1-m} h_norm[i]*h_norm[i+m]
        # d/d(h_norm[k]) = dt * (h_norm[k+m] if k+m < N else 0) + dt * (h_norm[k-m] if k-m >= 0 else 0)
        
        grad_conv = np.zeros(N)
        for m_idx in range(N):
            if weights[m_idx] < 1e-15:
                continue
            w = weights[m_idx] * beta
            if m_idx == 0:
                grad_conv += 2.0 * w * DELTA_T * h_norm
            else:
                # h_norm[k+m] contributes when k+m < N, i.e., k < N-m
                if m_idx < N:
                    grad_conv[:N-m_idx] += w * DELTA_T * h_norm[m_idx:]
                    # h_norm[k-m] contributes when k-m >= 0, i.e., k >= m
                    grad_conv[m_idx:] += w * DELTA_T * h_norm[:N-m_idx]
        
        # Now grad of objective w.r.t. h_norm is grad_conv
        # h_norm = h / s, where s = sum(h)*dt
        # d(h_norm[k])/d(h[j]) = (delta[k,j]*s - h[k]*dt) / s^2 = (delta[k,j] - h_norm[k]*dt) / s
        # Wait: h_norm[k] = h[k]/s, s = dt*sum(h)
        # d(h_norm[k])/d(h[j]) = (delta[k,j]*s - h[k]*dt) / s^2
        # = (delta[k,j] - h[k]*dt/s) / s = (delta[k,j] - h_norm[k]*dt) / s
        
        # Objective = peak(h_norm), so d(obj)/d(h[j]) = sum_k grad_conv[k] * d(h_norm[k])/d(h[j])
        # = sum_k grad_conv[k] * (delta[k,j] - h_norm[k]*dt) / s
        # = (grad_conv[j] - dt*sum_k grad_conv[k]*h_norm[k]) / s
        
        sum_grad_h = np.dot(grad_conv, h_norm)
        grad_h = (grad_conv - DELTA_T * sum_grad_h) / s
        
        # Now d(h[j])/d(z[j]) = sigmoid(z_clipped[j]) (derivative of log1p(exp(z)))
        # But only if z_clipped[j] is not clipped
        sigma = np.zeros_like(z)
        mask = (z > -50) & (z < 50)
        sigma[mask] = 1.0 / (1.0 + np.exp(-z[mask]))
        
        return grad_h * sigma
    
    # Initial guess: start with a smooth bump
    z0 = np.zeros(N)
    # Add some structure: Gaussian-like
    x = np.linspace(-0.25, 0.25, N)
    z0 = -20 * x**2  # will be transformed by softplus
    
    result = None
    try:
        result = minimize(objective, z0, method='BFGS', jac=gradient,
                         options={'maxiter': 200, 'gtol': 1e-8, 'disp': False})
    except Exception:
        pass
    
    if result is not None and result.success and result.fun < best_score:
        h_opt = np.log1p(np.exp(np.clip(result.x, -50, 50)))
        s = np.sum(h_opt) * DELTA_T
        if s > 1e-15:
            h_opt = h_opt / s
            score_opt = _score(h_opt)
            if score_opt < best_score:
                best_h = h_opt
                best_score = score_opt
    
    # Try a few random restarts if time permits
    np.random.seed(42)
    for restart in range(5):
        if time.time() > deadline - 20:
            break
        z0_rand = np.random.randn(N) * 0.5
        try:
            res = minimize(objective, z0_rand, method='BFGS', jac=gradient,
                         options={'maxiter': 100, 'gtol': 1e-7, 'disp': False})
        except Exception:
            continue
        if res is not None and res.fun < best_score:
            h_opt = np.log1p(np.exp(np.clip(res.x, -50, 50)))
            s = np.sum(h_opt) * DELTA_T
            if s > 1e-15:
                h_opt = h_opt / s
                score_opt = _score(h_opt)
                if score_opt < best_score:
                    best_h = h_opt
                    best_score = score_opt
    
    # Ensure nonnegativity and return
    best_h = np.maximum(best_h, 0.0)
    return list(best_h)
