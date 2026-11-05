"""Rewritten program for third autocorrelation inequality optimization."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.fft import fft, ifft
from scipy.optimize import minimize

def _eval_value(h):
    """Compute the objective value."""
    s = np.sum(h)
    if abs(s) < 1e-12:
        return np.inf
    conv = np.convolve(h, h)
    return abs(2.0 * N * np.max(conv) / (s * s))

def _convolve_fft(h):
    """Compute full autoconvolution using FFT."""
    n_conv = 2 * len(h) - 1
    n_fft = 1
    while n_fft < n_conv:
        n_fft *= 2
    H = fft(h, n=n_fft)
    conv = ifft(H * H, n=n_fft).real[:n_conv]
    return conv

def _soft_max(x, beta):
    """Numerically stable soft maximum."""
    m = np.max(x)
    if not np.isfinite(m):
        return m
    exp_vals = np.exp(beta * (x - m))
    sm = m + np.log(np.sum(exp_vals)) / beta
    return sm

def construct():
    """
    Returns a sequence of 400 signed step heights.
    
    Strategy: We work in the Fourier domain. The autoconvolution c[k] is the
    inverse DFT of |H|^2. We want to minimize max_k c[k].
    
    Key insight: If we make |H|^2 as flat as possible (constant spectrum),
    then c[k] = delta[k] (white noise autocorrelation), giving min peak = 1/N.
    But we need sum(h) != 0, so c[0] = (sum h)^2 is fixed.
    
    Actually, the peak is at lag 0: c[0] = sum(h^2) >= (sum h)^2 / N by Cauchy-Schwarz.
    So value >= 2*N * (sum h)^2/N / (sum h)^2 = 2.
    
    Wait, that's not right. c[0] = sum(h_i^2), and the max over ALL lags includes
    lag 0. So max(conv) >= c[0] = sum(h^2).
    
    By Cauchy-Schwarz: sum(h^2) >= (sum h)^2 / N.
    So value >= 2*N * (sum h)^2/N / (sum h)^2 = 2.
    
    This means the value can never be below 2! But the seed gives 2.0, and
    the current program gets 1.31... Let me re-read.
    
    Oh wait, the max is over SIGNED entries. So if all conv entries are negative
    except c[0], and c[0] is the max, then... but c[0] = sum(h^2) >= 0.
    
    Hmm, but the problem says "the maximum is taken over its signed entries, with
    no absolute value inside np.max". So np.max(conv) could be negative if all
    entries are negative. But c[0] = sum(h^2) is always >= 0.
    
    So np.max(conv) >= c[0] = sum(h^2) >= (sum h)^2 / N.
    
    Therefore value = 2*N*max(conv)/(sum h)^2 >= 2*N*(sum h)^2/N/(sum h)^2 = 2.
    
    But the current program scores 1.31, which is less than 2. That contradicts
    my analysis. Let me check: for h = ones(400), sum(h) = 400, sum(h^2) = 400.
    conv[0] = 400, value = 2*400*400/400^2 = 2. Correct.
    
    How can value < 2? Only if max(conv) < sum(h^2), which is impossible since
    conv[0] = sum(h^2). Unless... the convolution is defined differently.
    
    np.convolve(h, h) with full mode: result[k] = sum_i h[i] * h[k-i].
    For k=0: sum_i h[i]*h[-i]. Since h is only defined for 0..N-1, h[-i] = 0 for i>0.
    So conv[0] = h[0]*h[0] = h[0]^2.
    
    For k=N-1: sum_i h[i]*h[N-1-i] = sum_i h[i]*h[N-1-i]. This is the full overlap.
    
    I had the indexing wrong! np.convolve(h, h, 'full')[k] = sum_i h[i]*h[k-i]
    where h[j] = 0 for j < 0 or j >= N.
    
    So conv[N-1] = sum_i h[i]*h[N-1-i] which is the "center" of the convolution.
    And conv[0] = h[0]^2, conv[N-1] = sum h[i]*h[N-1-i].
    
    The maximum could be at any lag. For h = ones(N), conv[k] = N - |k - (N-1)| for
    k = 0..2N-2. So max is at k = N-1, conv[N-1] = N.
    value = 2*N*N/N^2 = 2.
    
    So the key is: we want to minimize max_k conv[k] where conv = h*h (full conv).
    The center lag k=N-1 gives conv[N-1] = sum_i h[i]*h[N-1-i].
    
    If we make h[i]*h[N-1-i] negative for all i, then conv[N-1] could be small.
    But other lags might be large.
    
    Strategy: Make h symmetric or antisymmetric in a way that cancels the center.
    If h[i] = -h[N-1-i], then conv[N-1] = sum h[i]*h[N-1-i] = -sum h[i]^2 < 0.
    But then sum(h) = 0, which is not allowed.
    
    Let's try h[i] = 1 - 2*delta(i, N-1-i) type constructions.
    
    Better approach: Use the fact that we can make the autoconvolution have
    a small maximum by making h have a specific structure.
    
    Let me try: h[i] = cos(2*pi*i/N) + small_offset.
    The autoconvolution of a cosine is a cosine at double frequency.
    
    Actually, let me just do a direct optimization in the time domain using
    gradient descent with a smooth approximation of the max.
    """
    deadline = time.time() + 145.0
    
    # Start with a good initial guess
    # Try antisymmetric-like: h[i] = 1 + a*(-1)^i or similar
    best_h = np.ones(N)
    best_val = 2.0
    
    # Phase 1: Try parametric families
    # Family: h[i] = 1 + a*cos(2*pi*m*i/N + phi)
    for m in range(1, 30):
        if time.time() > deadline - 100:
            break
        for a in np.linspace(-1.5, 1.5, 31):
            for phi in [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi]:
                h = 1.0 + a * np.cos(2.0 * np.pi * m * np.arange(N) / N + phi)
                if abs(np.sum(h)) < 1e-10:
                    continue
                v = _eval_value(h)
                if v < best_val:
                    best_val = v
                    best_h = h.copy()
    
    # Family: h[i] = 1 + a*sin(2*pi*m*i/N + phi)
    for m in range(1, 30):
        if time.time() > deadline - 100:
            break
        for a in np.linspace(-1.5, 1.5, 31):
            for phi in [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi]:
                h = 1.0 + a * np.sin(2.0 * np.pi * m * np.arange(N) / N + phi)
                if abs(np.sum(h)) < 1e-10:
                    continue
                v = _eval_value(h)
                if v < best_val:
                    best_val = v
                    best_h = h.copy()
    
    # Phase 2: Gradient descent with soft-max in time domain
    # We minimize f(h) = 2*N*softmax(conv(h), beta)/(sum(h))^2
    # subject to sum(h) = N (fix DC to simplify)
    
    def _objective(h, beta):
        """Soft-max objective with sum(h) fixed to N."""
        if not np.all(np.isfinite(h)):
            return 1e10
        s = np.sum(h)
        if abs(s) < 1e-10:
            return 1e10
        h_norm = h * (N / s)  # Normalize to sum = N
        conv = _convolve_fft(h_norm)
        sm = _soft_max(conv, beta)
        return abs(2.0 * N * sm / (N * N))
    
    def _objective_grad(h, beta):
        """Objective with gradient for L-BFGS-B."""
        if not np.all(np.isfinite(h)):
            return 1e10, np.zeros(N)
        s = np.sum(h)
        if abs(s) < 1e-10:
            return 1e10, np.zeros(N)
        h_norm = h * (N / s)
        conv = _convolve_fft(h_norm)
        m = np.max(conv)
        if not np.isfinite(m):
            return 1e10, np.zeros(N)
        
        # Soft-max
        exp_vals = np.exp(beta * (conv - m))
        sm = m + np.log(np.sum(exp_vals)) / beta
        val = abs(2.0 * N * sm / (N * N))
        
        # Gradient via finite differences (slow but works)
        # For speed, use a smaller beta for gradient estimation
        return val, None
    
    # Use L-BFGS-B with numerical gradient
    for beta in [5.0, 20.0, 50.0, 100.0, 200.0, 500.0]:
        if time.time() > deadline - 30:
            break
        
        def obj_func(h, beta=beta):
            return _objective(h, beta)
        
        # Try multiple random restarts
        rng = np.random.RandomState(int(beta * 7))
        for trial in range(3):
            if time.time() > deadline - 30:
                break
            
            if trial == 0:
                h_init = best_h.copy()
            else:
                h_init = best_h + 0.1 * rng.randn(N)
            
            try:
                result = minimize(
                    obj_func,
                    h_init,
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50}
                )
                
                h_opt = result.x
                if np.all(np.isfinite(h_opt)) and abs(np.sum(h_opt)) > 1e-10:
                    v_opt = _eval_value(h_opt)
                    if v_opt < best_val:
                        best_val = v_opt
                        best_h = h_opt.copy()
            except Exception:
                continue
    
    # Phase 3: Fourier domain optimization
    # Work with Fourier coefficients H[k] = fft(h)[k]
    # conv = ifft(|H|^2)
    # We want to minimize max(conv) subject to H[0] = N (sum h = N)
    
    def _fourier_obj(H_mag, H_phase, beta):
        """Objective in Fourier domain."""
        H = H_mag * np.exp(1j * H_phase)
        # Fix DC
        H[0] = N + 0j
        h = ifft(H).real
        if not np.all(np.isfinite(h)):
            return 1e10
        s = np.sum(h)
        if abs(s) < 1e-10:
            return 1e10
        h_norm = h * (N / s)
        conv = _convolve_fft(h_norm)
        sm = _soft_max(conv, beta)
        return abs(2.0 * N * sm / (N * N))
    
    # Try flattening the spectrum
    H_best = fft(best_h)
    
    for beta in [10.0, 50.0, 200.0]:
        if time.time() > deadline - 20:
            break
        
        # Try to make |H|^2 flat (except DC)
        # This is equivalent to making h as close to white noise as possible
        # while keeping sum(h) = N
        
        # Simple approach: iterate between time and frequency domains
        for it in range(20):
            if time.time() > deadline - 20:
                break
            
            h = ifft(H_best).real
            s = np.sum(h)
            if abs(s) < 1e-10:
                break
            h_norm = h * (N / s)
            
            # Compute |H|^2
            H = fft(h_norm)
            power = np.abs(H)**2
            
            # Make power flatter: replace with geometric mean or similar
            # Keep DC fixed
            power_ac = power[1:]
            if np.min(power_ac) < 1e-10:
                power_ac = np.maximum(power_ac, 1e-10)
            
            # Flatten: new_power = (old_power * target)^(1/2) iteratively
            target = np.mean(power_ac)
            new_power = np.sqrt(power_ac * target)
            
            # Update H
            H_best[1:] = np.sqrt(new_power) * np.exp(1j * np.angle(H_best[1:]))
            H_best[0] = N + 0j
            
            # Check improvement
            h_new = ifft(H_best).real
            if np.all(np.isfinite(h_new)) and abs(np.sum(h_new)) > 1e-10:
                v_new = _eval_value(h_new)
                if v_new < best_val:
                    best_val = v_new
                    best_h = h_new.copy()
    
    # Phase 4: Final polish with high-beta soft-max
    if time.time() < deadline - 10:
        for beta in [500.0, 1000.0, 2000.0]:
            if time.time() > deadline - 10:
                break
            
            def obj_polish(h, beta=beta):
                return _objective(h, beta)
            
            try:
                result = minimize(
                    obj_polish,
                    best_h.copy(),
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50}
                )
                h_opt = result.x
                if np.all(np.isfinite(h_opt)) and abs(np.sum(h_opt)) > 1e-10:
                    v_opt = _eval_value(h_opt)
                    if v_opt < best_val:
                        best_val = v_opt
                        best_h = h_opt.copy()
            except Exception:
                continue
    
    # Ensure validity
    if not np.all(np.isfinite(best_h)) or abs(np.sum(best_h)) < 1e-12:
        return np.ones(N)
    
    return best_h

# EVOLVE-BLOCK-END
