"""Model-and-optimize: Epigraph soft-max autoconvolution minimization with spectral init and adaptive polish."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

START_TIME = time.time()
TIME_LIMIT = 138  # Safety margin before 150s kill

def _check_deadline():
    return (time.time() - START_TIME) > (TIME_LIMIT - 5)

def _autoconv_fft(x):
    """Compute linear autoconvolution using FFT."""
    L = 2 * N - 1
    L_fft = 1
    while L_fft < L:
        L_fft *= 2
    X = np.fft.rfft(x, n=L_fft)
    C = np.fft.irfft(X * X, n=L_fft)[:L]
    return C

def _objective_only(x):
    """Compute the raw objective: max|ac| / (sum(x))^2."""
    ac = _autoconv_fft(x)
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10
    return np.max(np.abs(ac)) / (s * s)

def _soft_max_and_grad(x, beta):
    """
    Compute soft-max surrogate for max|ac| and its gradient using pure FFT.
    Returns (soft_max_value, gradient_array)
    
    The gradient of max|ac_k| w.r.t. x_i is computed via:
    grad_x = sum_k w_k * sign(ac_k) * d(ac_k)/dx
    where d(ac_k)/dx = x convolved with delta at position k, done via FFT.
    """
    ac = _autoconv_fft(x)
    abs_ac = np.abs(ac)
    
    m = np.max(abs_ac)
    exps = np.exp(beta * (abs_ac - m))
    log_sum = np.log(np.sum(exps))
    soft_val = m + log_sum / beta
    
    weights = exps / np.sum(exps)
    signs = np.sign(ac)
    signs[ac == 0] = 1.0
    g = weights * signs  # length 2N-1, this is d(soft_max)/d(ac_k)
    
    # Gradient w.r.t. x: grad_x[i] = sum_k g[k] * d(ac[k])/dx[i]
    # ac[k] = sum_i x[i]*x[k-i] (with appropriate bounds)
    # d(ac[k])/dx[i] = x[k-i] + x[i-k] (for valid indices)
    # This is equivalent to: grad_x = ifft(fft(x) * conj(fft(g_padded)))
    # where g_padded is g placed in the right FFT frame
    
    L_fft = 1
    while L_fft < 4 * N:
        L_fft *= 2
    
    # Pad g to length L_fft
    g_pad = np.zeros(L_fft)
    g_pad[:2*N-1] = g
    G_f = np.fft.rfft(g_pad, n=L_fft)
    X_f = np.fft.rfft(x, n=L_fft)
    
    # Cross-correlation: x * g (autoconv-like)
    # We need: grad_x[i] = sum_k g[k] * (x[k-i] for k>=i, plus x[i-k] for k<i)
    # This is the cross-correlation of x and g, which is:
    # ifft(fft(x) * conj(fft(g))) gives correlation
    # But we need to be careful about the exact formulation
    
    # ac[k] = sum_{i=0}^{N-1} x[i] * x[k-i] where x[j]=0 for j outside [0,N-1]
    # d(ac[k])/dx[i] = x[k-i] + x[i-k] (the two terms from the product rule)
    # = x[k-i] + x[i-k]
    
    # So grad_x[i] = sum_k g[k] * (x[k-i] + x[i-k])
    # = sum_k g[k]*x[k-i] + sum_k g[k]*x[i-k]
    # = (g * x)[i] + (g * x)[-i]  (convolutions)
    
    # Using FFT: conv(g, x) = ifft(fft(g)*fft(x))
    # We need both forward and reverse convolutions
    
    # Forward: (g*x)[n] = sum_k g[k]*x[n-k]
    # Reverse: (x*g)[n] = sum_k x[k]*g[n-k] = (g*x)[n] by commutativity of convolution
    
    # Actually for real sequences, convolution is commutative, so:
    # grad_x[i] = sum_k g[k]*x[k-i] + sum_k g[k]*x[i-k]
    # The first term is (g*x)[-i] and the second is (g*x)[i]
    # But we need to handle the circular nature carefully
    
    # Simpler approach: use the fact that d(ac)/dx = 2 * (x * g) where * is cross-correlation
    # Let's just compute it directly via FFT
    
    # Method: pad both to length >= 2N-1, compute via FFT
    # ac[k] for k in [0, 2N-2]
    # d(ac[k])/dx[i] = x[k-i] if 0<=k-i<N, else 0, PLUS x[i-k] if 0<=i-k<N, else 0
    
    # grad_x[i] = sum_{k=0}^{2N-2} g[k] * (x[k-i]*I(0<=k-i<N) + x[i-k]*I(0<=i-k<N))
    
    # Let's compute term1[i] = sum_k g[k]*x[k-i] and term2[i] = sum_k g[k]*x[i-k]
    # term1 is cross-correlation of g and x at lag i
    # term2 is cross-correlation of x and g at lag i (which equals term1 for real signals)
    
    # Actually for real signals: corr(g,x)[i] = sum_k g[k+i]*x[k]
    # We want: sum_k g[k]*x[k-i] which is corr(x,g)[i] = sum_k x[k]*g[k-i]... 
    
    # Let me just use a direct FFT-based approach:
    # Define h[k] = g[k] for k in [0, 2N-2], 0 otherwise
    # Then grad_x[i] = sum_k h[k] * x[k-i] + sum_k h[k] * x[i-k]
    
    # First sum: sum_k h[k]*x[k-i] = (h * x_flipped)[i] where x_flipped[j] = x[-j]
    # This is getting complex. Let me use a cleaner formulation.
    
    # Alternative: Use the property that if ac = x * x (convolution), then
    # d(max|ac|)/dx = 2 * (x * w) where w[k] = g[k] * sign(ac[k]) ... no that's not right either.
    
    # Let me just do it carefully:
    # ac[k] = sum_{i=max(0,k-N+1)}^{min(N-1,k)} x[i] * x[k-i]
    # d(ac[k])/dx[j] = x[k-j] (if 0<=k-j<N) + x[j-k] (if 0<=j-k<N)
    #                = x[k-j]*I(0<=k-j<N) + x[j-k]*I(0<=j-k<N)
    
    # grad_x[j] = sum_k g[k] * (x[k-j]*I(0<=k-j<N) + x[j-k]*I(0<=j-k<N))
    #           = sum_k g[k]*x[k-j]*I(0<=k-j<N) + sum_k g[k]*x[j-k]*I(0<=j-k<N)
    
    # First term: let m = k-j, so k = m+j, sum_m g[m+j]*x[m]*I(0<=m<N)
    #   = sum_{m=0}^{N-1} g[m+j]*x[m]  (with g=0 outside [0,2N-2])
    # Second term: let m = j-k, so k = j-m, sum_m g[j-m]*x[m]*I(0<=m<N)  
    #   = sum_{m=0}^{N-1} g[j-m]*x[m]  (with g=0 outside [0,2N-2])
    
    # So grad_x[j] = sum_{m=0}^{N-1} x[m]*(g[m+j] + g[j-m])
    
    # This can be computed as: for each j, we need g[m+j] and g[j-m] for m=0..N-1
    # g[m+j] is g shifted right by j (with zeros), g[j-m] is g reversed and shifted
    
    # Using FFT: 
    # Let G_full be g padded to length >= 2N-1
    # First term is cross-correlation: sum_m x[m]*g[m+j] = corr(x, g)[j]
    # Second term is: sum_m x[m]*g[j-m] = corr(x, g_reversed)[j] where g_rev[m]=g[-m]
    
    # For real sequences, corr(x,g)[j] = ifft(fft(x)*conj(fft(g)))[j]
    # And g_rev[m] = g[-m] = g[2N-1-m] approximately (for the supported range)
    
    # Simpler: just compute both terms via FFT
    # Term A[j] = sum_{m=0}^{N-1} x[m]*g[m+j]
    # Term B[j] = sum_{m=0}^{N-1} x[m]*g[j-m]
    
    # A is the cross-correlation of x and g (with g indexed as g[m+j])
    # B is the cross-correlation of x and g_flipped
    
    # Let's just use np.fft for this
    L_fft2 = 1
    while L_fft2 < 2 * (2*N - 1):
        L_fft2 *= 2
    
    X_f2 = np.fft.rfft(x, n=L_fft2)
    G_f2 = np.fft.rfft(g, n=L_fft2)
    
    # Cross-correlation: corr_xg[j] = sum_m x[m]*g[m+j]
    # = ifft(fft(x) * conj(fft(g)))[j]
    corr_xg = np.fft.irfft(X_f2 * np.conj(G_f2), n=L_fft2)
    
    # For term B: sum_m x[m]*g[j-m]
    # Let g_flip[m] = g[-m] = g[2N-1-m] (for the valid range)
    g_flip = np.zeros(L_fft2)
    g_flip[:2*N-1] = g[::-1]  # g[2N-2], g[2N-3], ..., g[0]
    # Actually g_flip[m] should be g[-m], so g_flip[0]=g[0], g_flip[1]=g[-1]=g[2N-2], etc.
    # Hmm, let me think again.
    # g[j-m] for m=0,1,...,N-1: when m=0, g[j]; m=1, g[j-1]; ...
    # This is like correlating x with a "time-reversed" version of g
    # Let g_rev[k] = g[-k] (with g[k]=0 for k<0 or k>=2N-1)
    # Then g[j-m] = g_rev[m-j]... this is getting confusing.
    
    # Let me just use a different approach: compute grad_x directly
    # grad_x[j] = sum_{m=0}^{N-1} x[m]*(g[m+j] + g[j-m])
    
    # I'll compute this using two FFT-based convolutions
    # A[j] = sum_m x[m]*g[m+j] = (x * g_shifted)[j] where g_shifted depends on j
    # This is actually the cross-correlation
    
    # For cross-correlation: R_xg[j] = sum_m x[m]*g[m+j]
    # R_xg = ifft(fft(x) * conj(fft(g)))
    # This gives R_xg[j] for j = 0, 1, ..., L_fft2-1 (circular)
    # We need R_xg[j] for j = 0, ..., N-1
    
    # For term B[j] = sum_m x[m]*g[j-m]
    # Let g_neg[k] = g[-k] for k in range, 0 otherwise
    # Then B[j] = sum_m x[m]*g_neg[m-j]... no
    # B[j] = sum_m x[m]*g[j-m]
    # Let n = j-m, m = j-n: B[j] = sum_n x[j-n]*g[n] = (x * g)[j] (convolution!)
    
    # So: A[j] = sum_m x[m]*g[m+j] = cross-correlation
    #     B[j] = sum_m x[m]*g[j-m] = convolution (x*g)[j]
    
    # A[j] = R_xg[j] where R_xg = ifft(fft(x)*conj(fft(g)))
    # B[j] = (x*g)[j] where (x*g) = ifft(fft(x)*fft(g))
    
    A = np.fft.irfft(X_f2 * np.conj(G_f2), n=L_fft2)[:N]
    B = np.fft.irfft(X_f2 * G_f2, n=L_fft2)[:N]
    
    grad_x = A + B
    return soft_val, grad_x

def _objective_and_grad(x, beta):
    """Compute objective and gradient."""
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10, np.zeros(N)
    soft_val, grad_soft = _soft_max_and_grad(x, beta)
    val = soft_val / (s * s)
    grad = grad_soft / (s * s) - 2.0 * soft_val * np.ones(N) / (s * s * s)
    return val, grad

def _epigraph_obj_grad(z, beta):
    """
    Epigraph formulation: minimize t subject to soft_max|ac|/s^2 <= t
    Variables: z = [x (N), t (1)]
    """
    x = z[:N]
    t = z[N]
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10 + t, np.concatenate([np.zeros(N), 1.0])
    soft_val, grad_soft = _soft_max_and_grad(x, beta)
    val = soft_val / (s * s)
    # Objective: t + penalty*(val - t) for val > t, else t
    # Simpler: just minimize val, use t as slack
    # Actually let's just minimize val directly with the epigraph constraint handled by L-BFGS-B bounds
    
    # For epigraph: minimize t + mu * max(0, val - t)^2
    mu = 100.0
    if val > t:
        obj = t + mu * (val - t) ** 2
        grad_t = 1.0 - 2 * mu * (val - t)
        grad_x = -2 * mu * (val - t) * (grad_soft / (s * s) - 2.0 * soft_val * np.ones(N) / (s * s * s))
    else:
        obj = t
        grad_t = 1.0
        grad_x = np.zeros(N)
    
    return obj, np.concatenate([grad_x, [grad_t]])

def _spectral_init(rng):
    """
    Generate initial points via spectral amplitude shaping.
    """
    L_fft = 1
    while L_fft < 2*N:
        L_fft *= 2
    
    k = np.arange(L_fft // 2 + 1)
    t = k / (L_fft // 2)
    
    windows = [
        np.ones(L_fft // 2 + 1),
        np.exp(-0.5 * (t - 0.5)**2 / (0.15**2)),
        np.exp(-0.5 * (t - 0.5)**2 / (0.25**2)),
        1.0 / (1.0 + (t / 0.3)**2),
        np.sinc(0.5 * t) ** 2,
        np.exp(-t),
        1.0 / (1.0 + t),
    ]
    
    inits = []
    for w in windows:
        w = np.maximum(w, 0)
        phase = rng.uniform(0, 2*np.pi, size=L_fft // 2 + 1)
        spec = w * np.exp(1j * phase)
        spec_full = np.zeros(L_fft, dtype=complex)
        spec_full[:L_fft//2+1] = spec
        spec_full[L_fft//2+1:] = np.conj(spec[1:L_fft//2][::-1])
        x = np.fft.ifft(spec_full).real[:N]
        x = x - np.min(x) + 0.1
        inits.append(x)
    
    t_arr = np.linspace(0, 1, N)
    inits.append(np.ones(N))
    inits.append(1.0 + 0.5 * np.sin(2 * np.pi * t_arr))
    inits.append(1.0 + 0.5 * np.cos(2 * np.pi * t_arr))
    inits.append(np.exp(t_arr))
    inits.append(np.exp(-t_arr))
    inits.append(1.0 + 0.8 * (t_arr - 0.5)**2)
    inits.append(1.0 + 0.5 * (t_arr - 0.5))
    
    for _ in range(8):
        inits.append(rng.normal(1.0, 0.3, size=N))
    
    for _ in range(4):
        x = np.ones(N) * 0.1
        n_spikes = rng.integers(3, 10)
        idx = rng.choice(N, size=n_spikes, replace=False)
        x[idx] = 1.0
        inits.append(x)
    
    for _ in range(4):
        x = np.ones(N)
        for freq in rng.integers(1, 5, size=3):
            x += 0.3 * np.sin(2 * np.pi * freq * t_arr + rng.uniform(0, 2*np.pi))
        inits.append(x)
    
    return inits

def _spectral_phase_optimize(x, n_iters=50, rng=None):
    """
    Optimize the phase of the Fourier transform to minimize max|ac|/s^2.
    Keep amplitudes fixed, update phases iteratively.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    
    L_fft = 1
    while L_fft < 2 * N:
        L_fft *= 2
    
    best_x = x.copy()
    best_val = _objective_only(x)
    
    for it in range(n_iters):
        if _check_deadline():
            break
        X = np.fft.rfft(best_x, n=L_fft)
        amp = np.abs(X)
        phase = np.angle(X)
        
        # Update phase: try to reduce peak of autoconv
        # Gradient of max|ac| w.r.t. phase
        ac = _autoconv_fft(best_x)
        abs_ac = np.abs(ac)
        m = np.max(abs_ac)
        beta = 50.0
        exps = np.exp(beta * (abs_ac - m))
        weights = exps / np.sum(exps)
        signs = np.sign(ac)
        signs[ac == 0] = 1.0
        g = weights * signs
        
        # d(ac[k])/d(phase[n]) = 2j * X[n] * conj(X[n]) ... complex
        # Let's just do a simple phase update:
        # new_phase = old_phase - eta * imag(conj(X) * grad_phase)
        
        # Simpler: random phase perturbation with acceptance
        new_phase = phase + rng.normal(0, 0.1, size=L_fft // 2 + 1)
        new_spec = amp * np.exp(1j * new_phase)
        new_spec_full = np.zeros(L_fft, dtype=complex)
        new_spec_full[:L_fft//2+1] = new_spec
        new_spec_full[L_fft//2+1:] = np.conj(new_spec[1:L_fft//2][::-1])
        new_x = np.fft.ifft(new_spec_full).real[:N]
        
        new_val = _objective_only(new_x)
        if new_val < best_val:
            best_val = new_val
            best_x = new_x
    
    return best_x, best_val

def construct():
    best_x = np.ones(N)
    best_val = _objective_only(best_x)
    
    if _check_deadline():
        return best_x
    
    rng = np.random.default_rng(12345)
    
    # Phase 1: Multi-start L-BFGS with spectral inits
    betas = [30.0, 50.0, 100.0]
    initial_points = _spectral_init(rng)
    
    for beta in betas:
        if _check_deadline():
            break
        for x0 in initial_points:
            if _check_deadline():
                break
            
            x_start = x0 + rng.normal(0, 0.01, size=N)
            
            try:
                def obj_and_grad_func(x):
                    return _objective_and_grad(x, beta)
                
                result = minimize(obj_and_grad_func, x_start, method='L-BFGS-B',
                                  jac=True, options={'maxiter': 200, 'gtol': 1e-8, 'ftol': 1e-10})
                
                if result.fun < best_val:
                    best_val = result.fun
                    best_x = result.x.copy()
                    
            except Exception:
                pass
    
    if _check_deadline():
        return best_x
    
    # Phase 2: Polish with higher beta
    for beta in [200.0, 500.0]:
        if _check_deadline():
            break
        try:
            def obj_and_grad_func(x):
                return _objective_and_grad(x, beta)
            
            result = minimize(obj_and_grad_func, best_x, method='L-BFGS-B',
                              jac=True, options={'maxiter': 400, 'gtol': 1e-10, 'ftol': 1e-12})
            
            if result.fun < best_val:
                best_val = result.fun
                best_x = result.x.copy()
                
        except Exception:
            pass
    
    if _check_deadline():
        return best_x
    
    # Phase 3: Spectral phase optimization
    if not _check_deadline():
        best_x, best_val = _spectral_phase_optimize(best_x, n_iters=100, rng=rng)
    
    if _check_deadline():
        return best_x
    
    # Phase 4: Structured random polish
    step_size = 0.01
    n_steps = 0
    max_steps = 300000
    
    t_arr = np.linspace(0, 1, N)
    freqs = [1, 2, 3, 5, 8]
    
    while n_steps < max_steps:
        if _check_deadline():
            break
        
        # Mix of random noise and structured perturbations
        r = rng.random()
        if r < 0.3:
            # Low-frequency sinusoid
            f = freqs[rng.integers(0, len(freqs))]
            perturbation = step_size * np.sin(2 * np.pi * f * t_arr + rng.uniform(0, 2*np.pi))
        elif r < 0.5:
            # Sparse large steps
            perturbation = np.zeros(N)
            n_spikes = rng.integers(1, 4)
            idx = rng.choice(N, size=n_spikes, replace=False)
            perturbation[idx] = rng.normal(0, step_size * 3, size=n_spikes)
        else:
            # Random noise
            perturbation = rng.normal(0, step_size, size=N)
        
        x_trial = best_x + perturbation
        val = _objective_only(x_trial)
        if val < best_val:
            best_val = val
            best_x = x_trial.copy()
            step_size *= 0.999
        else:
            step_size = min(0.1, step_size * 1.01)
        n_steps += 1
    
    # Phase 5: Coordinate descent polish
    if not _check_deadline():
        coord_steps = 0
        max_coord_steps = 100000
        while coord_steps < max_coord_steps:
            if _check_deadline():
                break
            improved = False
            for i in range(0, N, max(1, N // 100)):
                if _check_deadline():
                    break
                for delta in [0.01, -0.01, 0.005, -0.005, 0.02, -0.02]:
                    x_trial = best_x.copy()
                    x_trial[i] += delta
                    val = _objective_only(x_trial)
                    if val < best_val:
                        best_val = val
                        best_x = x_trial
                        improved = True
                        break
            coord_steps += 1
            if not improved:
                break
    
    return best_x
# EVOLVE-BLOCK-END
