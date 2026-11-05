"""Combined approach: smooth surrogate optimization with diverse initialization and local refinement."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.ndimage import gaussian_filter1d
from scipy.special import eval_chebyt

N = 400
DEADLINE = 148.0  # Safety margin before 150s kill

# Precompute Chebyshev basis
_x = np.linspace(-1, 1, N)
_basis_cache = {}

def _get_basis(degree):
    if degree in _basis_cache:
        return _basis_cache[degree]
    B = np.zeros((degree + 1, N))
    for k in range(degree + 1):
        B[k] = eval_chebyt(k, _x)
    _basis_cache[degree] = B
    return B

def _eval_value(h):
    """Compute the objective value (lower is better)."""
    h = np.asarray(h, dtype=np.float64)
    if np.any(~np.isfinite(h)):
        return np.inf
    sh = float(np.sum(h))
    if abs(sh) < 1e-12:
        return np.inf
    conv = np.convolve(h, h)
    max_conv = float(np.max(conv))
    val = abs(2.0 * N * max_conv / (sh * sh))
    return val

def _smooth_objective(x, beta=80.0, sign_hint=1.0):
    """Smooth surrogate using log-sum-exp for the max.
    sign_hint: if >0, we expect max_conv > 0; if <0, we expect max_conv < 0.
    This helps with the abs() kink."""
    h = x
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    conv = np.convolve(h, h)
    max_c = np.max(conv)
    shifted = conv - max_c
    # Numerically stable log-sum-exp
    log_sum_exp = max_c + np.log(np.sum(np.exp(beta * shifted))) / beta
    # Use sign_hint to guide which side of zero we're on
    # If sign_hint > 0, we want to minimize log_sum_exp (assuming it's positive)
    # If sign_hint < 0, we want to maximize log_sum_exp (assuming it's negative)
    if sign_hint > 0:
        val = 2.0 * N * log_sum_exp / (s * s)
    else:
        val = -2.0 * N * log_sum_exp / (s * s)
    if not np.isfinite(val):
        return 1e10
    return val

def _smooth_grad(x, beta=80.0, sign_hint=1.0):
    """Gradient of the smooth surrogate."""
    h = x
    s = np.sum(h)
    if abs(s) < 1e-10:
        return np.zeros_like(h)
    
    conv = np.convolve(h, h)
    max_c = np.max(conv)
    shifted = conv - max_c
    weights = np.exp(beta * shifted)
    weights /= np.sum(weights)
    
    # Vectorized gradient computation using FFT for speed
    # grad_lse[i] = 2 * sum_{j=0}^{N-1} weights[i+j] * h[j]
    # This is a cross-correlation that can be done via FFT
    # Pad to avoid circular convolution
    n_fft = 2 * N
    W_padded = np.zeros(n_fft)
    W_padded[:N] = weights
    h_padded = np.zeros(n_fft)
    h_padded[:N] = h
    
    # Cross-correlation: grad_lse[i] = sum_j weights[i+j] * h[j]
    # This is equivalent to correlation of weights with h
    # Using FFT: corr = ifft(fft(weights) * conj(fft(h)))
    W_fft = np.fft.rfft(W_padded)
    h_fft = np.fft.rfft(h_padded)
    corr = np.fft.irfft(W_fft * np.conj(h_fft), n=n_fft)
    grad_lse = 2.0 * corr[:N]
    
    lse = max_c + np.log(np.sum(weights)) / beta
    if sign_hint > 0:
        f = 2.0 * N * lse / (s * s)
    else:
        f = -2.0 * N * lse / (s * s)
    
    df_dh = 2.0 * N * grad_lse / (s * s) - 4.0 * N * lse / (s * s * s)
    
    if sign_hint > 0:
        grad = df_dh
    else:
        grad = -df_dh
    
    return grad

def _local_exact_refine(h0, max_iter=500, tol=1e-12):
    """Local refinement using exact objective with subgradient methods."""
    h = h0.copy()
    best_h = h.copy()
    best_val = _eval_value(h)
    
    for it in range(max_iter):
        if best_val < tol:
            break
        
        # Compute subgradient of exact objective
        s = np.sum(h)
        if abs(s) < 1e-10:
            break
        conv = np.convolve(h, h)
        max_idx = np.argmax(conv)
        max_conv = conv[max_idx]
        
        # Subgradient: find j such that max_idx = i + j for some i
        # conv[k] = sum_{j} h[j] * h[k-j] where 0 <= j < N, 0 <= k-j < N
        # So for k = max_idx, valid j are max(0, max_idx - (N-1)) to min(N-1, max_idx)
        j_min = max(0, max_idx - (N - 1))
        j_max = min(N - 1, max_idx)
        
        # Subgradient of max_conv w.r.t. h[i]:
        # d(max_conv)/dh[i] = sum over valid pairs where i is involved
        # For conv[max_idx] = sum_j h[j] * h[max_idx - j]
        # d/dh[i] = h[max_idx - i] if 0 <= max_idx - i < N, plus
        #           h[i] * (contribution from h[max_idx - i] term)
        # Actually: conv[k] = sum_{j=max(0,k-N+1)}^{min(N-1,k)} h[j] * h[k-j]
        # d(conv[k])/dh[i] = h[k-i] if 0<=k-i<N, plus h[i] if 0<=k-i<N (from other term)
        # Wait, let me be more careful:
        # conv[k] = sum_j h[j]*h[k-j]
        # d/dh[i] = h[k-i] (when j=i) + h[i] (when k-j=i, i.e., j=k-i)
        # So if 0 <= k-i < N: d/dh[i] = h[k-i] + (h[i] if i != k-i else h[i])
        # If i == k-i (i.e., 2i = k), then it's just h[i] (one term, not two)
        
        k = max_idx
        subgrad = np.zeros(N)
        for i in range(N):
            j = k - i
            if 0 <= j < N:
                if i == j:
                    subgrad[i] = h[i]
                else:
                    subgrad[i] = h[j]
        
        # Now the full objective: val = |2*N*max_conv / s^2|
        # Let f = 2*N*max_conv / s^2
        # If f > 0: val = f, dval/dh = df/dh
        # If f < 0: val = -f, dval/dh = -df/dh
        # df/dh = 2*N*(d(max_conv)/dh * s^2 - max_conv * 2*s*ds/dh) / s^4
        #       = 2*N*(subgrad * s^2 - max_conv * 2*s * ones) / s^4
        #       = 2*N*subgrad / s^2 - 4*N*max_conv / s^3
        
        f_val = 2.0 * N * max_conv / (s * s)
        df_dh = 2.0 * N * subgrad / (s * s) - 4.0 * N * max_conv / (s * s * s)
        
        if f_val >= 0:
            grad = df_dh
        else:
            grad = -df_dh
        
        # Normalize gradient
        gnorm = np.linalg.norm(grad)
        if gnorm < 1e-15:
            break
        
        # Line search
        alpha = 1.0
        h_new = h - alpha * grad / gnorm * np.max(np.abs(grad))
        # Simple backtracking line search
        for _ in range(20):
            v_new = _eval_value(h_new)
            if v_new < best_val:
                best_val = v_new
                best_h = h_new.copy()
                h = h_new
                break
            alpha *= 0.5
            h_new = h - alpha * grad / gnorm * np.max(np.abs(grad))
        else:
            break
    
    return best_h, best_val

def construct():
    t0 = time.time()
    best_h = np.ones(N)
    best_val = _eval_value(best_h)  # 2.0
    
    def elapsed():
        return time.time() - t0
    
    rng = np.random.RandomState(42)
    
    # --- Generate diverse initial points ---
    initial_points = []
    
    # Seed
    initial_points.append(np.ones(N).astype(float))
    
    # Random uniform
    for _ in range(15):
        initial_points.append(rng.uniform(-1, 1, N))
    
    # Bimodal: mostly zero with random spikes
    for _ in range(15):
        h = np.zeros(N)
        k = rng.randint(50, 350)
        idx = rng.choice(N, size=k, replace=False)
        h[idx] = rng.choice([-1, 1], size=k)
        initial_points.append(h)
    
    # Alternating signs
    h_alt = np.ones(N)
    h_alt[1::2] = -1
    initial_points.append(h_alt)
    
    # Low-frequency cosines
    t = np.linspace(0, 1, N, endpoint=False)
    for freq in [1, 2, 3, 4, 5, 7, 10, 15, 20]:
        initial_points.append(np.cos(2 * np.pi * freq * t))
        initial_points.append(np.sin(2 * np.pi * freq * t))
    
    # Parameterized bump profiles
    i_arr = np.arange(N)
    for center in [N//8, N//4, N//2, 3*N//4, 7*N//8]:
        for width in [20, 50, 100, 150]:
            h_bump = np.exp(-((i_arr - center)**2) / (2 * width**2))
            initial_points.append(h_bump)
            initial_points.append(-h_bump)
            # Double bump with opposite signs
            h_double = np.exp(-((i_arr - center)**2) / (2 * width**2)) - np.exp(-((i_arr - (N-1-center)**2) / (2 * width**2)))
            initial_points.append(h_double)
            # Bump with offset
            h_off = np.exp(-((i_arr - center)**2) / (2 * width**2)) + 0.5 * np.ones(N)
            initial_points.append(h_off)
    
    # Random Fourier synthesis
    for n_comp in [3, 5, 8]:
        for _ in range(5):
            freqs = rng.randint(1, 40, n_comp)
            phases = rng.uniform(0, 2 * np.pi, n_comp)
            amps = rng.uniform(0.5, 2.0, n_comp)
            h_fourier = np.zeros(N)
            for f, p, a in zip(freqs, phases, amps):
                h_fourier += a * np.cos(2 * np.pi * f * t + p)
            initial_points.append(h_fourier)
    
    # Structured patterns: blocks of +1 and -1
    for block_size in [20, 50, 100]:
        n_blocks = N // block_size
        for _ in range(3):
            h_blocks = np.zeros(N)
            for b in range(n_blocks):
                sign = rng.choice([-1, 1])
                h_blocks[b*block_size:(b+1)*block_size] = sign
            initial_points.append(h_blocks)
    
    # Smooth random via Gaussian filter
    for _ in range(10):
        h_smooth = rng.uniform(-1, 1, N)
        h_smooth = gaussian_filter1d(h_smooth, sigma=rng.uniform(5, 20))
        initial_points.append(h_smooth)
        
    # Chebyshev-based initializations with random coefficients
    for degree in [5, 10, 15, 20]:
        B = _get_basis(degree)
        for _ in range(5):
            coeffs = rng.uniform(-1, 1, degree + 1)
            h_cheb = B.T @ coeffs
            initial_points.append(h_cheb)
    
    # --- Evaluate and sort initial points ---
    init_values = []
    for init_h in initial_points:
        val_init = _eval_value(init_h)
        init_values.append((val_init, init_h))
    init_values.sort(key=lambda x: x[0])
    
    # Take top 40 initial points for optimization
    top_inits = [h for _, h in init_values[:40]]
    
    # --- Optimize from each initial point using L-BFGS-B with smooth surrogate ---
    for init_h in top_inits:
        if elapsed() > DEADLINE - 10:
            break
        
        # Determine sign hint from current max_conv
        s_init = np.sum(init_h)
        if abs(s_init) < 1e-10:
            init_h = init_h + 0.1
            s_init = np.sum(init_h)
        
        conv_init = np.convolve(init_h, init_h)
        max_conv_init = np.max(conv_init)
        sign_hint = 1.0 if max_conv_init >= 0 else -1.0
        
        # Try multiple betas for the smooth surrogate
        for beta in [50.0, 100.0, 200.0]:
            if elapsed() > DEADLINE - 5:
                break
            
            try:
                result = minimize(
                    lambda x, b=beta, sh=sign_hint: _smooth_objective(x, b, sh),
                    init_h,
                    jac=lambda x, b=beta, sh=sign_hint: _smooth_grad(x, b, sh),
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'maxfun': 600, 'gtol': 1e-8}
                )
                
                if elapsed() > DEADLINE - 3:
                    break
                
                h_candidate = result.x
                v = _eval_value(h_candidate)
                if v < best_val and np.all(np.isfinite(h_candidate)) and abs(np.sum(h_candidate)) > 1e-10:
                    best_val = v
                    best_h = h_candidate.copy()
            except Exception:
                pass
    
    # --- Local exact refinement from best solution ---
    if elapsed() < DEADLINE - 10:
        try:
            h_refined, val_refined = _local_exact_refine(best_h, max_iter=200, tol=1e-14)
            if val_refined < best_val:
                best_val = val_refined
                best_h = h_refined
        except Exception:
            pass
    
    # --- Final polish: high-precision smooth refinement from best solution ---
    if elapsed() < DEADLINE - 5:
        try:
            h0 = best_h.copy()
            s0 = np.sum(h0)
            conv0 = np.convolve(h0, h0)
            max_c0 = np.max(conv0)
            sign0 = 1.0 if max_c0 >= 0 else -1.0
            
            # Use L-BFGS-B on the smooth objective with high beta for final polish
            for beta_polish in [500.0, 1000.0, 2000.0]:
                if elapsed() > DEADLINE - 2:
                    break
                res = minimize(
                    lambda x, b=beta_polish, sh=sign0: _smooth_objective(x, b, sh),
                    h0,
                    jac=lambda x, b=beta_polish, sh=sign0: _smooth_grad(x, b, sh),
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'maxfun': 200, 'gtol': 1e-12}
                )
                h_candidate = res.x
                v = _eval_value(h_candidate)
                if v < best_val and np.all(np.isfinite(h_candidate)) and abs(np.sum(h_candidate)) > 1e-10:
                    best_val = v
                    best_h = h_candidate.copy()
                    h0 = h_candidate.copy()
                    # Update sign hint
                    conv_new = np.convolve(h0, h0)
                    max_c_new = np.max(conv_new)
                    sign0 = 1.0 if max_c_new >= 0 else -1.0
        except Exception:
            pass
    
    # Final safety: ensure finite and non-zero sum
    if not np.all(np.isfinite(best_h)) or abs(np.sum(best_h)) < 1e-12:
        best_h = np.ones(N)
    
    return best_h
