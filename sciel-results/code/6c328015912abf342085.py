import numpy as np
import time
from scipy.optimize import minimize

N = 50
DEADLINE = 140.0  # Safety margin before 150s kill

def _elapsed(start):
    return time.time() - start

def _time_left(start):
    return DEADLINE - _elapsed(start)

def _autoconv(h):
    """Compute discrete autoconvolution of h (length N) -> length 2N-1."""
    return np.convolve(h, h, mode='full')

def _soft_max(conv, beta=50.0):
    """Compute soft-max of conv and its gradient."""
    max_val = np.max(conv)
    shifted = beta * (conv - max_val)
    shifted = np.clip(shifted, -50, 50)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    sm = max_val + np.log(sum_exp) / beta
    # gradient of soft_max w.r.t. conv
    grad_sm = exp_shifted / sum_exp
    return sm, grad_sm

def _ratio_and_grad(h, beta=50.0):
    """
    Compute ratio = ||h||_2^2 / (||h||_1 * max(conv)) and its gradient w.r.t. h.
    h is nonnegative, not necessarily normalized.
    """
    h = np.maximum(h, 0.0)
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0, np.zeros_like(h)
    h_norm = h / l1
    l2_sq = np.sum(h_norm ** 2)
    conv = _autoconv(h_norm)
    sm, grad_sm = _soft_max(conv, beta)
    if sm < 1e-15:
        return 0.0, np.zeros_like(h)
    ratio = l2_sq / sm

    # Compute gradient analytically.
    # We want d(ratio)/d(h_i) where ratio = l2_sq / sm.
    # l2_sq = sum_j h_norm[j]^2 = sum_j (h_j/l1)^2
    # sm = soft_max(conv) where conv_k = sum_j h_norm[j]*h_norm[k-j]
    
    # d(l2_sq)/d(h_i):
    # l2_sq = (1/l1^2) * sum_j h_j^2
    # d(l2_sq)/d(h_i) = 2*h_i/l1^2 - 2*l2_sq/l1  (since d(1/l1^2)/d(h_i) = -2/l1^3)
    # = 2*h_i/l1^2 - 2*(sum_j h_j^2/l1^2)/l1 = 2*h_i/l1^2 - 2*l2_sq/l1
    # Wait: d/dh_i [sum_j h_j^2 / l1^2] = 2*h_i/l1^2 - sum_j h_j^2 * 2/l1^3
    # = 2*h_i/l1^2 - 2*(l2_sq * l1^2)/l1^3 = 2*h_i/l1^2 - 2*l2_sq/l1
    
    # d(sm)/d(h_i) = sum_k grad_sm[k] * d(conv_k)/d(h_i)
    # conv_k = sum_j h_norm[j] * h_norm[k-j] = (1/l1^2) * sum_j h_j * h_{k-j}
    # Let S_k = sum_j h_j * h_{k-j} (with h[m]=0 if m not in [0,N))
    # conv_k = S_k / l1^2
    # d(conv_k)/d(h_i) = [dS_k/dh_i * l1^2 - S_k * 2*l1 * 1] / l1^4
    # = [dS_k/dh_i * l1 - 2*S_k] / l1^3
    # dS_k/dh_i: S_k = sum_j h_j * h_{k-j}
    # dS_k/dh_i = h_{k-i} (from j=i) + h_{i-k} (from k-j=i, i.e., j=k-i)
    # where h[m] = 0 if m < 0 or m >= N.
    
    # Build dS/dh vector efficiently:
    # dS_k/dh_i = h[k-i] + h[i-k] for valid indices.
    # This is essentially: for each i, dS_k/dh_i is nonzero only for k in [i, i+N-1] (for h[k-i]) 
    # and k in [i-N+1, i] (for h[i-k]).
    # More precisely:
    #   h[k-i] contributes when 0 <= k-i < N, i.e., i <= k < i+N
    #   h[i-k] contributes when 0 <= i-k < N, i.e., i-N < k <= i
    # So dS_k/dh_i is nonzero for k in [max(0, i-N+1), min(2N-2, i+N-1)]
    
    # Let's compute d(sm)/d(h_i) = sum_k grad_sm[k] * d(conv_k)/d(h_i)
    # = sum_k grad_sm[k] * [dS_k/dh_i * l1 - 2*S_k] / l1^3
    # = (l1 / l1^3) * sum_k grad_sm[k] * dS_k/dh_i - (2/l1^3) * sum_k grad_sm[k] * S_k
    # = (1/l1^2) * sum_k grad_sm[k] * dS_k/dh_i - (2/l1^3) * (grad_sm @ conv * l1^2)
    # Wait: S_k = conv_k * l1^2, so sum_k grad_sm[k] * S_k = l1^2 * sum_k grad_sm[k] * conv_k
    # Let G = sum_k grad_sm[k] * conv_k = grad_sm @ conv
    
    # So d(sm)/d(h_i) = (1/l1^2) * sum_k grad_sm[k] * dS_k/dh_i - (2/l1^3) * l1^2 * G
    # = (1/l1^2) * sum_k grad_sm[k] * dS_k/dh_i - (2*G/l1)
    
    # Now compute T_i = sum_k grad_sm[k] * dS_k/dh_i
    # dS_k/dh_i = h[k-i] (if 0<=k-i<N) + h[i-k] (if 0<=i-k<N)
    # T_i = sum_k grad_sm[k] * h[k-i] * I(0<=k-i<N) + sum_k grad_sm[k] * h[i-k] * I(0<=i-k<N)
    # Let m = k-i: first term = sum_m grad_sm[m+i] * h[m] * I(0<=m<N)
    # Let m = i-k: second term = sum_m grad_sm[i-m] * h[m] * I(0<=m<N)
    # T_i = sum_{m=0}^{N-1} h[m] * (grad_sm[m+i] + grad_sm[i-m])
    # where grad_sm is extended by zero outside [0, 2N-2].
    
    # This is a cross-correlation! T = corr(h, grad_sm) appropriately.
    # Let's compute it via convolution.
    # T_i = sum_m h[m] * grad_sm[m+i] + sum_m h[m] * grad_sm[i-m]
    # First part: (h * grad_sm_extended)[i] where grad_sm_extended has length 3N-1 or so
    # Actually, let's just compute it directly using np.correlate or manual.
    
    # Extend grad_sm to handle indexing
    # grad_sm has length 2N-1, indices 0..2N-2
    # For T_i, we need grad_sm[m+i] for m in [0,N-1], so m+i in [i, i+N-1]
    # and grad_sm[i-m] for m in [0,N-1], so i-m in [i-N+1, i]
    
    # Compute T_i = sum_{m=0}^{N-1} h[m] * (grad_sm[m+i] if 0<=m+i<2N-1 else 0)
    #                   + sum_{m=0}^{N-1} h[m] * (grad_sm[i-m] if 0<=i-m<2N-1 else 0)
    
    # This is: T = np.correlate(h, grad_sm, mode='full') appropriately sliced
    # Let's verify: np.correlate(h, g, 'full')[k] = sum_j h[j] * g[j + k - (N-1)]... 
    # Actually np.correlate(a, v, 'full')[k] = sum_j a[j] * conj(v[j + k - (len(v)-1)])
    # This is getting confusing. Let me just compute T directly.
    
    # T_i = sum_{m=0}^{N-1} h[m] * grad_sm[m+i] + sum_{m=0}^{N-1} h[m] * grad_sm[i-m]
    # First term: for fixed i, sum over m of h[m]*grad_sm[m+i]. This is like a cross-correlation.
    # Let's use FFT for efficiency.
    
    # Actually for N=50, direct computation is fine.
    T = np.zeros(N)
    for i in range(N):
        # First part: sum_m h[m] * grad_sm[m+i] for m in [0, N-1], m+i in [0, 2N-2]
        m_idx = np.arange(N)
        g_idx1 = m_idx + i
        mask1 = (g_idx1 >= 0) & (g_idx1 < 2*N-1)
        if np.any(mask1):
            T[i] += np.sum(h[m_idx[mask1]] * grad_sm[g_idx1[mask1]])
        # Second part: sum_m h[m] * grad_sm[i-m] for m in [0, N-1], i-m in [0, 2N-2]
        g_idx2 = i - m_idx
        mask2 = (g_idx2 >= 0) & (g_idx2 < 2*N-1)
        if np.any(mask2):
            T[i] += np.sum(h[m_idx[mask2]] * grad_sm[g_idx2[mask2]])
    
    G = np.dot(grad_sm, conv)
    
    d_sm_dh = T / (l1**2) - 2.0 * G / l1
    
    d_l2sq_dh = 2.0 * h / (l1**2) - 2.0 * l2_sq / l1
    
    d_ratio_dh = (d_l2sq_dh * sm - l2_sq * d_sm_dh) / (sm**2)
    
    return ratio, d_ratio_dh

def _objective_neg(h, beta=50.0):
    """Objective to minimize: negative ratio. h is nonnegative."""
    ratio, _ = _ratio_and_grad(h, beta)
    return -ratio

def _objective_neg_grad(h, beta=50.0):
    """Objective and gradient for L-BFGS-B."""
    ratio, grad = _ratio_and_grad(h, beta)
    return -ratio, -grad

def _gaussian_profile(sigma, center=0.0):
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.25/N
    h = np.exp(-((x - center) ** 2) / (2 * sigma ** 2))
    return h

def _lorentzian_profile(gamma, center=0.0):
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.25/N
    h = 1.0 / (1.0 + ((x - center) / gamma) ** 2)
    return h

def _sinc_like_profile(width, center=0.0):
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.25/N
    t = (x - center) / width
    h = np.sinc(t) ** 2
    return np.maximum(h, 0)

def _two_gaussian(sigma1, center1, sigma2, center2):
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.25/N
    h1 = np.exp(-((x - center1) ** 2) / (2 * sigma1 ** 2))
    h2 = np.exp(-((x - center2) ** 2) / (2 * sigma2 ** 2))
    return h1 + h2

def _trapezoid_profile(w1, w2, center=0.0):
    """Trapezoid profile."""
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.25/N
    t = np.abs(x - center)
    h = np.zeros(N)
    mask1 = t <= w1
    mask2 = (t > w1) & (t <= w2)
    h[mask1] = 1.0
    h[mask2] = 1.0 - (t[mask2] - w1) / (w2 - w1)
    return np.maximum(h, 0)

def _plateau_profile(w, center=0.0):
    """Plateau with sharp edges."""
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.25/N
    h = np.where(np.abs(x - center) <= w, 1.0, 0.0)
    return h

def _cosine_bump(w, center=0.0):
    """Cosine bump: h = 0.5*(1+cos(pi*t/w)) for |t|<w, else 0."""
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.25/N
    t = np.abs(x - center)
    h = np.zeros(N)
    mask = t < w
    h[mask] = 0.5 * (1.0 + np.cos(np.pi * t[mask] / w))
    return h

def construct():
    start_time = time.time()
    best_heights = [1.0] * N
    best_score = 0.0
    beta = 50.0
    rng = np.random.RandomState(42)
    
    # Generate diverse initializations as h profiles
    initializations = []
    
    # Gaussian profiles
    for sigma in [0.02, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2]:
        for center in [-0.15, -0.1, -0.05, 0.0, 0.05, 0.1, 0.15]:
            h = _gaussian_profile(sigma, center)
            if np.sum(h) > 0:
                initializations.append(h.copy())
    
    # Lorentzian profiles
    for gamma in [0.02, 0.03, 0.05, 0.08, 0.1]:
        for center in [-0.1, -0.05, 0.0, 0.05, 0.1]:
            h = _lorentzian_profile(gamma, center)
            if np.sum(h) > 0:
                initializations.append(h.copy())
    
    # Sinc-like profiles
    for width in [0.03, 0.05, 0.08, 0.1, 0.15]:
        for center in [-0.05, 0.0, 0.05]:
            h = _sinc_like_profile(width, center)
            if np.sum(h) > 0:
                initializations.append(h.copy())
    
    # Two-Gaussian profiles
    for params in [
        (0.03, 0.0, 0.03, 0.1),
        (0.05, -0.05, 0.05, 0.05),
        (0.08, 0.0, 0.05, 0.15),
        (0.03, -0.1, 0.03, 0.1),
        (0.05, -0.15, 0.05, 0.15),
        (0.02, 0.0, 0.1, 0.0),
    ]:
        h = _two_gaussian(*params)
        if np.sum(h) > 0:
            initializations.append(h.copy())
    
    # Trapezoid profiles
    for w1, w2 in [(0.02, 0.1), (0.05, 0.15), (0.03, 0.08), (0.01, 0.05)]:
        for center in [-0.05, 0.0, 0.05]:
            h = _trapezoid_profile(w1, w2, center)
            if np.sum(h) > 0:
                initializations.append(h.copy())
    
    # Plateau profiles
    for w in [0.02, 0.04, 0.06, 0.1, 0.15]:
        for center in [-0.1, -0.05, 0.0, 0.05, 0.1]:
            h = _plateau_profile(w, center)
            if np.sum(h) > 0:
                initializations.append(h.copy())
    
    # Cosine bump profiles
    for w in [0.03, 0.05, 0.08, 0.12]:
        for center in [-0.05, 0.0, 0.05]:
            h = _cosine_bump(w, center)
            if np.sum(h) > 0:
                initializations.append(h.copy())
    
    # Uniform
    initializations.append(np.ones(N).copy())
    
    # Random profiles
    for _ in range(15):
        h = rng.rand(N) ** rng.choice([1, 2, 3, 5])
        if np.sum(h) > 0:
            initializations.append(h.copy())
    
    # Optimize from each initialization using h-space with analytical gradient
    for i, h_init in enumerate(initializations):
        if _time_left(start_time) < 20:
            break
        
        # Normalize h
        l1 = np.sum(h_init)
        if l1 == 0:
            continue
        h_init = h_init / l1
        
        # Use L-BFGS-B in h-space with bounds [0, inf]
        try:
            result = minimize(
                _objective_neg_grad, h_init, args=(beta,),
                method='L-BFGS-B',
                jac=True,
                bounds=[(0, None)] * N,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-9, 'maxfun': 500}
            )
        except Exception:
            continue
        
        h_opt = np.maximum(result.x, 0.0)
        l1_opt = np.sum(h_opt)
        if l1_opt < 1e-15:
            continue
        h_opt = h_opt / l1_opt
        score_opt, _ = _ratio_and_grad(h_opt, beta)
        
        if score_opt > best_score:
            best_score = score_opt
            best_heights = h_opt.tolist()
    
    # Local refinement: perturb best and re-optimize
    if best_score > 0 and _time_left(start_time) > 25:
        h_best = np.array(best_heights)
        for _ in range(8):
            if _time_left(start_time) < 15:
                break
            # Add random perturbation
            scale = rng.choice([0.01, 0.05, 0.1, 0.2])
            h_pert = h_best + scale * rng.randn(N)
            h_pert = np.maximum(h_pert, 0.0)
            l1_pert = np.sum(h_pert)
            if l1_pert < 1e-15:
                continue
            h_pert = h_pert / l1_pert
            
            try:
                result = minimize(
                    _objective_neg_grad, h_pert, args=(beta,),
                    method='L-BFGS-B',
                    jac=True,
                    bounds=[(0, None)] * N,
                    options={'maxiter': 150, 'ftol': 1e-12, 'gtol': 1e-9, 'maxfun': 400}
                )
            except Exception:
                continue
            
            h_opt = np.maximum(result.x, 0.0)
            l1_opt = np.sum(h_opt)
            if l1_opt < 1e-15:
                continue
            h_opt = h_opt / l1_opt
            score_opt, _ = _ratio_and_grad(h_opt, beta)
            
            if score_opt > best_score:
                best_score = score_opt
                best_heights = h_opt.tolist()
    
    # Final safety: ensure we return valid non-negative heights
    if best_score <= 0:
        best_heights = [1.0] * N
    
    return best_heights
