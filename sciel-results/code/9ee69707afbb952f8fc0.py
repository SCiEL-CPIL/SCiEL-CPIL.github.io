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

def _ratio_from_g(g, beta=50.0):
    """Compute ratio given g where h = g^2 (element-wise), normalized."""
    h = g ** 2
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0, np.zeros_like(g)
    h_norm = h / l1
    l2_sq = np.sum(h_norm ** 2)
    conv = _autoconv(h_norm)
    sm, grad_sm = _soft_max(conv, beta)
    if sm < 1e-15:
        return 0.0, np.zeros_like(g)
    ratio = l2_sq / sm
    
    # Gradient via chain rule
    # ratio = l2_sq / sm
    # dl2_sq/dh_norm = 2*h_norm / l1 (since l2_sq = sum(h_norm^2), h_norm = h/l1)
    # Actually: h_norm = h/l1, l2_sq = sum(h_norm^2)
    # d(l2_sq)/d(h_i) = 2*h_norm_i/l1 - l2_sq/l1 (from quotient rule on normalization)
    # Let me be more careful:
    # l2_sq = sum_j (h_j/l1)^2 = (1/l1^2) * sum_j h_j^2
    # d(l2_sq)/d(h_i) = (2*h_i/l1^2) - (1/l1^3)*2*sum_j h_j^2 = 2*h_i/l1^2 - 2*l2_sq/l1
    # Wait: d/dh_i [sum_j h_j^2 / l1^2] = 2*h_i/l1^2 - sum_j h_j^2 * 2/l1^3 = 2*h_i/l1^2 - 2*l2_sq/l1
    
    # d(sm)/d(h_i) = sum_k grad_sm[k] * d(conv_k)/d(h_i)
    # conv_k = sum_j h_norm[j]*h_norm[k-j] (with appropriate indexing)
    # d(conv_k)/d(h_i) = h_norm[k-i] + h_norm[i-k] (if in range)
    
    # d(ratio)/d(h_i) = (d(l2_sq)/d(h_i) * sm - l2_sq * d(sm)/d(h_i)) / sm^2
    
    # Then d(ratio)/d(g_i) = d(ratio)/d(h_i) * 2*g_i
    
    # Compute d(sm)/d(h_i)
    # conv has length 2N-1, conv[k] = sum_j h_norm[j]*h_norm[k-j] for valid j,k-j
    # For h_norm normalized: conv[k] = sum_j (h_j/l1)*(h_{k-j}/l1) for 0<=j<N, 0<=k-j<N
    # d(conv_k)/d(h_i) = d/dh_i [sum_j h_j*h_{k-j}/l1^2]
    # = (h_{k-i} + h_{i-k})/l1^2 - 2*conv_k/l1 (if i is in valid range for k)
    
    # This is getting complex. Let me use a simpler approach:
    # Compute gradient numerically but efficiently with a few points
    # Actually, let me just use the formula directly.
    
    # d(conv_k)/d(h_i): conv_k = sum_j h_norm[j]*h_norm[k-j]
    # = sum_j h[j]*h[k-j] / l1^2
    # d/dh_i = [delta(j,i)*h[k-i] + delta(k-j,i)*h[j]] / l1^2 - sum_j h[j]*h[k-j] * 2*delta(j,i)/l1^3
    # Wait, l1 depends on h_i too: l1 = sum_j h_j, dl1/dh_i = 1
    # d(conv_k)/d(h_i) = [h[k-i]*I(0<=k-i<N) + h[i-k]*I(0<=i-k<N)] / l1^2 - 2*conv_k * I(0<=i<N) / l1^3
    # Wait no: d/dh_i [S/l1^2] where S = sum_j h[j]*h[k-j]
    # = (dS/dh_i * l1^2 - S * 2*l1 * dl1/dh_i) / l1^4
    # = (dS/dh_i * l1 - 2*S/l1) / l1^2
    # dS/dh_i = h[k-i] if 0<=k-i<N else 0, plus h[i-k] if 0<=i-k<N else 0
    # But wait, S = sum_j h[j]*h[k-j], so dS/dh_i = h[k-i] (from j=i term) + h[i-k] (from k-j=i, i.e., j=k-i term)
    # These are the same term when k-i = i-k, i.e., k=2i. Otherwise they're different.
    # Actually for j=i: term is h[i]*h[k-i]. For j=k-i: term is h[k-i]*h[i]. Same!
    # So dS/dh_i = 2*h[k-i] if 0<=k-i<N, else 0. (The two contributions are identical)
    
    # Hmm wait, that's only if i != k-i. If i = k-i (k=2i), then it's h[i]^2 and d/dh_i = 2*h[i].
    # In general: dS/dh_i = h[k-i] + h[i-k] where we interpret h[m]=0 if m<0 or m>=N.
    # But h[k-i] and h[i-k] are different indices unless k=2i.
    # For S = sum_j h[j]*h[k-j]:
    # dS/dh_i = h[k-i] (when j=i) + h[i-k] (when k-j=i, i.e., j=k-i)
    # But if j=i and j=k-i are the same (i=k-i, k=2i), then we get 2*h[i].
    # Otherwise: h[k-i] + h[i-k] but these are h at different indices.
    # Wait: when j=i, the term in S is h[i]*h[k-i]. d/dh_i = h[k-i].
    # When j=k-i, the term is h[k-i]*h[i]. d/dh_i = h[k-i].
    # So if i != k-i: dS/dh_i = h[k-i] + h[k-i] = 2*h[k-i] (provided both j=i and j=k-i are in [0,N))
    # If i == k-i (k=2i): dS/dh_i = 2*h[i]
    # In all cases: dS/dh_i = 2*h[k-i] if 0<=k-i<N, else 0.
    # Wait that's not right either. Let me reconsider.
    
    # S = sum_{j=0}^{N-1} h[j]*h[k-j] (where h[m]=0 for m outside [0,N-1])
    # dS/dh_i = sum_j [delta(j,i)*h[k-j] + h[j]*delta(k-j,i)]
    # = h[k-i] + h[i-k]  (where h[m]=0 if m not in [0,N-1])
    # = h[k-i] + h[i-k]
    
    # For non-negative indices: h[k-i] is valid if 0<=k-i<N, h[i-k] is valid if 0<=i-k<N.
    # Note k-i and i-k have opposite signs (unless k=i). So at most one is in range.
    # If k>i: k-i>0, i-k<0, so only h[k-i] contributes (if k-i<N)
    # If k<i: i-k>0, k-i<0, so only h[i-k] contributes (if i-k<N)
    # If k=i: both are h[0], so 2*h[0]... wait no, h[k-i]=h[0] and h[i-k]=h[0], so 2*h[0]?
    # No wait: when j=i=k, the term is h[k]*h[0]. d/dh_k = h[0].
    # When j=k-i=0, the term is h[0]*h[k]. d/dh_k = h[0].
    # So yes, 2*h[0] when k=i.
    
    # OK this is getting too complicated for inline. Let me just use finite differences 
    # on the g vector but with a coarser step and fewer evaluations.
    
    # Actually, let me just use scipy minimize with numerical gradient on g.
    # The key advantage is that g can be any real vector (no non-negativity constraint on h).
    return ratio, None

def _objective(g, beta=50.0):
    """Objective to minimize: negative ratio."""
    h = g ** 2
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 1e6
    h_norm = h / l1
    l2_sq = np.sum(h_norm ** 2)
    conv = _autoconv(h_norm)
    sm, _ = _soft_max(conv, beta)
    if sm < 1e-15:
        return 1e6
    return -l2_sq / sm

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

def construct():
    start_time = time.time()
    best_heights = [1.0] * N
    best_score = 0.0
    
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
    
    # Uniform
    initializations.append(np.ones(N).copy())
    
    # Random profiles
    rng = np.random.RandomState(42)
    for _ in range(10):
        h = rng.rand(N) ** rng.choice([1, 2, 3])
        if np.sum(h) > 0:
            initializations.append(h.copy())
    
    beta = 50.0
    
    # Optimize from each initialization
    for i, h_init in enumerate(initializations):
        if _time_left(start_time) < 15:
            break
        
        # Normalize h
        l1 = np.sum(h_init)
        if l1 == 0:
            continue
        h_init = h_init / l1
        
        # Convert to g: g = sqrt(h) (positive branch)
        g_init = np.sqrt(h_init)
        
        # Also try negative signs for some components
        for sign_pattern in [0, 1]:
            if _time_left(start_time) < 10:
                break
            
            g_try = g_init.copy()
            if sign_pattern == 1:
                # Flip signs of some random components
                flip_mask = rng.rand(N) < 0.3
                g_try[flip_mask] *= -1
            
            # Use L-BFGS-B
            try:
                result = minimize(
                    _objective, g_try, args=(beta,),
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
                )
            except Exception:
                continue
            
            g_opt = result.x
            h_opt = g_opt ** 2
            l1_opt = np.sum(h_opt)
            if l1_opt < 1e-15:
                continue
            h_opt = h_opt / l1_opt
            score_opt = -_objective(g_opt, beta)
            
            if score_opt > best_score:
                best_score = score_opt
                best_heights = h_opt.tolist()
        
        # Also try local perturbation of best found so far
        if best_score > 0 and _time_left(start_time) > 20:
            g_best = np.sqrt(np.array(best_heights))
            for _ in range(3):
                if _time_left(start_time) < 10:
                    break
                g_pert = g_best + 0.05 * rng.randn(N)
                try:
                    result = minimize(
                        _objective, g_pert, args=(beta,),
                        method='L-BFGS-B',
                        options={'maxiter': 80, 'ftol': 1e-10, 'gtol': 1e-8}
                    )
                except Exception:
                    continue
                g_opt = result.x
                h_opt = g_opt ** 2
                l1_opt = np.sum(h_opt)
                if l1_opt < 1e-15:
                    continue
                h_opt = h_opt / l1_opt
                score_opt = -_objective(g_opt, beta)
                if score_opt > best_score:
                    best_score = score_opt
                    best_heights = h_opt.tolist()
    
    # Final safety: ensure we return valid non-negative heights
    if best_score <= 0:
        best_heights = [1.0] * N
    
    return best_heights
