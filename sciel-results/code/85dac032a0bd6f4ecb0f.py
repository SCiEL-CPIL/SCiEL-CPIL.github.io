import time
import numpy as np
from scipy.optimize import minimize

N = 50
_START = time.time()
DEADLINE = 100.0  # safety margin before 150s kill

dx_step = 0.5 / N  # step width
step_centers = -0.25 + (np.arange(N) + 0.5) * dx_step

def _autocorr_max_exact(h):
    """Compute max of autoconvolution of step function h exactly.
    h: array of N nonneg heights on uniform steps of width dx_step over [-1/4, 1/4].
    The autoconvolution is piecewise linear; max occurs at t = k*dx_step for integer k.
    """
    # For step functions, (h*h)(t) at t = k*dx_step can be computed as:
    # (h*h)(k*dx_step) = dx_step * sum_i h[i] * h[i-k] (discrete autocorrelation)
    # Wait, let me verify this formula.
    # Actually for continuous step functions, the autoconvolution at t=k*w is:
    # (h*h)(k*w) = sum_{i,j} h[i]*h[j] * max(0, w - |k*w - (x_i_center - x_j_center)|) / w ... 
    # This is getting complex. Let me use the fact that for step functions,
    # the autoconvolution at multiples of w equals w times the discrete autocorrelation.
    
    # Discrete autocorrelation: R[k] = sum_i h[i] * h[i+k] for k = -(N-1)..(N-1)
    # The continuous autoconvolution max is w * max_k R[k]
    # Let me verify: if h is all ones, R[0] = N, so (h*h)(0) = w*N = 0.5. 
    # Actual: (h*h)(0) = ∫_{-0.25}^{0.25} 1*1 ds = 0.5. ✓
    # (h*h)(w) = ∫ h(s)h(s-w) ds. h(s)=1 on [-0.25,0.25], h(s-w)=1 on [-0.25+w, 0.25+w].
    # Overlap: [-0.25+w, 0.25] has length 0.5-w. So (h*h)(w) = 0.5-w.
    # w*R[1] = w*(N-1) = w*49 = 0.49. But 0.5-w = 0.5-0.01=0.49. ✓
    
    # So the formula is: (h*h)(k*w) = w * sum_i h[i]*h[i+k] for |k| < N, else 0.
    # But wait, this only holds when the step boundaries align. Let me check k=1 more carefully.
    # h[i] is on [x_i, x_{i+1}] where x_i = -0.25 + i*w.
    # (h*h)(w) = ∫ h(s)h(s-w) ds
    # h(s) = h[i] for s in [x_i, x_{i+1}]
    # h(s-w) = h[j] for s-w in [x_j, x_{j+1}], i.e., s in [x_j+w, x_{j+1}+w] = [x_{j+1}, x_{j+2}]
    # So h(s)h(s-w) = h[i]*h[j] when s in [x_i, x_{i+1}] ∩ [x_{j+1}, x_{j+2}]
    # The overlap is nonempty when i and j+1 are related. Specifically, [x_i, x_{i+1}] ∩ [x_{j+1}, x_{j+2}]
    # is nonempty iff i = j+1 or i = j+2 (adjacent or same).
    # If i = j+1: overlap is [x_{j+1}, x_{j+1}] = point, measure 0.
    # If i = j+2: overlap is [x_{j+2}, x_{j+2}] = point, measure 0.
    # Hmm, that gives 0, but we know (h*h)(w) = 0.5-w for uniform h.
    
    # I think I'm making an error. Let me reconsider.
    # h(s-w) is the function s -> h(s-w). If h is defined on [-0.25, 0.25], then h(s-w) is defined
    # when s-w in [-0.25, 0.25], i.e., s in [-0.25+w, 0.25+w].
    # For s in [x_i, x_{i+1}], h(s) = h[i].
    # h(s-w) = h[s-w] where s-w is in some interval. If s-w in [x_j, x_{j+1}], then h(s-w) = h[j].
    # So the integral is sum over i,j of h[i]*h[j] * length([x_i, x_{i+1}] ∩ [x_j+w, x_{j+1}+w])
    # [x_j+w, x_{j+1}+w] = [x_{j+1}, x_{j+2}]
    # So we need length([x_i, x_{i+1}] ∩ [x_{j+1}, x_{j+2}])
    # This is w if i = j+1, 0 otherwise (since intervals are adjacent or disjoint).
    # Wait, [x_i, x_{i+1}] and [x_{j+1}, x_{j+2}] overlap iff i = j+1 (then overlap is [x_{j+1}, x_{j+1}], measure 0) 
    # or if i+1 = j+1, i.e., i=j, then [x_i, x_{i+1}] ∩ [x_{i+1}, x_{i+2}] = {x_{i+1}}, measure 0.
    
    # I'm clearly making an error. Let me just compute it numerically for a small case.
    # N=2, w=0.25, x=[-0.25, 0, 0.25], h=[a,b]
    # (h*h)(0) = ∫_{-0.25}^{0.25} h(s)h(-s) ds = ∫_{-0.25}^{0} a*a ds + ∫_{0}^{0.25} b*b ds = a²*0.25 + b²*0.25
    # (h*h)(0.25) = ∫ h(s)h(s-0.25) ds. s-0.25 in [-0.5, 0]. h(s-0.25) nonzero when s-0.25 in [-0.25,0.25], i.e., s in [0, 0.5].
    # But s is in [-0.25, 0.25], so s in [0, 0.25]. On [0,0.25], h(s)=b, h(s-0.25): s-0.25 in [-0.25, 0], so h(s-0.25)=a.
    # So (h*h)(0.25) = ∫_0^{0.25} b*a ds = ab*0.25
    # (h*h)(-0.25) = similarly ab*0.25
    # (h*h)(0.5) = ∫ h(s)h(s-0.5) ds. s-0.5 in [-0.75, -0.25]. h(s-0.5) nonzero when s-0.5 in [-0.25,0.25], i.e., s in [0.25, 0.75]. 
    # s in [-0.25,0.25], so s in {0.25}, measure 0. So (h*h)(0.5)=0.
    
    # So for N=2: (h*h)(0) = 0.25(a²+b²), (h*h)(±0.25) = 0.25ab
    # Discrete autocorr: R[0]=a²+b², R[1]=ab, R[-1]=ab
    # w*R[0] = 0.25(a²+b²) ✓, w*R[1] = 0.25ab ✓
    
    # So the formula (h*h)(k*w) = w * R[k] where R[k] = sum_i h[i]*h[i+k] IS correct!
    
    h = np.asarray(h, dtype=float)
    # Compute discrete autocorrelation
    # R[k] = sum_i h[i]*h[i+k] for k = -(N-1)..(N-1)
    # Use FFT for efficiency
    n = 1
    while n < 2*N:
        n *= 2
    g = np.zeros(n)
    g[:N] = h
    F = np.fft.rfft(g)
    ac = np.fft.irfft(F * F.conj(), n=n)
    # ac[k] = sum_i g[i]*g[i+k] for k=0..n-1 (circular, but zero-padded so linear for k<N)
    # ac[N-k] = sum_i g[i]*g[i-k] = R[-k]
    R_pos = ac[:N]  # R[0], R[1], ..., R[N-1]
    R_neg = ac[1:n-N+1][::-1]  # R[-(N-1)], ..., R[-1]
    R = np.concatenate([R_neg, R_pos])  # R[-(N-1)], ..., R[-1], R[0], R[1], ..., R[N-1]
    max_R = float(np.max(R))
    if max_R <= 0:
        return 0.0
    return dx_step * max_R

def _f_from_h(h):
    h = np.asarray(h, dtype=float)
    h = np.maximum(h, 0.0)
    int_f2 = dx_step * float(np.sum(h*h))
    int_f = dx_step * float(np.sum(h))
    if int_f <= 0:
        return 0.0
    acmax = _autocorr_max_exact(h)
    if acmax <= 0:
        return 0.0
    return int_f2 / (int_f * acmax)

def _objective(x):
    x = np.maximum(x, 0.0)
    mx = np.max(x)
    if mx <= 0:
        return 0.0
    h = x / mx  # peak = 1
    return -_f_from_h(h)

def _param_shapes():
    shapes = []
    # Gaussian
    for sig in [0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.2]:
        shapes.append(np.exp(-step_centers**2 / (2*sig*sig)))
    # Lorentzian
    for sig in [0.02, 0.05, 0.1]:
        shapes.append(1.0/(1.0 + (step_centers/sig)**2))
    # Triangle
    for w in [0.02, 0.05, 0.1, 0.2]:
        shapes.append(np.clip(1.0 - np.abs(step_centers)/w, 0, None))
    # Plateau
    for w in [0.02, 0.05, 0.1]:
        shapes.append((np.abs(step_centers) < w).astype(float))
    # Single step (center)
    s = np.zeros(N); s[N//2] = 1.0; shapes.append(s)
    # Two steps
    s = np.zeros(N); s[N//2-2]=1; s[N//2+2]=1; shapes.append(s)
    # Concentrated shapes
    for width in [1, 2, 3, 5]:
        s = np.zeros(N)
        center = N // 2
        s[center-width//2:center-width//2+width] = 1.0
        shapes.append(s)
    return shapes

def construct():
    best_h = np.ones(N)
    best_f = _f_from_h(best_h)
    t0 = time.time()
    starts = _param_shapes()
    # add random initializations
    rng = np.random.default_rng(0)
    for _ in range(8):
        starts.append(rng.random(N))
    bounds = [(0.0, 10.0)] * N
    for s in starts:
        if time.time() - t0 > DEADLINE:
            break
        try:
            res = minimize(_objective, s, method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10})
            x = np.maximum(res.x, 0.0)
            mx = np.max(x)
            if mx <= 0: continue
            h = x / mx
            f = _f_from_h(h)
            if f > best_f:
                best_f = f
                best_h = h
        except Exception:
            pass
    # ensure nonneg
    best_h = np.maximum(best_h, 0.0)
    return [float(v) for v in best_h]
