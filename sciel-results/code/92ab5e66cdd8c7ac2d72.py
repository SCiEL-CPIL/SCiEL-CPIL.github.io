import numpy as np
import time
from scipy.optimize import minimize

N = 400
_EPS = 1e-6  # scale-anchoring penalty strength (negligible effect on value)
_DEADLINE = 130.0  # safety margin before 150s kill


def _convolve_fast(a, b):
    """Full convolution using FFT for speed."""
    n = len(a) + len(b) - 1
    size = 1 << (n - 1).bit_length()  # next power of 2
    fa = np.fft.rfft(a, size)
    fb = np.fft.rfft(b, size)
    return np.fft.irfft(fa * fb, size)[:n]


def _soft_max(c, beta):
    """Smooth approximation to max(c) using log-sum-exp."""
    if beta < 1e-6:
        return float(np.max(c))
    m = float(np.max(c))
    # Shift for numerical stability
    shifted = (c - m) * beta
    # Clip to avoid overflow
    shifted = np.clip(shifted, -50, 50)
    return m + np.log(np.sum(np.exp(shifted))) / beta


def _soft_max_grad(c, beta):
    """Gradient of soft-max with respect to c."""
    if beta < 1e-6:
        idx = np.argmax(c)
        g = np.zeros_like(c)
        g[idx] = 1.0
        return g
    m = float(np.max(c))
    shifted = (c - m) * beta
    shifted = np.clip(shifted, -50, 50)
    weights = np.exp(shifted) / np.sum(np.exp(shifted))
    return weights


def _value_and_grad(h, beta=0.0):
    """Return (value + penalty, gradient). value is scale-invariant."""
    S = float(np.sum(h))
    if abs(S) < 1e-12:
        return np.inf, np.zeros_like(h)
    c = _convolve_fast(h, h)
    
    if beta > 1e-6:
        m = _soft_max(c, beta)
        wc = _soft_max_grad(c, beta)
    else:
        m = float(np.max(c))
        jstar = int(np.argmax(c))
        wc = np.zeros_like(c)
        wc[jstar] = 1.0
    
    val = 2.0 * N * m / (S * S)

    # dm/dh[i] = sum_k wc[k] * d c_k / d h[i]
    # c_k = sum_j h_j h_{k-j}, so d c_k / d h[i] = h_{k-i} + h_{i-k} (if both in range)
    # But for autocorrelation c_k = sum_j h_j h_{k-j}, d c_k/d h_i = h_{k-i} (when valid)
    # Actually d c_k / d h_i = h_{k-i} if 0 <= k-i < N, plus h_{i-k} if 0 <= i-k < N
    # Wait: c_k = sum_j h_j * h_{k-j}
    # d c_k / d h_i: term j=i gives h_{k-i}; term k-j=i gives j=k-i, so h_{k-(k-i)}=h_i
    # So d c_k/d h_i = h_{k-i} (from j=i) + h_i (from j=k-i) if k-i != i
    # If k-i == i, i.e., k=2i, then it's 2*h_i
    # More carefully: d/dh_i [sum_j h_j h_{k-j}] = h_{k-i} + h_{k-i} if we're counting both terms
    # No: sum_j h_j h_{k-j}. Derivative wrt h_i:
    #   Term where j=i: h_{k-i}
    #   Term where k-j=i, i.e., j=k-i: h_{k-(k-i)} = h_i
    #   But if i = k-i, i.e., k=2i, these are the same term, so derivative is 2*h_i
    # So d c_k/d h_i = h_{k-i} + (h_i if k-i != i else h_i) 
    # Actually let me just compute: d c_k/d h_i = h_{k-i} + h_{k-i} when the two indices coincide?
    # Let's be precise. c_k = sum_{j=0}^{N-1} h_j h_{k-j} where h_m = 0 if m not in [0,N-1]
    # d c_k / d h_i = sum_j [delta_{j,i} h_{k-j} + h_j delta_{k-j,i}]
    # = h_{k-i} + h_{k-i} if k-i is valid and i != k-i
    # = 2 h_{k-i} if k-i == i (i.e., k = 2i)
    # = h_{k-i} if only one term valid
    # Simplify: d c_k/d h_i = h_{k-i} * (1 + [k-i == i and 0<=k-i<N])
    # But actually both terms give h_{k-i} when valid, so:
    # d c_k/d h_i = h_{k-i} if 0<=k-i<N, plus h_{k-i} if 0<=k-i<N (same condition)
    # Wait no. Let me redo.
    # c_k = sum_j h_j h_{k-j}
    # d/dh_i: for each j, d/dh_i (h_j h_{k-j}) = delta(j,i) h_{k-j} + h_j delta(k-j,i)
    # = delta(j,i) h_{k-j} + h_j delta(j,k-i)
    # Sum over j: h_{k-i} + h_{k-i} if k-i is in [0,N-1] and i is in [0,N-1]
    # But these are the same value h_{k-i} added twice only if both conditions hold
    # Actually: first term sum_j delta(j,i) h_{k-j} = h_{k-i} (if 0<=k-i<N)
    # Second term: sum_j h_j delta(j,k-i) = h_{k-i} (if 0<=k-i<N)
    # So d c_k/d h_i = 2 h_{k-i} if 0<=k-i<N, else 0
    # Wait, that's not right either. Let me check with small example.
    # h = [a, b, c], c_0 = a*a + b*b + c*c (no, c_0 = sum_j h_j h_{0-j} = h_0 h_0 = a^2)
    # c_1 = h_0 h_1 + h_1 h_0 = 2ab
    # c_2 = h_0 h_2 + h_2 h_0 = 2ac
    # c_3 = h_1 h_2 + h_2 h_1 = 2bc
    # c_4 = h_2 h_2 = c^2
    # d c_1/d a = 2b = 2*h_{1-0} = 2*h_1 ✓
    # d c_2/d a = 2c = 2*h_{2-0} = 2*h_2 ✓
    # d c_0/d a = 2a = 2*h_0 ✓
    # So yes, d c_k/d h_i = 2 h_{k-i} if 0 <= k-i < N, else 0
    
    dm_dh = np.zeros(N)
    for k in range(len(c)):
        if abs(wc[k]) > 1e-15:
            idx = k - np.arange(N)
            valid = (idx >= 0) & (idx < N)
            dm_dh[valid] += wc[k] * 2.0 * h[idx[valid]]

    grad_val = 2.0 * N * (dm_dh / (S * S) - 2.0 * m / (S ** 3))

    # Tiny penalty to fix the (flat) scale direction -> well-conditioned BFGS
    penalty = 0.5 * _EPS * (S - 1.0) ** 2
    grad_pen = _EPS * (S - 1.0) * np.ones(N)
    return val + penalty, grad_val + grad_pen


def _sinusoidal_value_and_grad(params):
    """
    2-parameter sinusoidal parameterization:
        h[i] = a + b * cos(2*pi*k*i/N + phase)
    
    params = [a, b, phase, k] where k is treated as continuous for gradient.
    We optimize over (a, b, phase) with k fixed per call, or use a grid over k.
    """
    a, b, phase, k = params
    i = np.arange(N)
    h = a + b * np.cos(2.0 * np.pi * k * i / N + phase)
    
    S = float(np.sum(h))
    if abs(S) < 1e-12:
        return 1e10, np.zeros(4)
    
    c = np.convolve(h, h)
    m = float(np.max(c))
    val = 2.0 * N * m / (S * S)
    
    # Compute gradient via finite differences for simplicity (2-param is cheap)
    eps = 1e-7
    grad = np.zeros(4)
    for j in range(3):  # a, b, phase
        p_plus = params.copy()
        p_plus[j] += eps
        v_plus = _sinusoidal_value_no_grad(p_plus)
        p_minus = params.copy()
        p_minus[j] -= eps
        v_minus = _sinusoidal_value_no_grad(p_minus)
        grad[j] = (v_plus - v_minus) / (2 * eps)
    
    return val, grad


def _sinusoidal_value_no_grad(params):
    """Evaluate the value for sinusoidal params without gradient."""
    a, b, phase, k = params
    i = np.arange(N)
    h = a + b * np.cos(2.0 * np.pi * k * i / N + phase)
    S = float(np.sum(h))
    if abs(S) < 1e-12:
        return 1e10
    c = np.convolve(h, h)
    m = float(np.max(c))
    return 2.0 * N * m / (S * S)


def _iterative_reweight(h, max_iter=50, tol=1e-10):
    """
    Iterative reweighting: alternate between minimizing max autocorrelation
    and adjusting weights to equalize the peak.
    """
    h = h.copy().astype(float)
    S = float(np.sum(h))
    if abs(S) < 1e-12:
        return h
    
    for it in range(max_iter):
        c = _convolve_fast(h, h)
        m = float(np.max(c))
        if m <= 0:
            break
        
        # Find the lag(s) achieving max
        jstar = int(np.argmax(c))
        
        # The gradient direction: we want to reduce c[jstar]
        # d c[jstar]/d h[i] = 2 h[jstar - i] for valid i
        idx = jstar - np.arange(N)
        valid = (idx >= 0) & (idx < N)
        
        # Simple gradient step to reduce the peak
        step = 0.1 * np.zeros(N)
        step[valid] = h[idx[valid]]
        
        # Normalize step to have unit norm
        snorm = np.linalg.norm(step)
        if snorm < 1e-15:
            break
        step = step / snorm
        
        # Line search
        best_alpha = 0.01
        best_val_step = 2.0 * N * m / (S * S)
        
        for alpha in [0.01, 0.05, 0.1, 0.2, 0.5]:
            h_trial = h - alpha * step
            S_trial = float(np.sum(h_trial))
            if abs(S_trial) < 1e-12:
                continue
            c_trial = _convolve_fast(h_trial, h_trial)
            m_trial = float(np.max(c_trial))
            val_trial = 2.0 * N * m_trial / (S_trial * S_trial)
            if val_trial < best_val_step:
                best_val_step = val_trial
                best_alpha = alpha
        
        h = h - best_alpha * step
        
        # Check convergence
        c_new = _convolve_fast(h, h)
        m_new = float(np.max(c_new))
        S_new = float(np.sum(h))
        val_new = 2.0 * N * m_new / (S_new * S_new)
        if abs(val_new - best_val_step) < tol:
            break
    
    return h


def construct():
    t0 = time.time()
    deadline = _DEADLINE - 5.0  # safety margin

    best_h = np.ones(N, dtype=float)
    best_val, _ = _value_and_grad(best_h)  # seed value = 2.0

    rng = np.random.default_rng(20240601)

    # Phase 1: Grid search over frequency k, then optimize (a, b, phase)
    # For each k, the problem reduces to 3 parameters
    # Use a subset of k values to save time, focusing on lower frequencies
    k_values = list(range(1, 20)) + list(range(40, 60, 2)) + list(range(100, 150, 5))
    
    for k in k_values:
        if (time.time() - t0) > (deadline - 25.0):
            break
        
        # Try multiple phase initializations
        for phase_init in [0.0, np.pi / 4, np.pi / 2, 3.0 * np.pi / 4, np.pi]:
            if (time.time() - t0) > (deadline - 25.0):
                break
            
            # Try multiple (a, b) initializations
            for a_init in [0.0, 1.0, -1.0, 0.5, -0.5]:
                for b_init in [0.5, 1.0, 2.0, 0.1, -0.5, -1.0]:
                    if (time.time() - t0) > (deadline - 25.0):
                        break
                    
                    params0 = np.array([a_init, b_init, phase_init, float(k)])
                    
                    def obj_2d(x):
                        v, _ = _sinusoidal_value_and_grad(x)
                        return v
                    
                    def grad_2d(x):
                        _, g = _sinusoidal_value_and_grad(x)
                        return g
                    
                    try:
                        res = minimize(obj_2d, params0, jac=grad_2d,
                                       method="L-BFGS-B",
                                       bounds=[(-10, 10), (-10, 10), (-np.pi, np.pi), (float(k), float(k))],
                                       options={"maxiter": 80, "maxfun": 200,
                                                "ftol": 1e-12, "gtol": 1e-10})
                        
                        if np.isfinite(res.fun) and res.fun < best_val:
                            a, b, phase, _ = res.x
                            i = np.arange(N)
                            h_candidate = a + b * np.cos(2.0 * np.pi * k * i / N + phase)
                            S = float(np.sum(h_candidate))
                            if abs(S) > 1e-9:
                                best_val = res.fun
                                best_h = h_candidate.copy()
                    except Exception:
                        pass
    
    # Phase 2: Full L-BFGS-B optimization from sinusoidal solutions and other starts
    def objective(h):
        v, _ = _value_and_grad(h, beta=0.0)
        return v

    def grad(h):
        _, gg = _value_and_grad(h, beta=0.0)
        return gg

    # Start from best sinusoidal solution
    start_points = [best_h.copy()]
    
    # Add some diverse starting points
    start_points.append(np.ones(N))
    start_points.append(rng.standard_normal(N))
    start_points.append(np.cumsum(rng.standard_normal(N)) - np.mean(rng.standard_normal(N)))
    
    for h0 in start_points:
        if (time.time() - t0) > (deadline - 15.0):
            break
        
        if abs(float(np.sum(h0))) < 1e-6:
            h0 = h0 + 1.0
        
        try:
            res = minimize(objective, h0, jac=grad, method="L-BFGS-B",
                           options={"maxiter": 150, "maxfun": 400,
                                    "ftol": 1e-14, "gtol": 1e-12})
            if np.isfinite(res.fun) and res.fun < best_val:
                best_val = res.fun
                best_h = res.x.copy()
        except Exception:
            pass
    
    # Phase 3: Iterative reweighting refinement
    if (time.time() - t0) < (deadline - 10.0):
        try:
            h_rw = _iterative_reweight(best_h, max_iter=80, tol=1e-12)
            S_rw = float(np.sum(h_rw))
            if abs(S_rw) > 1e-9:
                val_rw = 2.0 * N * float(np.max(_convolve_fast(h_rw, h_rw))) / (S_rw * S_rw)
                if val_rw < best_val:
                    best_val = val_rw
                    best_h = h_rw.copy()
        except Exception:
            pass
    
    # Phase 4: Local refinement with tighter tolerances
    if (time.time() - t0) < (deadline - 5.0):
        try:
            res = minimize(objective, best_h, jac=grad, method="L-BFGS-B",
                           options={"maxiter": 200, "maxfun": 600,
                                    "ftol": 1e-15, "gtol": 1e-14})
            if np.isfinite(res.fun) and res.fun < best_val:
                best_val = res.fun
                best_h = res.x.copy()
        except Exception:
            pass
    
    # Phase 5: Final iterative reweighting polish
    if (time.time() - t0) < (deadline - 3.0):
        try:
            h_rw = _iterative_reweight(best_h, max_iter=50, tol=1e-14)
            S_rw = float(np.sum(h_rw))
            if abs(S_rw) > 1e-9:
                val_rw = 2.0 * N * float(np.max(_convolve_fast(h_rw, h_rw))) / (S_rw * S_rw)
                if val_rw < best_val:
                    best_val = val_rw
                    best_h = h_rw.copy()
        except Exception:
            pass

    # Rescale to sum=1 (value is scale-invariant, so this is safe) for validity
    S = float(np.sum(best_h))
    if abs(S) > 1e-9:
        best_h = best_h / S
    else:
        best_h = best_h + 1.0
    return best_h
