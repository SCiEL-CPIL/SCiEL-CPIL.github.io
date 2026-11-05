"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _value(h):
    s = np.sum(h)
    if abs(s) < 1e-15:
        return 1e18
    conv = np.convolve(h, h, mode='full')
    return abs(2.0 * N * np.max(conv) / (s * s))


def _smooth_max(x, tau):
    m = np.max(x)
    return m + tau * np.log(np.sum(np.exp((x - m) / tau)))


def _smooth_max_grad(x, tau):
    m = np.max(x)
    e = np.exp((x - m) / tau)
    p = e / np.sum(e)
    return 1.0 / tau * p


def _build_initial_poly():
    """Build initial polynomial window with alternating sign."""
    n = np.arange(N, dtype=float)
    # Alternating sign pattern: (-1)^k
    sign = np.ones(N)
    sign[1::2] = -1.0
    
    # Low-order polynomial window (smooth taper)
    # Use a raised-cosine-like window
    t = n / (N - 1)
    window = 0.5 * (1.0 + np.cos(np.pi * t))  # 0 to 1 to 0
    window = window ** 0.5  # Sharpen the taper
    
    return sign * window


def _try_polynomial_optimization(deadline):
    """Optimize polynomial window coefficients."""
    n = np.arange(N, dtype=float)
    t = n / (N - 1)
    
    # Basis functions: alternating sign times polynomial in t
    # h[k] = (-1)^k * sum_{j=0}^{d} c[j] * t^j
    d = 8  # degree
    
    # Precompute basis
    basis = np.zeros((d + 1, N))
    for j in range(d + 1):
        sign = np.ones(N)
        sign[1::2] = -1.0
        basis[j] = sign * (t ** j)
    
    # Initial coefficients
    c0 = np.zeros(d + 1)
    c0[0] = 1.0
    
    def obj_c(c):
        h = basis.T @ c
        s = np.sum(h)
        if abs(s) < 1e-12:
            return 1e18
        conv = np.convolve(h, h, mode='full')
        # Use smooth max for gradient
        tau = 0.01
        return _smooth_max(conv, tau)
    
    def grad_c(c):
        h = basis.T @ c
        s = np.sum(h)
        if abs(s) < 1e-12:
            return np.zeros(d + 1)
        conv = np.convolve(h, h, mode='full')
        tau = 0.01
        grad_conv = _smooth_max_grad(conv, tau)
        # Gradient of smooth_max(conv) w.r.t. c
        # conv[i] = sum_{j,k: j+k=i} h[j]*h[k]
        # d conv[i] / d c[a] = sum_{j,k: j+k=i} (basis[a,j]*h[k] + h[j]*basis[a,k])
        # = 2 * sum_{j,k: j+k=i} basis[a,j]*h[k] (by symmetry, but careful)
        # Actually: d/dc_a sum_{j+k=i} h_j h_k = sum_{j+k=i} (dh_j/dc_a * h_k + h_j * dh_k/dc_a)
        # dh_j/dc_a = basis[a,j]
        # = sum_{j+k=i} basis[a,j]*h_k + sum_{j+k=i} h_j*basis[a,k]
        # Let u=j, v=k: first term = sum_u basis[a,u]*h_{i-u} for u where i-u in [0,N-1]
        # = (basis[a] * h)_full[i] via cross-correlation
        # Actually this is convolve-like
        # Use FFT for efficiency
        # For now, use direct for correctness
        grad = np.zeros(d + 1)
        for a in range(d + 1):
            # d conv / d c_a
            # conv = convolve(h, h)
            # d conv[i] / d c_a = sum_{j+k=i} (basis[a,j]*h[k] + h[j]*basis[a,k])
            # = sum_{j+k=i} basis[a,j]*h[k] + sum_{j+k=i} h[j]*basis[a,k]
            # First term: cross-correlation of basis[a] and h at lag i
            # = (basis[a] * h)_full[i] where * denotes convolution? No.
            # sum_j basis[a,j] * h[k] with j+k=i => k=i-j
            # = sum_j basis[a,j] * h[i-j] for valid indices
            # This is the cross-correlation (not convolution)
            # In numpy: np.correlate(basis[a], h, mode='full')
            # Actually np.correlate(a, b) = sum_k a[k] * conj(b[k-n])
            # Let me just use convolve approach
            # h = basis.T @ c, so d h / d c_a = basis[a]
            # conv = np.convolve(h, h)
            # d conv / d c_a = np.convolve(basis[a], h) + np.convolve(h, basis[a])
            # But np.convolve(a, b) = np.convolve(b, a) for real
            # So d conv / d c_a = 2 * np.convolve(basis[a], h) ... wait
            # conv[i] = sum_{j+k=i} h[j] h[k]
            # d/dc_a: sum_{j+k=i} (basis[a,j] h[k] + h[j] basis[a,k])
            # = sum_j basis[a,j] h[i-j] + sum_k h[i-k] basis[a,k]  (with j+k=i)
            # Both terms are the same: sum_j basis[a,j] h[i-j]
            # So = 2 * sum_j basis[a,j] h[i-j]
            # This is 2 * np.correlate(h, basis[a], mode='full')? 
            # np.correlate(h, basis[a], 'full')[i] = sum_j h[j] basis[a][j - (i - (2N-2))]? 
            # Let me just compute directly for safety with N=400
            term = np.zeros(2 * N - 1)
            for j in range(N):
                for k in range(N):
                    if j + k < 2 * N - 1:
                        term[j + k] += basis[a, j] * h[k] + h[j] * basis[a, k]
            grad[a] = np.dot(grad_conv, term)
        return grad
    
    # Use L-BFGS-B
    best_val = 1e18
    best_c = c0.copy()
    
    start = time.time()
    try:
        res = minimize(obj_c, c0, method='L-BFGS-B', jac=grad_c,
                       options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-8})
        if res.fun < best_val:
            best_val = res.fun
            best_c = res.x.copy()
    except Exception:
        pass
    
    if time.time() > deadline - 15:
        return None, 1e18
    
    # Refine with decreasing tau
    for tau in [0.005, 0.002, 0.001]:
        def obj_c_t(c, tau=tau):
            h = basis.T @ c
            s = np.sum(h)
            if abs(s) < 1e-12:
                return 1e18
            conv = np.convolve(h, h, mode='full')
            return _smooth_max(conv, tau)
        
        def grad_c_t(c, tau=tau):
            h = basis.T @ c
            s = np.sum(h)
            if abs(s) < 1e-12:
                return np.zeros(d + 1)
            conv = np.convolve(h, h, mode='full')
            grad_conv = _smooth_max_grad(conv, tau)
            grad = np.zeros(d + 1)
            for a in range(d + 1):
                term = np.zeros(2 * N - 1)
                for j in range(N):
                    for k in range(N):
                        if j + k < 2 * N - 1:
                            term[j + k] += basis[a, j] * h[k] + h[j] * basis[a, k]
                grad[a] = np.dot(grad_conv, term)
            return grad
        
        try:
            res = minimize(obj_c_t, best_c, method='L-BFGS-B', jac=grad_c_t,
                           options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-8})
            h_new = basis.T @ res.x
            val_new = _value(h_new)
            if val_new < best_val:
                best_val = val_new
                best_c = res.x.copy()
        except Exception:
            pass
        
        if time.time() > deadline - 10:
            break
    
    h = basis.T @ best_c
    val = _value(h)
    return h, val


def _try_dct_optimization(deadline):
    """Optimize in DCT basis for smoothness."""
    # Use DCT basis to encourage smoothness
    from scipy.fft import dct
    
    d = 20  # number of DCT modes
    
    # Initial: use the polynomial result as starting point
    h_init = _build_initial_poly()
    
    # Project onto DCT basis
    # h[k] = sum_{m=0}^{d-1} a[m] * dct_basis[m, k]
    k = np.arange(N, dtype=float)
    dct_basis = np.zeros((d, N))
    for m in range(d):
        dct_basis[m] = np.cos(np.pi * m * (k + 0.5) / N)
    
    # Initial coefficients
    a0 = np.zeros(d)
    for m in range(d):
        a0[m] = np.dot(h_init, dct_basis[m]) / np.dot(dct_basis[m], dct_basis[m])
    
    def obj_a(a):
        h = dct_basis.T @ a
        s = np.sum(h)
        if abs(s) < 1e-12:
            return 1e18
        conv = np.convolve(h, h, mode='full')
        return _smooth_max(conv, 0.005)
    
    def grad_a(a):
        h = dct_basis.T @ a
        s = np.sum(h)
        if abs(s) < 1e-12:
            return np.zeros(d)
        conv = np.convolve(h, h, mode='full')
        grad_conv = _smooth_max_grad(conv, 0.005)
        grad = np.zeros(d)
        for m in range(d):
            # d conv / d a_m = 2 * sum_j dct_basis[m,j] * h[i-j] (similar logic)
            term = np.zeros(2 * N - 1)
            for j in range(N):
                for kk in range(N):
                    term[j + kk] += dct_basis[m, j] * h[kk] + h[j] * dct_basis[m, kk]
            grad[m] = np.dot(grad_conv, term)
        return grad
    
    best_val = 1e18
    best_a = a0.copy()
    
    try:
        res = minimize(obj_a, a0, method='L-BFGS-B', jac=grad_a,
                       options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-8})
        if res.fun < best_val:
            best_val = res.fun
            best_a = res.x.copy()
    except Exception:
        pass
    
    h = dct_basis.T @ best_a
    val = _value(h)
    return h, val


def _try_local_polish(h0, deadline):
    """Local optimization on raw h with constraints."""
    best_h = h0.copy()
    best_val = _value(h0)
    
    def obj_h(h):
        s = np.sum(h)
        if abs(s) < 1e-12:
            return 1e18
        conv = np.convolve(h, h, mode='full')
        return _smooth_max(conv, 0.002)
    
    def grad_h(h):
        s = np.sum(h)
        if abs(s) < 1e-12:
            return np.zeros(N)
        conv = np.convolve(h, h, mode='full')
        grad_conv = _smooth_max_grad(conv, 0.002)
        # d conv[i] / d h[j] = sum_{k: j+k=i} h[k] + h[j] (from second term)
        # = h[i-j] + h[i-j] for symmetric... 
        # Actually: conv[i] = sum_{j+k=i} h[j] h[k]
        # d/dh[j] = sum_k where j+k=i: h[k] + sum_k where j+k=i: h[k] (since h[j]h[k] derivative w.r.t h[j] is h[k], and h[k]h[j] derivative is h[k])
        # = 2 * h[i-j] if i-j is valid index, else 0... 
        # Wait: conv[i] = sum_{j=0}^{min(i,N-1)} h[j] h[i-j] (with h[i-j]=0 if i-j out of range)
        # d conv[i] / d h[j] = h[i-j] + (h[j] * 1 if i-j==j else 0)... no
        # d/dh[j] [sum_{k} h[k] h[i-k]] where k runs over valid indices
        # = h[i-j] (from the term where k=j, derivative of h[j] is 1) 
        #   + sum over terms where i-k = j, i.e., k = i-j: h[k] * 1 = h[i-j]
        # But if j = i-j, i.e., 2j=i, then we only count it once? No:
        # conv[i] = sum_{k=0}^{N-1} h[k] h[i-k] where h[i-k]=0 if i-k not in [0,N-1]
        # d/dh[j] = h[i-j] (from k=j term) + h[i-j] (from k=i-j term) = 2*h[i-j]
        # Unless j = i-j, in which case it's h[j]^2 and derivative is 2*h[j], same formula.
        # So d conv[i] / d h[j] = 2 * h[i-j] if 0 <= i-j <= N-1, else 0
        
        grad = np.zeros(N)
        for j in range(N):
            for i in range(2 * N - 1):
                k = i - j
                if 0 <= k < N:
                    grad[j] += grad_conv[i] * 2 * h[k]
        return grad
    
    try:
        res = minimize(obj_h, best_h, method='L-BFGS-B', jac=grad_h,
                       options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-6})
        h_new = res.x
        val_new = _value(h_new)
        if val_new < best_val:
            best_val = val_new
            best_h = h_new.copy()
    except Exception:
        pass
    
    return best_h, best_val


def construct():
    start_time = time.time()
    DEADLINE = 145.0  # Safety margin before 150s
    
    best_h = np.ones(N)
    best_val = _value(best_h)  # 2.0
    
    # Strategy 1: Polynomial window optimization
    if time.time() < start_time + DEADLINE - 40:
        try:
            h_poly, val_poly = _try_polynomial_optimization(start_time + DEADLINE)
            if h_poly is not None and val_poly < best_val:
                best_val = val_poly
                best_h = h_poly.copy()
        except Exception:
            pass
    
    # Strategy 2: DCT basis optimization
    if time.time() < start_time + DEADLINE - 25:
        try:
            h_dct, val_dct = _try_dct_optimization(start_time + DEADLINE)
            if val_dct < best_val:
                best_val = val_dct
                best_h = h_dct.copy()
        except Exception:
            pass
    
    # Strategy 3: Local polish
    if time.time() < start_time + DEADLINE - 10:
        try:
            h_polished, val_polished = _try_local_polish(best_h, start_time + DEADLINE)
            if val_polished < best_val:
                best_val = val_polished
                best_h = h_polished.copy()
        except Exception:
            pass
    
    # Final safety check
    if time.time() > start_time + DEADLINE - 3:
        return np.ones(N)
    
    # Validate result
    if not np.all(np.isfinite(best_h)) or abs(np.sum(best_h)) < 1e-10:
        return np.ones(N)
    
    return best_h
# EVOLVE-BLOCK-END
