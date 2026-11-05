"""Low-dimensional basis optimization for third autocorrelation inequality."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.special import eval_chebyt

N = 400
DEADLINE = 145.0  # Safety margin before 150s kill

# Precompute Chebyshev basis up to degree D
D = 40
_x = np.linspace(-1, 1, N)
_basis = np.zeros((D + 1, N))
for k in range(D + 1):
    _basis[k] = eval_chebyt(k, _x)

# Precompute FFT size for cross-correlation
_nfft = 1
while _nfft < 2 * N - 1:
    _nfft *= 2
_FFT_SIZE = _nfft

def _value(h):
    """Compute the raw objective value (lower is better)."""
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

def _h_from_coeffs(c):
    """Reconstruct h from Chebyshev coefficients."""
    return _basis.T @ c

def _autocorr_fft(h):
    """Compute full autocorrelation via FFT."""
    h_padded = np.zeros(_FFT_SIZE)
    h_padded[:N] = h
    H = np.fft.rfft(h_padded)
    conv = np.fft.irfft(H * np.conj(H), n=_FFT_SIZE)[:2*N-1]
    return conv

def _smooth_objective(c, beta=50.0):
    """Smooth surrogate: replace max with soft-max via log-sum-exp."""
    h = _h_from_coeffs(c)
    if np.any(~np.isfinite(h)):
        return 1e10
    sh = float(np.sum(h))
    if abs(sh) < 1e-12:
        return 1e10
    conv = _autocorr_fft(h)
    conv_max = np.max(conv)
    shifted = conv - conv_max
    logsumexp = conv_max + np.log(np.sum(np.exp(beta * shifted))) / beta
    val = abs(2.0 * N * logsumexp / (sh * sh))
    return val

def _smooth_grad(c, beta=50.0):
    """Gradient of smooth objective using vectorized cross-correlation."""
    h = _h_from_coeffs(c)
    if np.any(~np.isfinite(h)):
        return np.zeros_like(c)
    sh = float(np.sum(h))
    if abs(sh) < 1e-12:
        return np.zeros_like(c)
    
    conv = _autocorr_fft(h)
    conv_max = np.max(conv)
    shifted = conv - conv_max
    exp_shifted = np.exp(beta * shifted)
    sum_exp = np.sum(exp_shifted)
    logsumexp = conv_max + np.log(sum_exp) / beta
    w = exp_shifted / sum_exp  # shape (2N-1,)
    
    # Vectorized gradient computation
    j_idx = np.arange(N)[:, None]  # (N, 1)
    m_idx = np.arange(2*N-1)[None, :]  # (1, 2N-1)
    
    idx1 = m_idx - j_idx  # (N, 2N-1)
    mask1 = (idx1 >= 0) & (idx1 < N)
    idx1_safe = np.clip(idx1, 0, N-1)
    h_idx1 = h[idx1_safe]
    term1 = np.where(mask1, w[None, :] * h_idx1, 0.0)
    
    idx2 = j_idx - m_idx  # (N, 2N-1)
    mask2 = (idx2 >= 0) & (idx2 < N)
    idx2_safe = np.clip(idx2, 0, N-1)
    h_idx2 = h[idx2_safe]
    term2 = np.where(mask2, w[None, :] * h_idx2, 0.0)
    
    grad_h = np.sum(term1 + term2, axis=1)  # (N,)
    
    inner = 2.0 * N * logsumexp / (sh * sh)
    sign = 1.0 if inner >= 0 else -1.0
    dval_dh = sign * (2.0 * N * grad_h / (sh * sh) - 4.0 * N * logsumexp / (sh**3))
    
    grad_c = _basis @ dval_dh
    return grad_c

def _local_refine(c_start, maxiter=200, ftol=1e-12, gtol=1e-10, beta=50.0):
    """Local refinement using L-BFGS-B with smooth objective."""
    def obj(c):
        return _smooth_objective(c, beta=beta)
    
    def grad(c):
        return _smooth_grad(c, beta=beta)
    
    res = minimize(
        obj, c_start, jac=grad,
        method='L-BFGS-B',
        options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol}
    )
    return res.x

def construct():
    t0 = time.time()
    
    def elapsed():
        return time.time() - t0
    
    # Initial best: seed program
    best_h = np.ones(N)
    best_val = _value(best_h)  # 2.0
    
    rng = np.random.RandomState(123)
    
    # Generate diverse starting points in coefficient space
    starts = []
    
    # 1. Structured starts: single frequencies
    for k in [1, 2, 3, 5, 8, 13, 20]:
        c_struct = np.zeros(D + 1)
        c_struct[k] = 1.0
        starts.append(c_struct)
        c_struct_neg = np.zeros(D + 1)
        c_struct_neg[k] = -1.0
        starts.append(c_struct_neg)
    
    # 2. Linear ramp
    c_linear = np.zeros(D + 1)
    c_linear[1] = 1.0
    starts.append(c_linear)
    
    # 3. Quadratic
    c_quad = np.zeros(D + 1)
    c_quad[2] = 1.0
    starts.append(c_quad)
    
    # 4. Random starts - reduced count to save time
    for _ in range(15):
        c_rand = rng.uniform(-3, 3, D + 1)
        starts.append(c_rand)
    
    # 5. Constant
    c_const = np.zeros(D + 1)
    c_const[0] = 1.0
    starts.append(c_const)
    
    # Refine each start with progressive beta
    for c_start in starts:
        if elapsed() > DEADLINE - 10:
            break
        
        try:
            if not np.all(np.isfinite(c_start)):
                continue
            
            # Phase 1: coarse optimization with lower beta
            c_opt1 = _local_refine(c_start, maxiter=80, ftol=1e-10, gtol=1e-8, beta=10.0)
            
            # Phase 2: fine optimization with higher beta
            if elapsed() > DEADLINE - 5:
                break
            c_opt2 = _local_refine(c_opt1, maxiter=150, ftol=1e-12, gtol=1e-10, beta=50.0)
            
            # Phase 3: very fine polish
            if elapsed() > DEADLINE - 3:
                break
            c_opt3 = _local_refine(c_opt2, maxiter=200, ftol=1e-14, gtol=1e-12, beta=100.0)
            
            h_candidate = _h_from_coeffs(c_opt3)
            v_candidate = _value(h_candidate)
            
            if v_candidate < best_val and np.all(np.isfinite(h_candidate)) and abs(np.sum(h_candidate)) > 1e-10:
                best_val = v_candidate
                best_h = h_candidate.copy()
                
        except Exception:
            pass
    
    # Final sanity check
    if not np.all(np.isfinite(best_h)) or abs(np.sum(best_h)) < 1e-12:
        best_h = np.ones(N)
        best_val = 2.0
    
    return best_h
