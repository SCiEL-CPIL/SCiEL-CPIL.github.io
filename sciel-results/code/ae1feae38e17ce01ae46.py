import numpy as np
import time
from scipy.optimize import minimize
from scipy.signal.windows import dpss

N = 600
W = 1.0 / 1200.0

# Fine grid for accurate autoconvolution evaluation
FINE_N = 3000
X_FINE = np.linspace(-0.25, 0.25, FINE_N, endpoint=False)
DX = 0.5 / FINE_N
# Map fine grid points to step function intervals
# Interval i covers [ -0.25 + i*W, -0.25 + (i+1)*W )
CELL = np.clip(((X_FINE + 0.25) / W).astype(np.int64), 0, N - 1)
N_FFT = 1 << (2 * FINE_N - 1).bit_length()

def ratio_from_h(h):
    """
    Compute the normalized maximum of the autoconvolution.
    """
    h = np.asarray(h, dtype=np.float64)
    if h.size != N:
        h = np.resize(h, N)
    h = np.clip(h, 0.0, None)
    s = h.sum()
    if s < 1e-15:
        return float('inf')
    
    # Map step heights to fine grid
    f_fine = h[CELL]
    
    # FFT-based autoconvolution
    F = np.fft.rfft(f_fine, N_FFT)
    conv = np.fft.irfft(F * F, N_FFT)[:2 * FINE_N - 1]
    conv *= DX  # scale by dx for continuous integral
    
    max_conv = np.max(conv)
    
    # Normalization: (integral f)^2 = (sum(h) * W)^2
    integral_sq = (s * W) ** 2
    if integral_sq < 1e-15:
        return float('inf')
    
    return max_conv / integral_sq

def softplus(x):
    return np.log1p(np.exp(np.clip(x, -50, 50)))

def coeffs_to_h(coeffs, dpss_basis):
    """
    Convert DPSS coefficients to step heights using softplus activation.
    """
    z = coeffs @ dpss_basis
    return softplus(z)

def objective(coeffs, dpss_basis, tau=0.01):
    """
    Smooth surrogate objective using LSE for the max.
    """
    h = coeffs_to_h(coeffs, dpss_basis)
    s = h.sum()
    if s < 1e-15:
        return 1e10
    
    f_fine = h[CELL]
    F = np.fft.rfft(f_fine, N_FFT)
    conv = np.fft.irfft(F * F, N_FFT)[:2 * FINE_N - 1]
    conv *= DX
    
    # Log-sum-exp for smooth max
    m = np.max(conv)
    if m < 1e-15:
        return 1e10
    lse = m + tau * np.log(np.sum(np.exp((conv - m) / tau)))
    
    integral_sq = (s * W) ** 2
    if integral_sq < 1e-15:
        return 1e10
    
    return lse / integral_sq

def construct():
    """
    Construct a nonnegative step function with 600 equal intervals on [-1/4, 1/4]
    to minimize the normalized maximum of its autoconvolution.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill

    # Initial best: uniform
    best_h = np.ones(N, dtype=np.float64)
    best_val = ratio_from_h(best_h)

    np.random.seed(42)

    # Generate DPSS basis
    # Time-bandwidth product NW. Higher NW means more concentrated in frequency.
    # We want a balance. Let's try a few NW values.
    K = 8 # Number of DPSS modes to use
    NW = 10.0
    
    # Generate DPSS basis matrix: shape (K, N)
    # dpss(M, NW, n) returns the n-th DPSS sequence of length M
    dpss_basis = np.zeros((K, N))
    for k in range(K):
        dpss_basis[k] = dpss(N, NW, k)

    # Normalize basis vectors to have unit L2 norm? 
    # The dpss function returns orthonormal sequences (sum of squares = 1).
    # This is good.

    # Generate diverse initializations
    inits = []
    
    # 1. Uniform-ish (all coeffs similar)
    for _ in range(5):
        p0 = np.ones(K) * 0.5
        p0 += np.random.normal(0, 0.1, K)
        inits.append(p0)

    # 2. Dominant first mode (most concentrated)
    for _ in range(5):
        p0 = np.zeros(K)
        p0[0] = 1.0
        p0 += np.random.normal(0, 0.1, K)
        inits.append(p0)

    # 3. Random
    for _ in range(10):
        p0 = np.random.normal(0, 0.5, K)
        inits.append(p0)

    # 4. Alternating signs to create structure
    for _ in range(5):
        p0 = np.zeros(K)
        for i in range(K):
            p0[i] = np.random.uniform(-0.5, 0.5) * (1 if i % 2 == 0 else -1)
        inits.append(p0)

    # Optimize from each initialization
    for idx, p0 in enumerate(inits):
        if time.time() > deadline - 20.0:
            break
            
        try:
            res = minimize(
                objective,
                p0,
                args=(dpss_basis, 0.01),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
            )
            
            if np.isfinite(res.fun):
                h_candidate = coeffs_to_h(res.x, dpss_basis)
                val = ratio_from_h(h_candidate)
                if val < best_val and np.isfinite(val):
                    best_val = val
                    best_h = h_candidate.copy()
        except Exception:
            pass

    # Direct coordinate descent polish on the 600 bin heights
    # This helps fine-tune the discrete step function after the parametric search
    if time.time() < deadline - 5.0:
        # Gradient-free local search: random perturbations
        # We can also try simple hill climbing on a subset of bins
        for it in range(1000):
            if time.time() > deadline - 2.0:
                break
            
            # Perturb a random subset of bins
            n_perturb = np.random.randint(5, 15)
            idx = np.random.choice(N, n_perturb, replace=False)
            h_try = best_h.copy()
            # Small random steps
            h_try[idx] += np.random.randn(n_perturb) * 0.01
            h_try = np.clip(h_try, 0, None)
            
            val = ratio_from_h(h_try)
            if val < best_val:
                best_val = val
                best_h = h_try.copy()
            else:
                # If not improved, try a smaller step or skip
                pass

    return list(best_h)
