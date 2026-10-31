"""
Optimized program combining spectral basis parameterization with local search.

Strategy:
- Parameterize heights as a non-negative linear combination of a small set of
  orthogonal basis functions (DCT-like modes). This encourages spectrally flat
  autoconvolution, avoiding the high-peak local minima of sparse Sidon sets.
- Use scipy.optimize L-BFGS-B on the coefficient space (with log-parameterization
  to maintain non-negativity) to refine the coefficients.
- Combine with a local search that adjusts individual heights for fine-tuning.
- Deadline guard to ensure we return within 115 seconds.
"""
import time
import numpy as np
from scipy.optimize import minimize

N = 50
_DEADLINE = 115.0  # seconds, safety margin before 120s kill

def _autoconvolution_peak(x):
    """Compute max of autoconvolution of x (length N) using FFT."""
    n_fft = 1
    while n_fft < 2 * N:
        n_fft <<= 1
    X = np.fft.rfft(x, n=n_fft)
    conv = np.fft.irfft(X * X, n=n_fft)
    return np.max(np.abs(conv[:2 * N - 1]))

def _score(x):
    """Compute L2^2 / (L1 * sup_autoconv)."""
    x = np.asarray(x, dtype=np.float64)
    l2_sq = np.sum(x * x)
    l1 = np.sum(x)
    if l1 < 1e-15:
        return 0.0
    peak = _autoconvolution_peak(x)
    if peak < 1e-15:
        return 0.0
    return l2_sq / (l1 * peak)

def _build_dct_basis(n_basis):
    """
    Build a set of DCT-like basis functions of length N.
    We use cosine modes: phi_k(i) = cos(pi * k * (i + 0.5) / N)
    for k = 0, 1, ..., n_basis-1
    """
    i = np.arange(N)
    basis = np.zeros((n_basis, N))
    for k in range(n_basis):
        basis[k] = np.cos(np.pi * k * (i + 0.5) / N)
    # Normalize each basis vector to unit L2 norm
    norms = np.sqrt(np.sum(basis ** 2, axis=1))
    norms[norms < 1e-12] = 1.0
    basis /= norms[:, None]
    return basis

def _coeffs_to_heights(coeffs, basis):
    """Convert non-negative coefficients to heights."""
    h = np.sum(coeffs[:, None] * basis, axis=0)
    return np.maximum(h, 0.0)

def _smooth_max(x, beta=100.0):
    m = np.max(x)
    if m < 0:
        m = 0
    return m + np.log(np.sum(np.exp(beta * (x - m)))) / beta

def _negative_ratio_smooth(params, basis, beta=100.0):
    """
    Objective for L-BFGS-B: minimize -log(ratio) approximately.
    params are log-coefficients to ensure non-negativity.
    """
    coeffs = np.exp(params)
    h = _coeffs_to_heights(coeffs, basis)
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 1e10
    h = h / l1  # normalize to sum 1
    l2_sq = np.sum(h ** 2)
    if l2_sq < 1e-15:
        return 1e10
    conv = np.convolve(h, h, mode='full')
    max_conv = _smooth_max(conv, beta)
    if max_conv < 1e-15:
        return 1e10
    return -np.log(l2_sq) + np.log(max_conv)

def _local_search(x, max_iters=3000):
    """Local search: adjust individual heights to improve score."""
    x = np.array(x, dtype=np.float64)
    best_score = _score(x)
    best_x = x.copy()
    
    for it in range(max_iters):
        if time.time() > _DEADLINE - 10:
            break
        improved = False
        # Try moving mass between pairs
        for i in range(N):
            if best_x[i] <= 1e-12:
                continue
            for j in range(N):
                if i == j:
                    continue
                # Try moving a small fraction of mass
                for frac in [0.1, 0.25, 0.5, 0.75, 1.0]:
                    trial = best_x.copy()
                    trial[j] += trial[i] * frac
                    trial[i] *= (1.0 - frac)
                    s = _score(trial)
                    if s > best_score + 1e-9:
                        best_x = trial
                        best_score = s
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break
    
    return best_x, best_score

def construct():
    start_time = time.time()
    
    # Build basis
    n_basis = 8
    basis = _build_dct_basis(n_basis)
    
    best_x = np.ones(N)
    best_score = _score(best_x)
    
    # Try various initializations
    np.random.seed(42)
    
    # Initialization 1: Uniform (all coefficients equal)
    coeffs_init = np.ones(n_basis) / n_basis
    h = _coeffs_to_heights(coeffs_init, basis)
    if np.sum(h) > 0:
        h = h / np.sum(h)
    s = _score(h)
    if s > best_score:
        best_score = s
        best_x = h.copy()
    
    # Initialization 2: Random coefficients
    for trial in range(30):
        if time.time() > _DEADLINE - 5:
            break
        coeffs_init = np.random.exponential(1.0, n_basis)
        coeffs_init = coeffs_init / np.sum(coeffs_init)
        
        # Optimize using L-BFGS-B
        params_init = np.log(coeffs_init + 1e-10)
        best_local_score = 0.0
        best_local_h = None
        
        for beta in [20.0, 50.0, 100.0]:
            if time.time() > _DEADLINE - 10:
                break
            try:
                res = minimize(
                    _negative_ratio_smooth,
                    params_init,
                    args=(basis, beta),
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'ftol': 1e-10}
                )
                coeffs_opt = np.exp(res.x)
                h_opt = _coeffs_to_heights(coeffs_opt, basis)
                if np.sum(h_opt) > 0:
                    h_opt = h_opt / np.sum(h_opt)
                s_opt = _score(h_opt)
                if s_opt > best_local_score:
                    best_local_score = s_opt
                    best_local_h = h_opt.copy()
                    params_init = res.x
            except Exception:
                continue
        
        if best_local_h is not None and best_local_score > best_score:
            best_score = best_local_score
            best_x = best_local_h.copy()
    
    # Also try sparse Sidon-like initializations for comparison
    def _sidon_set(n, size):
        chosen = []
        diffs = set()
        for i in range(n):
            ok = True
            for j in chosen:
                d = abs(i - j)
                if d in diffs:
                    ok = False
                    break
            if ok:
                chosen.append(i)
                for j in chosen[:-1]:
                    diffs.add(abs(i - j))
            if len(chosen) >= size:
                break
        return chosen
    
    for k in [10, 15, 20]:
        if time.time() > _DEADLINE - 5:
            break
        sidon = _sidon_set(N, k)
        if len(sidon) < k:
            continue
        x = np.zeros(N)
        for idx in sidon:
            x[idx] = 1.0 / k
        s = _score(x)
        if s > best_score:
            best_score = s
            best_x = x.copy()
    
    # Final local search on best found
    if time.time() < _DEADLINE - 10:
        lx, ls = _local_search(best_x, max_iters=3000)
        if ls > best_score:
            best_score = ls
            best_x = lx
    
    # Return as list of floats
    return [float(v) for v in best_x]
