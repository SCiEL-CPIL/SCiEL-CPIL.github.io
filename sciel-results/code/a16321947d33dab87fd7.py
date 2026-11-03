"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _autoconv_max_abs(h):
    """Compute max absolute value of linear autoconvolution using FFT for speed."""
    n = len(h)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)
    return np.max(np.abs(ac))

def _score(h):
    """Compute the objective value: max|autoconv| / (sum h)^2."""
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    return _autoconv_max_abs(h) / (s * s)

def _spectral_flat_search():
    """
    Directly optimize in frequency domain: minimize max_k |H[k]|^2 / H[0]^2
    by making the magnitude spectrum as flat as possible.
    
    For a real sequence, H[-k] = conj(H[k]), so we parameterize by H[0], H[1..N//2].
    The constraint is that the IDFT gives a real sequence (automatically satisfied
    by conjugate symmetry).
    
    We use a smooth surrogate: minimize the variance of |H[k]|^2 / H[0]^2
    plus a small max-term approximation.
    """
    t_start = time.time()
    deadline = t_start + 60.0
    
    best_score = np.inf
    best_h = np.ones(N) / N
    
    # Generate candidate spectra with controlled flatness
    # A chirp exp(i*pi*n^2/N) has flat magnitude spectrum
    # We try various quadratic phase modulations
    
    i = np.arange(N)
    
    candidates = []
    
    # Quadratic chirps with different rates
    for rate in [0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0]:
        h = np.exp(1j * np.pi * rate * i**2 / N)
        h_real = np.real(h)
        if abs(np.sum(h_real)) > 1e-9:
            h_real = h_real / np.sum(h_real)
            candidates.append(h_real)
    
    # Cubic chirps
    for rate in [0.1, 0.5, 1.0, 2.0]:
        h = np.exp(1j * np.pi * rate * i**3 / N**2)
        h_real = np.real(h)
        if abs(np.sum(h_real)) > 1e-9:
            h_real = h_real / np.sum(h_real)
            candidates.append(h_real)
    
    # Sum of two chirps
    for r1 in [0.5, 1.0, 2.0]:
        for r2 in [0.3, 0.7, 1.5]:
            h = np.exp(1j * np.pi * r1 * i**2 / N) + np.exp(1j * np.pi * r2 * i**2 / N)
            h_real = np.real(h)
            if abs(np.sum(h_real)) > 1e-9:
                h_real = h_real / np.sum(h_real)
                candidates.append(h_real)
    
    # Random phases (deterministic seed)
    rng = np.random.RandomState(42)
    for _ in range(50):
        phases = rng.uniform(0, 2*np.pi, N)
        h = np.exp(1j * phases)
        h_real = np.real(h)
        if abs(np.sum(h_real)) > 1e-9:
            h_real = h_real / np.sum(h_real)
            candidates.append(h_real)
    
    # Evaluate all candidates
    for h_cand in candidates:
        if time.time() > deadline - 5:
            break
        s = _score(h_cand)
        if s < best_score:
            best_score = s
            best_h = h_cand.copy()
    
    # Refine best with local search
    if time.time() < deadline - 3:
        h_refined = best_h.copy()
        best_local = best_score
        for _ in range(3):
            improved = False
            for idx in range(N):
                if time.time() > deadline - 1:
                    break
                h_refined[idx] = -h_refined[idx]
                new_val = _score(h_refined)
                if new_val < best_local:
                    best_local = new_val
                    improved = True
                else:
                    h_refined[idx] = -h_refined[idx]
            if not improved:
                break
        
        if best_local < best_score:
            best_score = best_local
            best_h = h_refined.copy()
    
    return best_h, best_score

def _spectral_gradient_descent():
    """
    Use gradient descent in the time domain with a smooth approximation of the max-abs
    autoconvolution. We use a soft-max approximation: 
    soft_max(x, t) = (1/t) * log(sum(exp(t*x)))
    applied to the autoconvolution values.
    
    This gives a differentiable objective that we can optimize with gradient methods.
    """
    t_start = time.time()
    deadline = t_start + 50.0
    
    i = np.arange(N)
    h0 = np.cos(2.0 * np.pi * i**2 / N)
    h0 = h0 / np.sum(h0)
    
    def soft_max_obj(x, t=5.0):
        """Soft-max approximation of max|autoconv| / (sum)^2."""
        h = x
        s = np.sum(h)
        if abs(s) < 1e-9:
            return 1e10
        n = len(h)
        size = 2 * n - 1
        fsize = 1
        while fsize < size:
            fsize *= 2
        H = np.fft.rfft(h, fsize)
        ac = np.fft.irfft(H * H, fsize)
        # Soft max of |ac|
        ac_abs = np.abs(ac)
        max_val = np.max(ac_abs)
        if max_val < 1e-10:
            return 0.0
        # Use log-sum-exp for stability
        shifted = ac_abs - max_val
        soft_max_val = max_val + np.log(np.sum(np.exp(t * shifted))) / t
        return soft_max_val / (s * s)
    
    # Use L-BFGS-B with projection onto sum=1
    best_h = h0.copy()
    best_score = _score(h0)
    
    # Try a few iterations of projected gradient descent
    lr = 0.01
    h = h0.copy()
    for step in range(200):
        if time.time() > deadline - 5:
            break
        # Compute gradient numerically (central difference)
        eps = 1e-5
        grad = np.zeros(N)
        for j in range(0, N, 4):  # Subsample for speed
            h_plus = h.copy()
            h_plus[j] += eps
            h_minus = h.copy()
            h_minus[j] -= eps
            fp = soft_max_obj(h_plus)
            fm = soft_max_obj(h_minus)
            grad[j] = (fp - fm) / (2 * eps)
            # Also for j+1, j+2, j+3
            for jj in [j+1, j+2, j+3]:
                if jj < N:
                    h_plus[jj] = h[jj]
                    h_minus[jj] = h[jj]
                    h_plus = h.copy()
                    h_plus[jj] += eps
                    h_minus = h.copy()
                    h_minus[jj] -= eps
                    fp = soft_max_obj(h_plus)
                    fm = soft_max_obj(h_minus)
                    grad[jj] = (fp - fm) / (2 * eps)
        
        # Project gradient: subtract mean to maintain sum
        grad_proj = grad - np.mean(grad)
        
        h_new = h - lr * grad_proj
        # Normalize to sum = 1
        s = np.sum(h_new)
        if abs(s) > 1e-9:
            h_new = h_new / s
        
        score_new = _score(h_new)
        if score_new < best_score:
            best_score = score_new
            best_h = h_new.copy()
            h = h_new.copy()
        else:
            # Reduce learning rate
            lr *= 0.8
            if lr < 1e-6:
                break
    
    # Local search
    if time.time() < deadline - 3:
        h_refined = best_h.copy()
        best_local = best_score
        for _ in range(3):
            improved = False
            for idx in range(N):
                if time.time() > deadline - 1:
                    break
                h_refined[idx] = -h_refined[idx]
                new_val = _score(h_refined)
                if new_val < best_local:
                    best_local = new_val
                    improved = True
                else:
                    h_refined[idx] = -h_refined[idx]
            if not improved:
                break
        
        if best_local < best_score:
            best_score = best_local
            best_h = h_refined.copy()
    
    return best_h, best_score

def construct():
    t_start = time.time()
    deadline = t_start + 115.0  # Safety margin (120s limit)
    
    best_score = np.inf
    best_h = np.ones(N) / N
    
    # Baseline
    base_score = _score(best_h)
    best_score = min(best_score, base_score)
    
    # Strategy 1: Spectral flat search (fast, deterministic)
    if time.time() < deadline - 30:
        h_spec, score_spec = _spectral_flat_search()
        if score_spec < best_score:
            best_score = score_spec
            best_h = h_spec.copy()
    
    # Strategy 2: Gradient descent with soft-max
    if time.time() < deadline - 15:
        h_grad, score_grad = _spectral_gradient_descent()
        if score_grad < best_score:
            best_score = score_grad
            best_h = h_grad.copy()
    
    # Final local search: sign flips
    if time.time() < deadline - 3:
        h_refined = best_h.copy()
        best_local = best_score
        for _ in range(5):
            improved = False
            for idx in range(N):
                if time.time() > deadline - 1:
                    break
                h_refined[idx] = -h_refined[idx]
                new_val = _score(h_refined)
                if new_val < best_local:
                    best_local = new_val
                    improved = True
                else:
                    h_refined[idx] = -h_refined[idx]
            if not improved:
                break
        
        if best_local < best_score:
            best_score = best_local
            best_h = h_refined.copy()
    
    return best_h

# EVOLVE-BLOCK-END
