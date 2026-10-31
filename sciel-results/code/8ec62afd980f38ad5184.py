"""Rewritten program: SDP/spectral approach for Erdos minimum overlap."""
import time
import numpy as np

N = 100

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
import random

def _max_overlap_fft(h):
    """Compute max circular overlap of h with 1-h using FFT."""
    N = len(h)
    g = 1.0 - h
    H = np.fft.fft(h)
    G = np.fft.fft(g)
    overlaps = np.real(np.fft.ifft(H * np.conj(G)))
    return np.max(overlaps)

def _fix_sum(h, N):
    """Ensure h sums to exactly N/2 and is in [0,1]."""
    h = np.clip(h, 0.0, 1.0)
    diff = N / 2.0 - np.sum(h)
    if abs(diff) > 1e-14:
        if diff > 0:
            order = np.argsort(1.0 - h)
            for idx in order:
                if diff <= 1e-14:
                    break
                add = min(diff, 1.0 - h[idx])
                h[idx] += add
                diff -= add
        else:
            order = np.argsort(-h)
            for idx in order:
                if -diff <= 1e-14:
                    break
                sub = min(-diff, h[idx])
                h[idx] -= sub
                diff += sub
    return h

def _logsumexp_overlap(h, alpha=500.0):
    """Smooth surrogate for max overlap using log-sum-exp."""
    N = len(h)
    g = 1.0 - h
    H = np.fft.fft(h)
    G = np.fft.fft(g)
    overlaps = np.real(np.fft.ifft(H * np.conj(G)))
    # log-sum-exp for stability
    m = np.max(overlaps)
    if m < -1e10:
        return m
    return m + np.log(np.sum(np.exp((overlaps - m) * alpha)) / N) / alpha

def _try_two_block(N):
    """Two-block pattern: a repeated p times, then b repeated q times."""
    best_h = None
    best_val = float('inf')
    
    for p in range(1, N + 1):
        q = N - p
        if q == 0:
            continue
        # Sum constraint: p*a + q*b = N/2
        # Overlap analysis: for shift k, count how many pairs (i, i+k) are in same block vs different
        # This is complex; use numerical optimization
        def obj(x):
            a, b = x
            a = np.clip(a, 0, 1)
            b = np.clip(b, 0, 1)
            # Adjust to satisfy sum
            target = N / 2.0
            current_sum = p * a + q * b
            if abs(current_sum) > 1e-14:
                scale = target / current_sum
                a2 = np.clip(a * scale, 0, 1)
                b2 = np.clip(b * scale, 0, 1)
            else:
                a2, b2 = 0.5, 0.5
            h = np.concatenate([np.full(p, a2), np.full(q, b2)])
            h = _fix_sum(h, N)
            return _max_overlap_fft(h)
        
        # Try a few starting points
        for a0 in [0.3, 0.5, 0.7]:
            b0 = (N/2.0 - p * a0) / q if q > 0 else 0.5
            b0 = np.clip(b0, 0, 1)
            try:
                res = minimize(obj, [a0, b0], method='Nelder-Mead',
                               options={'maxiter': 100, 'xatol': 1e-7, 'fatol': 1e-9})
                val = obj(res.x)
                if val < best_val:
                    best_val = val
                    a2 = np.clip(res.x[0], 0, 1)
                    b2 = np.clip(res.x[1], 0, 1)
                    h = np.concatenate([np.full(p, a2), np.full(q, b2)])
                    h = _fix_sum(h, N)
                    best_h = h
            except Exception:
                pass
    
    return best_h, best_val

def _try_fourier_basis_optimize(N, n_basis=10, time_budget=15.0):
    """
    Represent h as h[i] = 0.5 + sum_{k=1}^{n_basis} a_k cos(2*pi*k*i/N) + b_k sin(2*pi*k*i/N)
    Optimize coefficients to minimize max overlap.
    """
    start = time.time()
    
    # Build Fourier basis
    i = np.arange(N)
    basis = np.zeros((N, 2 * n_basis))
    for k in range(1, n_basis + 1):
        basis[:, 2*(k-1)] = np.cos(2 * np.pi * k * i / N)
        basis[:, 2*(k-1)+1] = np.sin(2 * np.pi * k * i / N)
    
    def _h_from_coeffs(x):
        h = 0.5 + basis @ x
        return h
    
    def _objective(x):
        if time.time() - start > time_budget:
            return 1e10
        h = _h_from_coeffs(x)
        h = np.clip(h, 0, 1)
        h = _fix_sum(h, N)
        return _max_overlap_fft(h)
    
    best_x = None
    best_val = float('inf')
    
    for trial in range(5):
        if trial == 0:
            x0 = np.zeros(2 * n_basis)
        else:
            x0 = np.random.uniform(-0.1, 0.1, 2 * n_basis)
        
        try:
            res = minimize(_objective, x0, method='Nelder-Mead',
                           options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8})
            val = _objective(res.x)
            if val < best_val:
                best_val = val
                best_x = res.x
        except Exception:
            pass
    
    if best_x is None:
        return None, float('inf')
    
    h = _h_from_coeffs(best_x)
    h = np.clip(h, 0, 1)
    h = _fix_sum(h, N)
    return h, _max_overlap_fft(h)

def _try_spectral_equalization(N, n_basis=8, time_budget=20.0):
    """
    Directly optimize the Fourier coefficients to equalize the overlap spectrum.
    The overlap at shift k is related to the Fourier transform of h.
    overlap[k] = sum_i h[i]*(1-h[(i+k)%N])
    In Fourier domain: overlap = ifft(|H|^2 * something)
    
    Actually: overlap[k] = sum_i h[i] - sum_i h[i]*h[(i+k)%N]
    = N/2 - autocorr_h[k]
    where autocorr_h[k] = sum_i h[i]*h[(i+k)%N]
    
    So max overlap = N/2 - min_k autocorr_h[k]
    Minimizing max overlap = maximizing min autocorr_h[k]
    
    autocorr_h[k] = ifft(|H|^2)[k]
    
    So we want to maximize the minimum of ifft(|H|^2).
    """
    start = time.time()
    
    i = np.arange(N)
    # Use real Fourier basis for efficiency
    n_harmonics = min(n_basis, N // 2)
    basis = np.zeros((N, 2 * n_harmonics))
    for k in range(1, n_harmonics + 1):
        basis[:, 2*(k-1)] = np.cos(2 * np.pi * k * i / N)
        basis[:, 2*(k-1)+1] = np.sin(2 * np.pi * k * i / N)
    
    def _h_from_coeffs(x):
        h = 0.5 + basis @ x
        return h
    
    def _min_autocorr(h):
        H = np.fft.fft(h)
        power = np.abs(H) ** 2
        autocorr = np.real(np.fft.ifft(power))
        return np.min(autocorr)
    
    def _objective(x):
        if time.time() - start > time_budget:
            return 1e10
        h = _h_from_coeffs(x)
        h = np.clip(h, 0, 1)
        h = _fix_sum(h, N)
        # We want to maximize min autocorr, so minimize negative
        return -_min_autocorr(h)
    
    best_x = None
    best_val = float('inf')
    
    for trial in range(8):
        if trial == 0:
            x0 = np.zeros(2 * n_harmonics)
        elif trial == 1:
            # Small perturbation
            x0 = np.random.uniform(-0.05, 0.05, 2 * n_harmonics)
        else:
            x0 = np.random.uniform(-0.2, 0.2, 2 * n_harmonics)
        
        try:
            res = minimize(_objective, x0, method='Nelder-Mead',
                           options={'maxiter': 800, 'xatol': 1e-7, 'fatol': 1e-9})
            val = _objective(res.x)
            if val < best_val:
                best_val = val
                best_x = res.x
        except Exception:
            pass
    
    if best_x is None:
        return None, float('inf')
    
    h = _h_from_coeffs(best_x)
    h = np.clip(h, 0, 1)
    h = _fix_sum(h, N)
    score = _max_overlap_fft(h)
    return h, score

def _local_search(h, max_iter=3000, step_size=0.02, time_limit=10.0):
    """Local search with pairwise updates."""
    N = len(h)
    current = h.copy()
    current_score = _max_overlap_fft(current)
    start = time.time()
    
    for it in range(max_iter):
        if time.time() - start > time_limit:
            break
        i = random.randint(0, N - 1)
        j = random.randint(0, N - 1)
        if i == j:
            continue
        delta = random.uniform(-step_size, step_size)
        new_i = current[i] + delta
        new_j = current[j] - delta
        if new_i < 0 or new_i > 1 or new_j < 0 or new_j > 1:
            continue
        
        new_h = current.copy()
        new_h[i] = new_i
        new_h[j] = new_j
        
        new_score = _max_overlap_fft(new_h)
        if new_score < current_score - 1e-13:
            current = new_h
            current_score = new_score
    
    return current, current_score

def _differential_evolution(N, time_budget=20.0):
    """DE search with periodic restarts."""
    start = time.time()
    
    def _objective(x):
        if time.time() - start > time_budget - 3:
            return 1e10
        h = np.clip(x, 0, 1)
        h = _fix_sum(h, N)
        return _max_overlap_fft(h)
    
    bounds = [(0, 1)] * N
    try:
        res = differential_evolution(_objective, bounds, maxiter=40,
                                     popsize=12, tol=1e-6, seed=42,
                                     polish=True, init='sobol')
        h = np.clip(res.x, 0, 1)
        h = _fix_sum(h, N)
        return h, _max_overlap_fft(h)
    except Exception:
        return None, float('inf')

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    random.seed(42)
    np.random.seed(42)
    
    # Start with alternating pattern (score = N/4 = 25)
    best_h = np.zeros(N)
    best_h[0::2] = 0.5
    best_h[1::2] = 0.5
    best_score = _max_overlap_fft(best_h)
    
    # Try two-block patterns
    if time.time() < deadline - 80:
        h2b, s2b = _try_two_block(N)
        if h2b is not None and s2b < best_score:
            best_score = s2b
            best_h = h2b
    
    # Try Fourier basis optimization
    if time.time() < deadline - 50:
        h4, s4 = _try_fourier_basis_optimize(N, n_basis=8, time_budget=15.0)
        if h4 is not None and s4 < best_score:
            best_score = s4
            best_h = h4
    
    # Try spectral equalization
    if time.time() < deadline - 40:
        hs, ss = _try_spectral_equalization(N, n_basis=8, time_budget=20.0)
        if hs is not None and ss < best_score:
            best_score = ss
            best_h = hs
    
    # Local search refinement
    if time.time() < deadline - 20:
        h_ls, s_ls = _local_search(best_h, max_iter=3000, step_size=0.02, time_limit=10.0)
        if s_ls < best_score:
            best_score = s_ls
            best_h = h_ls
    
    # Differential evolution
    if time.time() < deadline - 15:
        h_de, s_de = _differential_evolution(N, time_budget=15.0)
        if h_de is not None and s_de < best_score:
            best_score = s_de
            best_h = h_de
    
    # Final local search
    if time.time() < deadline - 3:
        h_ls2, s_ls2 = _local_search(best_h, max_iter=1500, step_size=0.01, time_limit=5.0)
        if s_ls2 < best_score:
            best_score = s_ls2
            best_h = h_ls2
    
    # Ensure exact sum and bounds
    best_h = _fix_sum(best_h, N)
    
    return best_h
# EVOLVE-BLOCK-END
