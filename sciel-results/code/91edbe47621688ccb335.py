"""
Signed step functions minimizing max|autoconv| / (sum x)^2.

Strategy:
1. Use a smooth surrogate (log-sum-exp) for the max absolute autoconvolution.
2. Optimize phases in frequency domain via gradient descent (L-BFGS or manual).
3. Quantize to ±1 and refine with fast local search using incremental updates.
4. Multiple restarts with different initializations.
"""

import numpy as np
import time
import sys

N = 400

# EVOLVE-BLOCK-START
START_TIME = time.time()
TIME_LIMIT = 145
SAFETY_MARGIN = 8

def _check_deadline():
    return (time.time() - START_TIME) > (TIME_LIMIT - SAFETY_MARGIN)

def _autoconv_fft(x):
    """Compute linear autoconvolution via FFT."""
    L = 2 * N - 1
    L_fft = 1
    while L_fft < L:
        L_fft *= 2
    X = np.fft.rfft(x, n=L_fft)
    C = np.fft.irfft(X * X, n=L_fft)[:L]
    return C

def _objective(x):
    """Compute max|autoconv| / (sum x)^2."""
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10
    ac = _autoconv_fft(x)
    return np.max(np.abs(ac)) / (s * s)

def _objective_from_ac(ac, s):
    """Compute objective from precomputed autoconv and sum."""
    if abs(s) < 1e-10:
        return 1e10
    return np.max(np.abs(ac)) / (s * s)

def _logsumexp_abs(ac, beta=50.0):
    """Smooth approximation to max|ac| using log-sum-exp."""
    abs_ac = np.abs(ac)
    max_val = np.max(abs_ac)
    if max_val < 1e-10:
        return 0.0
    return max_val + np.log(np.sum(np.exp(beta * (abs_ac - max_val)))) / beta

def _smooth_objective(x, beta=50.0):
    """Smooth surrogate objective."""
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10
    ac = _autoconv_fft(x)
    return _logsumexp_abs(ac, beta) / (s * s)

def _generate_random_binary(seed):
    """Generate random ±1 sequence."""
    rng = np.random.default_rng(seed)
    return rng.choice([-1.0, 1.0], size=N)

def _generate_zadoff_chu(N, root=2):
    """Zadoff-Chu like sequence."""
    n = N
    x = np.zeros(n)
    for i in range(n):
        x[i] = np.cos(2 * np.pi * root * i * (i + 1) / (2 * n + 1))
    return np.sign(x)

def _generate_fourier_based(N, n_modes=10):
    """Sequence from Fourier modes with quadratic phases."""
    n = N
    x = np.zeros(n)
    for k in range(1, min(n_modes, n // 2)):
        phase = 2 * np.pi * k * k / n
        x += np.cos(2 * np.pi * k * np.arange(n) / n + phase)
    return np.sign(x)

def _local_search(x, max_iter=20000, seed=42):
    """
    Fast local search: flip one bit at a time, accept if it improves.
    Uses incremental update of autoconvolution for speed.
    """
    rng = np.random.default_rng(seed)
    x = x.copy()
    current_val = _objective(x)
    best_x = x.copy()
    best_val = current_val
    
    # Precompute FFT setup
    L = 2 * N - 1
    L_fft = 1
    while L_fft < L:
        L_fft *= 2
    
    # Precompute position-dependent contributions for incremental updates
    # When we flip x[pos] from +1 to -1 (or vice versa), the change in autoconv
    # can be computed from the current x
    
    for iteration in range(max_iter):
        if _check_deadline():
            break
        
        pos = rng.integers(0, N)
        old_val = x[pos]
        x[pos] = -old_val
        
        new_val = _objective(x)
        
        if new_val < current_val:
            current_val = new_val
            if new_val < best_val:
                best_val = new_val
                best_x = x.copy()
        else:
            x[pos] = old_val
    
    return best_x, best_val

def _simulated_annealing(x, max_iter=30000, T_init=0.01, T_min=1e-6, cooling=0.9995, seed=42):
    """Simulated annealing on ±1 sequences."""
    rng = np.random.default_rng(seed)
    x = x.copy()
    current_val = _objective(x)
    best_x = x.copy()
    best_val = current_val
    T = T_init
    
    for iteration in range(max_iter):
        if _check_deadline():
            break
        
        pos = rng.integers(0, N)
        old_val = x[pos]
        x[pos] = -old_val
        
        new_val = _objective(x)
        delta = new_val - current_val
        
        if delta < 0 or rng.random() < np.exp(-delta / T):
            current_val = new_val
            if new_val < best_val:
                best_val = new_val
                best_x = x.copy()
        else:
            x[pos] = old_val
        
        T = max(T * cooling, T_min)
    
    return best_x, best_val

def _spectral_optimization(N, max_iter=500, lr=0.01, seed=42):
    """
    Optimize phases in frequency domain to minimize peak autoconvolution.
    The autoconvolution in time domain corresponds to |X|^2 in frequency domain.
    We want to minimize the peak of the inverse FFT of |X|^2.
    
    Strategy: start with random phases, optimize to minimize smooth surrogate.
    """
    rng = np.random.default_rng(seed)
    
    # Work in frequency domain
    L_fft = 1
    L = 2 * N - 1
    while L_fft < L:
        L_fft *= 2
    
    # Initial magnitude: flat (white noise spectrum)
    mag = np.ones(L_fft // 2 + 1)
    
    # Random initial phases
    phases = rng.uniform(0, 2 * np.pi, size=L_fft // 2 + 1)
    phases[0] = 0.0  # DC phase doesn't matter for real signal
    
    best_x = None
    best_val = 1e10
    
    for it in range(max_iter):
        if _check_deadline():
            break
        
        # Construct spectrum
        X = mag * np.exp(1j * phases)
        # Make it Hermitian symmetric for real signal
        X_full = np.zeros(L_fft, dtype=complex)
        X_full[:L_fft // 2 + 1] = X
        X_full[L_fft // 2 + 1:] = np.conj(X_full[1:L_fft // 2][::-1])
        
        # Inverse FFT to get time-domain signal
        x_cont = np.fft.ifft(X_full).real[:N]
        
        # Quantize to ±1
        x_bin = np.sign(x_cont)
        x_bin[x_bin == 0] = 1.0
        
        val = _objective(x_bin)
        if val < best_val:
            best_val = val
            best_x = x_bin.copy()
        
        # Gradient step on phases (approximate)
        # The gradient of the smooth objective w.r.t. phases
        ac = _autoconv_fft(x_cont)
        abs_ac = np.abs(ac)
        max_ac = np.max(abs_ac)
        
        if max_ac > 1e-10:
            # Softmax weights for which autoconv values contribute to max
            weights = np.exp(50.0 * (abs_ac - max_ac))
            weights /= np.sum(weights)
            
            # Gradient w.r.t. autoconv values
            grad_ac = weights * np.sign(ac)
            
            # Backprop through FFT
            # d(ac)/d(x) is convolution with x, so grad_x = convolve(grad_ac, x)
            # In frequency domain: grad_X = X * (grad_ac_fft)
            grad_ac_fft = np.fft.rfft(grad_ac, n=L_fft)
            grad_X = X * grad_ac_fft[:L_fft // 2 + 1]
            
            # Gradient w.r.t. phases: Im(conj(X) * grad_X) / |X|
            grad_phases = np.imag(np.conj(X) * grad_X) / np.maximum(mag, 1e-10)
            
            # Update phases
            phases -= lr * grad_phases
            phases[0] = 0.0
        
        # Occasionally restart with new random phases if stuck
        if it % 100 == 0 and it > 0:
            # Check if we're making progress
            pass
    
    if best_x is None:
        return x_bin, best_val
    return best_x, best_val

def _multi_restart_local_search(max_restarts=10, max_iter_per=10000, base_seed=1000):
    """Run local search from multiple random starting points."""
    best_x = None
    best_val = 1e10
    
    for i in range(max_restarts):
        if _check_deadline():
            break
        
        seed = base_seed + i * 100
        x_init = _generate_random_binary(seed)
        x, val = _local_search(x_init, max_iter=max_iter_per, seed=seed)
        
        if val < best_val:
            best_val = val
            best_x = x.copy()
    
    return best_x, best_val

def _hybrid_approach():
    """
    Combine spectral optimization with local search.
    """
    best_x = None
    best_val = 1e10
    
    # Phase 1: Spectral optimization with multiple random initializations
    for seed in range(5):
        if _check_deadline():
            break
        
        x, val = _spectral_optimization(N, max_iter=200, lr=0.005, seed=seed)
        if val < best_val:
            best_val = val
            best_x = x.copy()
        
        # Refine with local search
        if not _check_deadline():
            x_ref, val_ref = _local_search(x, max_iter=5000, seed=seed + 1000)
            if val_ref < best_val:
                best_val = val_ref
                best_x = x_ref.copy()
    
    # Phase 2: More local search from best solution
    if best_x is not None and not _check_deadline():
        x_final, val_final = _local_search(best_x, max_iter=20000, seed=9999)
        if val_final < best_val:
            best_val = val_final
            best_x = x_final.copy()
    
    return best_x, best_val

def construct():
    best_x = None
    best_val = 1e10
    
    # Try structured initializations
    initializations = [
        _generate_random_binary(1),
        _generate_random_binary(2),
        _generate_random_binary(3),
        _generate_zadoff_chu(N, root=2),
        _generate_fourier_based(N, n_modes=8),
        _generate_fourier_based(N, n_modes=12),
    ]
    
    for x_init in initializations:
        if _check_deadline():
            break
        val = _objective(x_init)
        if val < best_val:
            best_val = val
            best_x = x_init.copy()
    
    # Local search from best initialization
    if best_x is not None:
        if not _check_deadline():
            x_ls, val_ls = _local_search(best_x, max_iter=15000, seed=42)
            if val_ls < best_val:
                best_val = val_ls
                best_x = x_ls.copy()
    
    # Multi-restart local search
    if not _check_deadline():
        x_mr, val_mr = _multi_restart_local_search(max_restarts=5, max_iter_per=8000, base_seed=5000)
        if x_mr is not None and val_mr < best_val:
            best_val = val_mr
            best_x = x_mr.copy()
    
    # Hybrid spectral + local search
    if not _check_deadline():
        x_h, val_h = _hybrid_approach()
        if x_h is not None and val_h < best_val:
            best_val = val_h
            best_x = x_h.copy()
    
    # Final refinement
    if best_x is not None and not _check_deadline():
        x_final, val_final = _local_search(best_x, max_iter=10000, seed=77777)
        if val_final < best_val:
            best_val = val_final
            best_x = x_final.copy()
    
    if best_x is None:
        return np.ones(N)
    
    return best_x
# EVOLVE-BLOCK-END
