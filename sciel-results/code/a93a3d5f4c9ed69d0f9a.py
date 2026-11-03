"""C3-prime: minimize max|autoconv| / (sum h)^2 for signed step sequences.

Structural change: Model the sequence as the real part of a complex exponential sum
with continuous phases. Optimize phases via gradient descent on a smooth surrogate
(soft-max of |autoconv|) to find low-PAPR spectral configurations. Then discretize
to signed steps and polish with vectorized sign-flip local search.

This exploits the fact that low-autocorrelation sequences (Zadoff-Chu-like) correspond
to flat power spectra, which minimize peak autoconvolution relative to total energy.
"""

import numpy as np
import time
from scipy.optimize import minimize

N = 400
_DEADLINE = 135.0  # safety margin before 150s kill


def _autoconv_fft(h):
    """Compute linear autoconvolution via FFT."""
    n_fft = 2 * N
    H = np.fft.rfft(h, n=n_fft)
    ac = np.fft.irfft(H * np.conj(H), n=n_fft)[:2 * N - 1]
    return ac


def _score(h):
    """Compute max|autoconv| / (sum h)^2 using FFT."""
    s = h.sum()
    if abs(s) < 1e-12:
        return float('inf')
    ac = _autoconv_fft(h)
    return np.max(np.abs(ac)) / (s * s)


def _soft_max_abs(ac, beta=10.0):
    """Smooth approximation to max(|ac|) using log-sum-exp."""
    x = np.abs(ac)
    m = np.max(x)
    if m < 1e-12:
        return 0.0
    # log-sum-exp trick for numerical stability
    shifted = beta * (x - m)
    # Avoid overflow in exp
    shifted = np.clip(shifted, -100, 100)
    return m + np.log(np.sum(np.exp(shifted))) / beta


def _objective_phases(phases, beta=10.0):
    """Objective: soft-max of |autoconv| of real part of exp(i*phases) / (sum)^2."""
    # phases has shape (N,)
    z = np.cos(phases) + 1j * np.sin(phases)
    h = np.real(z)
    s = h.sum()
    if abs(s) < 1e-8:
        return 1e10
    ac = _autoconv_fft(h)
    sm = _soft_max_abs(ac, beta)
    return sm / (s * s)


def _grad_objective_phases(phases, beta=10.0):
    """Gradient of objective w.r.t. phases using automatic differentiation approximation."""
    z = np.cos(phases) + 1j * np.sin(phases)
    h = np.real(z)
    s = h.sum()
    if abs(s) < 1e-8:
        return np.zeros(N)
    
    ac = _autoconv_fft(h)
    x = np.abs(ac)
    m = np.max(x)
    if m < 1e-12:
        return np.zeros(N)
    shifted = beta * (x - m)
    shifted = np.clip(shifted, -100, 100)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    
    # d(soft_max)/d(x_i) = beta * exp(beta*(x_i - m)) / sum(exp)
    d_sm_dx = beta * exp_shifted / sum_exp
    
    # d|x_i|/dx_i = sign(ac_i) where x_i = |ac_i|
    sign_ac = np.sign(ac)
    d_sm_dac = d_sm_dx * sign_ac
    
    # d(ac_k)/d(h_j) = h_{j+k} for k>=0, h_{j-k} for k<0 (linear autoconv)
    # Actually ac_k = sum_j h_j * h_{j+k} for k>=0
    # d(ac_k)/d(h_j) = h_{j+k} + h_{j-k} (for valid indices)
    
    # Compute gradient via FFT for efficiency
    # ac = ifft(|H|^2), H = fft(h)
    # d(ac)/d(h) is a convolution-like operation
    
    # For simplicity, use numerical gradient on a few dimensions or 
    # compute analytically
    # d(ac_k)/d(h_j) = sum over the autoconv definition
    # ac_k = sum_j h_j h_{j+k}
    # d(ac_k)/d(h_j) = h_{j+k} + h_{j-k} (with boundary handling)
    
    # Let's compute d_obj/d(h) first, then d_obj/d(phases) = d_obj/d(h) * dh/d(phases)
    # dh_j/d(phases_j) = -sin(phases_j)
    
    # Compute d_obj/d(h_j):
    # obj = soft_max(|ac|) / s^2
    # d_obj/d(h_j) = (1/s^2) * sum_k d_sm_dac_k * d(ac_k)/d(h_j) - 2*soft_max/s^3 * 1
    
    # d(ac_k)/d(h_j): ac_k = sum_i h_i h_{i+k}
    # d(ac_k)/d(h_j) = h_{j+k} + h_{j-k} if indices valid
    # This is a convolution of h with itself, essentially
    
    # More carefully:
    # ac_k = sum_i h_i * h_{i+k}
    # d(ac_k)/d(h_j) = h_{j+k} (from first factor) + h_{j-k} (from second factor, when i+k=j => i=j-k)
    # So d(ac_k)/d(h_j) = h_{j+k} + h_{j-k}
    
    # Sum over k: sum_k d_sm_dac_k * (h_{j+k} + h_{j-k})
    # = sum_k d_sm_dac_k * h_{j+k} + sum_k d_sm_dac_k * h_{j-k}
    # = (d_sm_dac convolved with h)[j] + (d_sm_dac convolved with h)[j]  (by symmetry)
    # Actually these are cross-correlations
    
    # Let's just do it via FFT for the full gradient
    # d_obj/d(h) = (1/s^2) * 2 * ifft(fft(d_sm_dac) * conj(fft(h))) - 2*soft_max/s^3
    
    # Wait, let me think more carefully.
    # ac = ifft(|fft(h)|^2)
    # This is the circular autoconvolution, but we're using linear by zero-padding.
    
    # For linear autoconv with zero-padding:
    # ac_k = sum_j h_j h_{j+k} for k = -(N-1), ..., N-1
    # In our indexing ac[0..2N-2], ac[k] corresponds to lag k-(N-1)
    
    # Let's use the fact that d(ac)/d(h) is a Toeplitz matrix.
    # For computational efficiency, let's just do a numerical gradient estimate
    # on the phase parameters using finite differences, but that's slow.
    
    # Alternative: use the chain rule with FFT
    # Let H = fft(h, n=2N)
    # ac_circular = ifft(H * conj(H))
    # d(ac_circular_k)/d(h_j) = ... 
    
    # For now, let's use a simpler approach: compute the gradient numerically
    # but only for a subset of phases, or use a different optimization strategy.
    
    # Actually, let's just return a zero gradient and use Nelder-Mead or another
    # derivative-free method, or compute the gradient via finite differences.
    
    # Let's do finite differences for the gradient
    eps = 1e-5
    grad = np.zeros(N)
    for i in range(N):
        phases_plus = phases.copy()
        phases_plus[i] += eps
        phases_minus = phases.copy()
        phases_minus[i] -= eps
        f_plus = _objective_phases(phases_plus, beta)
        f_minus = _objective_phases(phases_minus, beta)
        grad[i] = (f_plus - f_minus) / (2 * eps)
    
    return grad


def _discretize_phases(phases):
    """Convert continuous phases to signed step sequence."""
    z = np.cos(phases) + 1j * np.sin(phases)
    h = np.real(z)
    # Discretize to signs
    return np.sign(h).astype(np.float64)


def _local_search_flip(h, max_steps=5000, deadline=None):
    """Vectorized sign-flip local search."""
    best_h = h.copy()
    best_val = _score(best_h)
    rng = np.random.default_rng(123)
    
    for step in range(max_steps):
        if deadline is not None and time.time() > deadline:
            break
        
        # Try flipping a random subset of positions
        n_flips = rng.integers(1, 5)
        indices = rng.choice(N, size=n_flips, replace=False)
        
        h_trial = best_h.copy()
        h_trial[indices] *= -1.0
        
        val = _score(h_trial)
        if val < best_val:
            best_val = val
            best_h = h_trial
    
    return best_h, best_val


def _local_search_block(h, max_steps=2000, deadline=None):
    """Block sign-flip local search."""
    best_h = h.copy()
    best_val = _score(best_h)
    rng = np.random.default_rng(456)
    
    for step in range(max_steps):
        if deadline is not None and time.time() > deadline:
            break
        
        # Try flipping a contiguous block
        start = rng.integers(0, N)
        length = rng.integers(1, N // 4)
        end = min(start + length, N)
        
        h_trial = best_h.copy()
        h_trial[start:end] *= -1.0
        
        val = _score(h_trial)
        if val < best_val:
            best_val = val
            best_h = h_trial
    
    return best_h, best_val


def construct():
    """Return a sequence of N signed step heights minimizing C3'."""
    t0 = time.time()
    deadline = t0 + _DEADLINE
    
    best_val = float('inf')
    best_h = np.ones(N, dtype=np.float64)
    
    # Candidate 0: constant ones (baseline)
    val = _score(best_h)
    if val < best_val:
        best_val = val
    
    # Candidate 1: alternating signs
    h_alt = np.ones(N, dtype=np.float64)
    h_alt[1::2] = -1.0
    val = _score(h_alt)
    if val < best_val:
        best_val, best_h = val, h_alt
    
    # Candidate 2: Zadoff-Chu-like sequence
    # Zadoff-Chu: z_n = exp(-i*pi*n*(n+1)/N) for N odd, or exp(-i*pi*n^2/N) for N even
    # Take real part and discretize
    n = np.arange(N)
    # For even N, use exp(-i*pi*n^2/N)
    z_chu = np.exp(-1j * np.pi * n * n / N)
    h_chu = np.sign(np.real(z_chu)).astype(np.float64)
    val = _score(h_chu)
    if val < best_val:
        best_val, best_h = val, h_chu
    
    # Candidate 3: Frank-Zadoff-Chu code
    # Frank-Zadoff-Chu: block structure with Zadoff-Chu within blocks
    # For N=400, use block size 20, 20 blocks
    block_size = 20
    n_blocks = N // block_size
    h_fzc = np.zeros(N, dtype=np.float64)
    for b in range(n_blocks):
        # Phase offset for block b
        phase_offset = 2 * np.pi * b / n_blocks
        n_block = np.arange(block_size)
        z_block = np.exp(-1j * np.pi * n_block * n_block / block_size + 1j * phase_offset)
        h_fzc[b*block_size:(b+1)*block_size] = np.sign(np.real(z_block))
    val = _score(h_fzc)
    if val < best_val:
        best_val, best_h = val, h_fzc
    
    # Candidate 4: Phase optimization via gradient descent
    if time.time() < deadline - 60:
        # Initialize phases with Zadoff-Chu-like pattern
        n = np.arange(N)
        phases_init = -np.pi * n * n / N  # Zadoff-Chu phase
        
        # Try multiple initializations
        for init_idx in range(3):
            if time.time() > deadline - 30:
                break
            
            if init_idx == 0:
                phases0 = phases_init.copy()
            elif init_idx == 1:
                phases0 = np.random.default_rng(init_idx * 100).standard_normal(N)
            else:
                # Random Zadoff-Chu-like with random frequency
                freq = np.random.default_rng(init_idx * 200).integers(1, N // 2)
                phases0 = -np.pi * n * n * freq / N
            
            # Optimize phases using L-BFGS-B
            try:
                result = minimize(
                    _objective_phases,
                    phases0,
                    method='L-BFGS-B',
                    options={'maxiter': 50, 'ftol': 1e-10, 'gtol': 1e-6,
                             'maxfun': 200, 'maxls': 20}
                )
                phases_opt = result.x
                h_opt = _discretize_phases(phases_opt)
                val = _score(h_opt)
                if val < best_val:
                    best_val, best_h = val, h_opt
            except Exception:
                pass
    
    # Candidate 5: Local search on best found
    if time.time() < deadline - 30:
        best_h, best_val = _local_search_flip(best_h, max_steps=3000, deadline=deadline - 15)
    
    # Candidate 6: Block local search
    if time.time() < deadline - 15:
        best_h, best_val = _local_search_block(best_h, max_steps=1000, deadline=deadline - 5)
    
    # Final polish: single flip search
    if time.time() < deadline - 5:
        best_h, best_val = _local_search_flip(best_h, max_steps=2000, deadline=deadline - 2)
    
    return best_h
