"""
Second autocorrelation inequality: maximize (sum h_i^2) / (sum h_i * max_k (sum_i h_i h_{i+k}))
over 50 nonnegative heights on equal intervals of [-1/4, 1/4].

Strategy:
- Use DFT spectral parameterization: h = |DFT(z)|^2 / N (ensures non-negativity)
- Optimize z in complex space with L-BFGS-B (unconstrained in z-space)
- Multiple diverse initializations
- Deadline guard with safety margin
"""
import time
import numpy as np
from scipy.optimize import minimize

N = 50
DEADLINE = 110.0  # seconds, safety margin before 120s kill

def _autoconvolution(h):
    """Compute autoconvolution of h with itself (circular, length N)."""
    # Circular convolution via FFT
    H = np.fft.fft(h)
    conv = np.fft.ifft(H * H).real
    return conv

def _score(h):
    """Compute the ratio: (sum h^2) / (sum h * max autoconv)."""
    l2_sq = np.sum(h * h)
    l1 = np.sum(h)
    if l1 <= 0:
        return 0.0
    conv = _autoconvolution(h)
    sup = np.max(conv)
    if sup <= 0:
        return 0.0
    return l2_sq / (l1 * sup)

def _h_from_z(z):
    """Convert complex z (length N) to nonnegative h via |IFFT(z)|^2 / N."""
    h = np.abs(np.fft.ifft(z)) ** 2
    h = h / N  # normalize so sum is roughly 1
    return h

def _objective_and_grad(z_flat):
    """Objective to minimize: -score(h) where h = |IFFT(z)|^2/N.
    z_flat is real vector of length 2N (real and imaginary parts of z)."""
    z = z_flat[:N] + 1j * z_flat[N:]
    h = _h_from_z(z)
    s = _score(h)
    return -s  # minimize negative score

def _construct_initializations():
    """Generate diverse initial z vectors."""
    inits = []
    
    # 1. All-ones in h-space -> uniform h -> z is delta at 0
    z0 = np.zeros(N, dtype=complex)
    z0[0] = np.sqrt(N)  # so |IFFT(z0)|^2 = N/N = 1 each
    inits.append(np.concatenate([z0.real, z0.imag]))
    
    # 2. Random complex z
    rng = np.random.default_rng(42)
    for _ in range(20):
        z = rng.standard_normal(N) + 1j * rng.standard_normal(N)
        inits.append(np.concatenate([z.real, z.imag]))
    
    # 3. Low-frequency dominant
    for _ in range(10):
        z = np.zeros(N, dtype=complex)
        n_low = rng.integers(1, 5)
        idx = rng.choice(N, size=n_low, replace=False)
        for i in idx:
            z[i] = rng.standard_normal() + 1j * rng.standard_normal()
        inits.append(np.concatenate([z.real, z.imag]))
    
    # 4. Sparse h-like: a few peaks
    for _ in range(10):
        h_target = np.zeros(N)
        n_peaks = rng.integers(2, 8)
        peak_idx = rng.choice(N, size=n_peaks, replace=False)
        for i in peak_idx:
            h_target[i] = rng.uniform(0.5, 2.0)
        h_target = h_target / np.sum(h_target) * N  # scale
        # Find z such that |IFFT(z)|^2 ~ h_target (approximate via sqrt in freq domain)
        H = np.fft.fft(h_target)
        # This is approximate; just use it as a starting point
        z = np.sqrt(np.maximum(H, 0))  # rough
        inits.append(np.concatenate([z.real, z.imag]))
    
    return inits

def construct():
    start_time = time.time()
    best_score = 0.0
    best_h = np.ones(N)  # default: uniform
    
    inits = _construct_initializations()
    
    for init in inits:
        if time.time() - start_time > DEADLINE - 10:
            break
        
        try:
            res = minimize(
                _objective_and_grad,
                init,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            z_flat = res.x
            z = z_flat[:N] + 1j * z_flat[N:]
            h = _h_from_z(z)
            
            # Ensure nonnegativity (should be guaranteed but clip tiny negatives)
            h = np.maximum(h, 0.0)
            if np.sum(h) <= 0:
                continue
            
            s = _score(h)
            if s > best_score:
                best_score = s
                best_h = h.copy()
        except Exception:
            continue
    
    # If we have time, do a polish pass with tighter tolerance on the best
    if time.time() - start_time < DEADLINE - 5 and best_score > 0:
        try:
            # Re-optimize from best_h: convert back to z-space
            # h = |IFFT(z)|^2 / N, so IFFT(z) = sqrt(h * N) (phase ambiguous)
            # Use the phase from the previous z if available, or just use magnitude
            h_scaled = best_h * N
            # Take square root in time domain, then FFT to get z
            z_mag = np.sqrt(np.maximum(h_scaled, 0))
            z = np.fft.fft(z_mag)  # this gives a valid z
            z_flat = np.concatenate([z.real, z.imag])
            
            res = minimize(
                _objective_and_grad,
                z_flat,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            z_flat = res.x
            z = z_flat[:N] + 1j * z_flat[N:]
            h = _h_from_z(z)
            h = np.maximum(h, 0.0)
            if np.sum(h) > 0:
                s = _score(h)
                if s > best_score:
                    best_score = s
                    best_h = h.copy()
        except Exception:
            pass
    
    # Return as list of floats
    return [float(x) for x in best_h]
