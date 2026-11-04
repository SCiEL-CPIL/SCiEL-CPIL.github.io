"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.special import logsumexp

_start_time = time.time()
_TIME_LIMIT = 140  # safety margin before 150s kill

def _score_h(h):
    """
    Compute the maximum overlap of h with shifted (1-h).
    Uses FFT-based cross-correlation for efficiency.
    """
    sw = 2.0 / N
    autocorr = np.correlate(h, h, mode='full')
    R = sw * autocorr[N-1 : 2*N-1]
    C = 1.0 - R
    return np.max(C)

def _project(h):
    """
    Project h onto the feasible set: 0 <= h_i <= 1, sum(h) = N/2.
    """
    h = np.clip(h, 0, 1)
    diff = N / 2.0 - np.sum(h)
    for _ in range(50):
        if abs(diff) < 1e-12:
            break
        if diff > 0:
            room = 1.0 - h
            total_room = np.sum(room)
            if total_room < 1e-12:
                break
            h += diff / total_room * room
        else:
            room = h
            total_room = np.sum(room)
            if total_room < 1e-12:
                break
            h += diff / total_room * room
        h = np.clip(h, 0, 1)
        diff = N / 2.0 - np.sum(h)
    return h

def _spectral_design(alpha, beta):
    """
    Design h using spectral factorization.
    We want h to have approximately flat autocorrelation at non-zero lags.
    
    Target: R(k) ≈ 0.5 * sw for k=1,...,N-1 (flat non-zero lag autocorrelation)
    This means C(k) = 1 - R(k) ≈ 0.5 * (2 - sw) for all non-zero k.
    
    We construct h as a filtered version of a base signal.
    """
    # Start with alternating pattern
    h_base = np.zeros(N)
    h_base[::2] = 1.0
    
    # Apply a low-pass filter to smooth out high frequencies
    # The filter is designed to have a flat power spectrum at non-zero frequencies
    
    # Create filter in frequency domain
    freqs = np.fft.fftfreq(N)
    
    # Target power spectrum: flat at non-zero frequencies
    # H(f) should have |H(f)|^2 ≈ constant for f != 0
    # This is achieved by a raised cosine or similar window
    
    # Use a parameterized filter
    # alpha controls the bandwidth, beta controls the shape
    target_power = np.ones(N)
    target_power[0] = 0  # Remove DC component for the filter shape
    
    # Actually, let's think differently.
    # We want h such that its autocorrelation is flat at non-zero lags.
    # By Wiener-Khinchin, this means |H(f)|^2 is flat (except at f=0).
    
    # Let's construct h directly in frequency domain.
    # H(f) = A * exp(i*phi(f)) where |A| is chosen for flat power.
    
    # For a flat power spectrum, we can use:
    # |H(f)|^2 = c for f != 0, and |H(0)|^2 = (N/2)^2 / sw^2 (to satisfy sum constraint)
    
    # Actually, let's use a simpler approach:
    # Start with a binary sequence and optimize its phase in frequency domain.
    
    # Compute FFT of base signal
    H = np.fft.fft(h_base)
    
    # Modify the phase to create a more uniform autocorrelation
    # Use a random phase for each frequency bin
    rng = np.random.default_rng(int(alpha * 1000))
    phases = rng.uniform(0, 2 * np.pi, N)
    
    # Apply the phases
    H_modified = H * np.exp(1j * phases)
    
    # Inverse FFT
    h_modified = np.real(np.fft.ifft(H_modified))
    
    # Project to feasible set
    h_modified = _project(h_modified)
    
    return h_modified

def _gradient_descent_spectral(h_init, max_iter=2000, lr=0.01):
    """
    Gradient descent on the spectral representation.
    We optimize the phases in frequency domain to minimize max overlap.
    """
    h = h_init.copy()
    sw = 2.0 / N
    
    # Get initial FFT
    H = np.fft.fft(h)
    
    # Initialize phases
    magnitudes = np.abs(H)
    phases = np.angle(H)
    
    # Keep DC magnitude fixed (to preserve sum)
    dc_mag = magnitudes[0]
    
    best_h = h.copy()
    best_val = _score_h(best_h)
    
    for it in range(max_iter):
        if time.time() > _start_time + _TIME_LIMIT - 20:
            break
        
        # Compute current autocorrelation
        autocorr = np.correlate(h, h, mode='full')
        R = sw * autocorr[N-1 : 2*N-1]
        C = 1.0 - R
        
        # Find the shift with maximum overlap
        k_max = np.argmax(C)
        
        # Compute gradient of max overlap with respect to h
        # dC(k)/dh_i = -sw * h_{i+k} for i < N-k
        grad = np.zeros(N)
        for i in range(N - k_max):
            grad[i] -= sw * h[i + k_max]
        
        # Project gradient to feasible set (tangent space)
        # Sum constraint: sum(grad) should be 0
        grad -= np.mean(grad)
        
        # Update h
        h_new = h - lr * grad
        h_new = _project(h_new)
        
        # Evaluate
        val = _score_h(h_new)
        if val < best_val:
            best_h = h_new.copy()
            best_val = val
            h = h_new
        else:
            # Reduce learning rate
            lr *= 0.95
    
    return best_h, best_val

def construct():
    global _start_time
    deadline = _start_time + _TIME_LIMIT
    
    # Start with several different initializations
    best_h = None
    best_val = np.inf
    
    # 1. Uniform
    h0 = np.full(N, 0.5)
    val0 = _score_h(h0)
    best_h = h0.copy()
    best_val = val0
    
    # 2. Alternating binary
    h_alt = np.zeros(N)
    h_alt[::2] = 1.0
    h_alt = _project(h_alt)
    val_alt = _score_h(h_alt)
    if val_alt < best_val:
        best_h = h_alt.copy()
        best_val = val_alt
    
    # 3. Spectral designs with different parameters
    for alpha in [0.3, 0.5, 0.7, 1.0, 1.5]:
        if time.time() > deadline - 30:
            break
        try:
            h_spec = _spectral_design(alpha, 0.5)
            val_spec = _score_h(h_spec)
            if val_spec < best_val:
                best_h = h_spec.copy()
                best_val = val_spec
        except Exception:
            pass
    
    # 4. Gradient descent from best solution
    if time.time() < deadline - 10:
        try:
            h_grad, val_grad = _gradient_descent_spectral(best_h, max_iter=3000, lr=0.005)
            if val_grad < best_val:
                best_h = h_grad.copy()
                best_val = val_grad
        except Exception:
            pass
    
    # 5. Final polish with SLSQP on LSE objective
    if time.time() < deadline - 5:
        try:
            def _objective_lse(h, beta=50.0):
                sw = 2.0 / N
                autocorr = np.correlate(h, h, mode='full')
                R = sw * autocorr[N-1 : 2*N-1]
                C = 1.0 - R
                return logsumexp(beta * C) / beta
            
            bounds = [(0, 1)] * N
            constraints = [{'type': 'eq', 'fun': lambda h: np.sum(h) - N/2}]
            
            result = minimize(
                _objective_lse, best_h, method='SLSQP',
                bounds=bounds, constraints=constraints,
                options={'maxiter': 500, 'ftol': 1e-10}
            )
            if result.success:
                h_opt = _project(result.x)
                val_opt = _score_h(h_opt)
                if val_opt < best_val:
                    best_h = h_opt.copy()
                    best_val = val_opt
        except Exception:
            pass
    
    best_h = _project(best_h)
    return best_h
# EVOLVE-BLOCK-END
