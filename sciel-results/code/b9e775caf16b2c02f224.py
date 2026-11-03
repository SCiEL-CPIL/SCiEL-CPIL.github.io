"""Spectral shaping strategy for Erdos minimum overlap."""
import numpy as np
from scipy.optimize import minimize
import time

N = 100

# EVOLVE-BLOCK-START

def _max_overlap_fast(h):
    """Compute max overlap of h with shifted (1-h) using FFT-based circular autocorrelation."""
    H = np.fft.fft(h)
    autocorr = np.real(np.fft.ifft(H * np.conj(H)))
    min_autocorr = np.min(autocorr)
    max_overlap = N / 2.0 - min_autocorr
    return max_overlap


def _autocorr_from_spectrum(power):
    """Given power spectrum (real, non-negative), compute autocorrelation via IFFT."""
    # power[k] = |H[k]|^2, so autocorr = IFFT(power)
    autocorr = np.real(np.fft.ifft(power))
    return autocorr


def _h_from_spectrum(amplitudes, phases):
    """Reconstruct h from amplitudes and phases in frequency domain."""
    # Build H from amplitudes and phases
    H = np.zeros(N, dtype=complex)
    for k in range(N):
        H[k] = amplitudes[k] * np.exp(1j * phases[k])
    # Ensure Hermitian symmetry for real h
    # H[k] = conj(H[N-k]) for k=1..N-1
    for k in range(1, N // 2 + 1):
        H[N - k] = np.conj(H[k])
    h = np.real(np.fft.ifft(H))
    return h


def _project_to_feasible(h):
    """Project h to satisfy: sum = N/2, 0 <= h[i] <= 1."""
    h = h.copy()
    h = np.clip(h, 0, 1)
    target = N / 2.0
    current_sum = np.sum(h)
    diff = target - current_sum
    if abs(diff) > 1e-12:
        if diff > 0:
            room = 1.0 - h
            if np.sum(room) > 1e-12:
                h += diff * room / np.sum(room)
        else:
            if np.sum(h) > 1e-12:
                h += diff * h / np.sum(h)
        h = np.clip(h, 0, 1)
    # Final adjustment
    current_sum = np.sum(h)
    if abs(current_sum - target) > 1e-10:
        residual = target - current_sum
        if residual > 0:
            room = 1.0 - h
            mask = room > 1e-10
            if np.sum(room[mask]) > 0:
                h[mask] += residual / np.sum(mask)
        else:
            mask = h > 1e-10
            if np.sum(h[mask]) > 0:
                h[mask] += residual / np.sum(mask)
        h = np.clip(h, 0, 1)
    return h


def _spectral_objective(params):
    """
    Objective in spectral domain.
    params: [amplitudes (N/2+1), phases (N/2)]
    We want to maximize min autocorrelation, so minimize -min(autocorr).
    """
    half = N // 2
    amplitudes = params[:half + 1]
    phases = params[half + 1:]
    
    # Ensure amplitudes are non-negative
    amplitudes = np.maximum(amplitudes, 0)
    
    # Build H
    H = np.zeros(N, dtype=complex)
    H[0] = amplitudes[0]  # DC component, phase = 0
    for k in range(1, half + 1):
        H[k] = amplitudes[k] * np.exp(1j * phases[k])
        H[N - k] = np.conj(H[k])
    
    # Compute autocorrelation from power spectrum
    power = np.abs(H) ** 2
    autocorr = np.real(np.fft.ifft(power))
    
    # We want to maximize min(autocorr), so minimize -min(autocorr)
    # Add constraint penalty: sum(h) should be N/2
    h = np.real(np.fft.ifft(H))
    sum_penalty = (np.sum(h) - N / 2.0) ** 2
    bound_penalty = np.sum(np.maximum(0, h - 1.0) ** 2) + np.sum(np.maximum(0, -h) ** 2)
    
    return -np.min(autocorr) + 10.0 * (sum_penalty + bound_penalty)


def _spectral_gradient(params):
    """Numerical gradient for spectral objective."""
    eps = 1e-6
    grad = np.zeros_like(params)
    base = _spectral_objective(params)
    for i in range(len(params)):
        params_p = params.copy()
        params_p[i] += eps
        grad[i] = (_spectral_objective(params_p) - base) / eps
    return grad


def _spectral_optimize(h0, max_iters=2000, time_limit=10.0):
    """Optimize in spectral domain using BFGS."""
    start_time = time.time()
    half = N // 2
    
    # Initialize from h0
    H0 = np.fft.fft(h0)
    amplitudes = np.abs(H0)
    phases = np.angle(H0)
    
    # Ensure Hermitian symmetry in the parameterization
    # We only store amplitudes[0..half] and phases[1..half]
    amp_init = np.zeros(half + 1)
    phase_init = np.zeros(half)
    amp_init[0] = amplitudes[0]
    for k in range(1, half + 1):
        amp_init[k] = amplitudes[k]
        phase_init[k] = phases[k]
    
    params_init = np.concatenate([amp_init, phase_init])
    
    try:
        result = minimize(
            _spectral_objective,
            params_init,
            method='L-BFGS-B',
            options={'maxiter': max_iters, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        best_params = result.x
    except Exception:
        best_params = params_init
    
    # Reconstruct h
    amplitudes = np.maximum(best_params[:half + 1], 0)
    phases = best_params[half + 1:]
    
    H = np.zeros(N, dtype=complex)
    H[0] = amplitudes[0]
    for k in range(1, half + 1):
        H[k] = amplitudes[k] * np.exp(1j * phases[k])
        H[N - k] = np.conj(H[k])
    
    h = np.real(np.fft.ifft(H))
    h = _project_to_feasible(h)
    
    elapsed = time.time() - start_time
    score = _max_overlap_fast(h)
    return h, score, elapsed


def _coordinate_descent_spectral(h0, max_iters=3000, time_limit=10.0):
    """Coordinate descent in spatial domain with adaptive steps."""
    h = h0.copy()
    best_score = _max_overlap_fast(h)
    start_time = time.time()
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        i = np.random.randint(N)
        # Try multiple step sizes
        for step in [0.05, 0.1, 0.2, 0.025, 0.05, 0.1, 0.2]:
            for sign in [1, -1]:
                new_h = h.copy()
                new_h[i] = np.clip(new_h[i] + sign * step, 0, 1)
                new_h = _project_to_feasible(new_h)
                score = _max_overlap_fast(new_h)
                if score < best_score - 1e-10:
                    h = new_h
                    best_score = score
                    break
            else:
                continue
            break
        # If no improvement, try smaller steps
        if time.time() - start_time > time_limit:
            break
    
    return h, best_score


def _generate_spectral_candidates():
    """Generate diverse spectral candidates."""
    candidates = []
    half = N // 2
    
    # Flat spectrum (white noise)
    for _ in range(5):
        amplitudes = np.random.uniform(0.1, 1.0, half + 1)
        phases = np.random.uniform(0, 2 * np.pi, half)
        H = np.zeros(N, dtype=complex)
        H[0] = amplitudes[0]
        for k in range(1, half + 1):
            H[k] = amplitudes[k] * np.exp(1j * phases[k])
            H[N - k] = np.conj(H[k])
        h = np.real(np.fft.ifft(H))
        h = _project_to_feasible(h)
        candidates.append(h)
    
    # Low-pass filtered (smooth)
    for _ in range(5):
        amplitudes = np.zeros(half + 1)
        for k in range(half + 1):
            amplitudes[k] = np.exp(-k / 10.0) * np.random.uniform(0.5, 1.0)
        phases = np.random.uniform(0, 2 * np.pi, half)
        H = np.zeros(N, dtype=complex)
        H[0] = amplitudes[0]
        for k in range(1, half + 1):
            H[k] = amplitudes[k] * np.exp(1j * phases[k])
            H[N - k] = np.conj(H[k])
        h = np.real(np.fft.ifft(H))
        h = _project_to_feasible(h)
        candidates.append(h)
    
    # High-frequency emphasis
    for _ in range(5):
        amplitudes = np.zeros(half + 1)
        for k in range(half + 1):
            amplitudes[k] = (k / half) * np.random.uniform(0.5, 1.0)
        phases = np.random.uniform(0, 2 * np.pi, half)
        H = np.zeros(N, dtype=complex)
        H[0] = amplitudes[0]
        for k in range(1, half + 1):
            H[k] = amplitudes[k] * np.exp(1j * phases[k])
            H[N - k] = np.conj(H[k])
        h = np.real(np.fft.ifft(H))
        h = _project_to_feasible(h)
        candidates.append(h)
    
    # Sinusoidal-based
    for freq in [1, 2, 3, 5, 7, 10]:
        for phase in [0, np.pi/4, np.pi/2]:
            x = np.linspace(0, 2 * np.pi * freq, N, endpoint=False)
            h = 0.5 + 0.4 * np.sin(x + phase)
            h = _project_to_feasible(h)
            candidates.append(h)
    
    return candidates


def construct():
    start_time = time.time()
    deadline = 145.0  # 5 second safety margin
    np.random.seed(42)
    
    best_h = np.full(N, 0.5)
    best_score = _max_overlap_fast(best_h)  # 25.0
    
    # Generate diverse candidates
    candidates = _generate_spectral_candidates()
    
    # Add some random spatial candidates
    for _ in range(50):
        h = np.random.uniform(0, 1, N)
        h = _project_to_feasible(h)
        candidates.append(h)
    
    # Evaluate all candidates
    scored_candidates = []
    for h in candidates:
        if time.time() > deadline - 60:
            break
        score = _max_overlap_fast(h)
        scored_candidates.append((score, h))
    
    scored_candidates.sort(key=lambda x: x[0])
    
    # Spectral optimization from top candidates
    top_candidates = [h for _, h in scored_candidates[:15]]
    
    for h0 in top_candidates:
        if time.time() > deadline - 20:
            break
        remaining = deadline - time.time() - 5
        if remaining < 2:
            break
        h_opt, score_opt, elapsed = _spectral_optimize(
            h0, max_iters=500, time_limit=min(8.0, remaining)
        )
        if score_opt < best_score:
            best_score = score_opt
            best_h = h_opt
    
    # Coordinate descent refinement
    if time.time() < deadline - 10:
        remaining = deadline - time.time() - 5
        if remaining > 2:
            h_cd, score_cd = _coordinate_descent_spectral(
                best_h, max_iters=5000, time_limit=min(15.0, remaining)
            )
            if score_cd < best_score:
                best_score = score_cd
                best_h = h_cd
    
    # Final spectral polish
    if time.time() < deadline - 5:
        remaining = deadline - time.time() - 3
        if remaining > 2:
            h_final, score_final, _ = _spectral_optimize(
                best_h, max_iters=300, time_limit=min(6.0, remaining)
            )
            if score_final < best_score:
                best_score = score_final
                best_h = h_final
    
    # Ensure validity
    best_h = _project_to_feasible(best_h)
    
    return best_h

# EVOLVE-BLOCK-END
