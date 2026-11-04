"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _autoconvolution_max_abs(f):
    """Compute max absolute value of autoconvolution (excluding zero-lag self term)."""
    n = len(f)
    size = 2 * n
    f_pad = np.zeros(size)
    f_pad[:n] = f
    F = np.fft.fft(f_pad, size)
    conv = np.fft.ifft(F * F).real
    if len(conv) > 1:
        return np.max(np.abs(conv[1:]))
    return 0.0


def _compute_score(f):
    """The score is the upper bound on C3 (lower is better)."""
    return _autoconvolution_max_abs(f)


def _harmonics_to_binary(freqs, amps, phases):
    """
    Generate a signed binary sequence from continuous harmonic parameters.
    freqs: array of angular frequencies (radians per unit interval)
    amps: array of amplitudes
    phases: array of phases
    Returns: array of +1/-1 of length N
    """
    x = np.linspace(-0.25, 0.25, N, endpoint=False)
    signal = np.zeros(N)
    for i in range(len(freqs)):
        signal += amps[i] * np.cos(freqs[i] * x + phases[i])
    return np.sign(signal)


def _score_from_params(freqs, amps, phases):
    """Compute score from harmonic parameters."""
    f = _harmonics_to_binary(freqs, amps, phases)
    return _compute_score(f)


def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s kill
    
    # Strategy: harmonic synthesis with irrational frequency ratios
    # Represent f as sign of sum of cosines with quasi-periodic structure
    # Optimize continuous parameters: frequencies, amplitudes, phases
    
    np.random.seed(12345)
    
    best_score = np.inf
    best_f = np.ones(N)  # Default: all +1
    
    # Number of harmonics to use
    n_harmonics = 5
    
    # Base frequencies: use irrational multiples to avoid periodicity
    # Golden ratio conjugate and related values
    phi = (1 + np.sqrt(5)) / 2
    base_freqs = np.array([1.0, 1.0/phi, 1.0/phi**2, 2.0/phi, 1.0/phi**3])
    
    # Try multiple random initializations
    n_restarts = 20
    
    for restart in range(n_restarts):
        if time.time() > deadline - 30:
            break
        
        # Random initialization of parameters
        # Frequencies: perturb base frequencies slightly
        freqs = base_freqs * (1 + 0.1 * np.random.randn(n_harmonics))
        # Ensure frequencies are positive
        freqs = np.abs(freqs)
        
        # Amplitudes: random positive values
        amps = np.abs(np.random.randn(n_harmonics)) + 0.1
        
        # Phases: random in [0, 2*pi)
        phases = 2 * np.pi * np.random.rand(n_harmonics)
        
        current_score = _score_from_params(freqs, amps, phases)
        
        if current_score < best_score:
            best_score = current_score
            best_f = _harmonics_to_binary(freqs, amps, phases).copy()
        
        # Local search in parameter space
        step_sizes = {
            'freqs': 0.05,
            'amps': 0.1,
            'phases': 0.1
        }
        
        improved = True
        while improved and time.time() < deadline - 30:
            improved = False
            
            # Try perturbing each parameter
            for i in range(n_harmonics):
                if time.time() > deadline - 30:
                    break
                
                # Perturb frequency
                old_freq = freqs[i]
                freqs[i] += step_sizes['freqs'] * np.random.randn()
                freqs[i] = max(0.01, freqs[i])  # Keep positive
                new_score = _score_from_params(freqs, amps, phases)
                if new_score < current_score - 1e-10:
                    current_score = new_score
                    improved = True
                    if current_score < best_score:
                        best_score = current_score
                        best_f = _harmonics_to_binary(freqs, amps, phases).copy()
                else:
                    freqs[i] = old_freq
                
                # Perturb amplitude
                old_amp = amps[i]
                amps[i] += step_sizes['amps'] * np.random.randn()
                amps[i] = max(0.01, amps[i])  # Keep positive
                new_score = _score_from_params(freqs, amps, phases)
                if new_score < current_score - 1e-10:
                    current_score = new_score
                    improved = True
                    if current_score < best_score:
                        best_score = current_score
                        best_f = _harmonics_to_binary(freqs, amps, phases).copy()
                else:
                    amps[i] = old_amp
                
                # Perturb phase
                old_phase = phases[i]
                phases[i] += step_sizes['phases'] * np.random.randn()
                new_score = _score_from_params(freqs, amps, phases)
                if new_score < current_score - 1e-10:
                    current_score = new_score
                    improved = True
                    if current_score < best_score:
                        best_score = current_score
                        best_f = _harmonics_to_binary(freqs, amps, phases).copy()
                else:
                    phases[i] = old_phase
        
        # Adaptive step size reduction
        step_sizes['freqs'] *= 0.7
        step_sizes['amps'] *= 0.7
        step_sizes['phases'] *= 0.7
    
    # Final polish: try flipping individual bits of best solution
    if time.time() < deadline - 10:
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            for idx in range(N):
                if time.time() > deadline - 10:
                    break
                old_val = best_f[idx]
                best_f[idx] = -old_val
                s = _compute_score(best_f)
                if s < best_score - 1e-12:
                    best_score = s
                    improved = True
                else:
                    best_f[idx] = old_val
    
    return best_f
# EVOLVE-BLOCK-END
