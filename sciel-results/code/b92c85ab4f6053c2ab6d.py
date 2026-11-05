"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _autoconvolution_max_abs(f):
    """Compute max absolute value of autoconvolution (excluding zero-lag self term)."""
    n = len(f)
    # Linear convolution via FFT
    size = 2 * n
    f_pad = np.zeros(size)
    f_pad[:n] = f
    F = np.fft.fft(f_pad, size)
    conv = np.fft.ifft(F * F).real
    # conv[0] is the zero-lag (self) term: sum(f^2)
    # We want max |conv[k]| for k != 0
    if len(conv) > 1:
        return np.max(np.abs(conv[1:]))
    return 0.0


def _compute_score(f):
    """The score is the upper bound on C3 (lower is better)."""
    return _autoconvolution_max_abs(f)


def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s kill
    
    # Strategy: Use Thue-Morse sequence as initialization, then local search with sign flips
    # The Thue-Morse sequence has good autocorrelation properties (low peak sidelobe)
    
    def _thue_morse(n):
        """Generate Thue-Morse sequence of length n: 0,1,1,0,1,0,0,1,..."""
        tm = np.zeros(n, dtype=int)
        for i in range(1, n):
            tm[i] = tm[i >> 1] ^ (i & 1)
        return tm
    
    # Create Thue-Morse sequence mapped to {-1, +1}
    tm = _thue_morse(N)
    f_tm = 1 - 2 * tm  # Map 0->1, 1->-1
    
    best_f = f_tm.copy()
    best_score = _compute_score(best_f)
    
    # Local search: flip individual signs to improve
    improved = True
    while improved and time.time() < deadline - 15:
        improved = False
        for i in range(N):
            if time.time() > deadline - 15:
                break
            old_val = best_f[i]
            best_f[i] = -old_val
            new_score = _compute_score(best_f)
            if new_score < best_score - 1e-12:
                best_score = new_score
                improved = True
            else:
                best_f[i] = old_val
    
    # Try random restarts with different seeds for Thue-Morse variants
    rng = np.random.RandomState(42)
    for restart in range(5):
        if time.time() > deadline - 10:
            break
        
        # Create a perturbed Thue-Morse: flip some random positions
        f_restart = f_tm.copy()
        n_flips = rng.randint(10, 50)
        flip_indices = rng.choice(N, size=n_flips, replace=False)
        f_restart[flip_indices] *= -1
        
        current_score = _compute_score(f_restart)
        
        # Local search from this starting point
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            for i in range(N):
                if time.time() > deadline - 10:
                    break
                old_val = f_restart[i]
                f_restart[i] = -old_val
                new_score = _compute_score(f_restart)
                if new_score < current_score - 1e-12:
                    current_score = new_score
                    improved = True
                else:
                    f_restart[i] = old_val
        
        if current_score < best_score:
            best_score = current_score
            best_f = f_restart.copy()
    
    return best_f
# EVOLVE-BLOCK-END
