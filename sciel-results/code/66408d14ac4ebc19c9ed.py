"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

_DEADLINE = time.time() + 130.0


def _acf_peak(s):
    """Compute the peak sidelobe of the autocorrelation of signed step function s."""
    n_fft = 1 << (2 * N - 1).bit_length()
    F = np.fft.rfft(s, n=n_fft)
    power = F * np.conj(F)
    acf = np.fft.irfft(power, n=n_fft)
    sidelobes = acf[1:N]
    return np.max(np.abs(sidelobes))


def _evaluate(s):
    return _acf_peak(s)


def construct():
    global _DEADLINE
    rng = random.Random(42)
    np_rng = np.random.RandomState(42)
    
    # Initialize with random signs
    s = np.where(np_rng.rand(N) < 0.5, 1.0, -1.0)
    best_s = s.copy()
    best_val = _evaluate(best_s)
    current_val = best_val
    
    # Multi-scale simulated annealing with block flips
    block_sizes = [1, 2, 3, 4, 5, 8, 12, 16, 24, 32, 48, 64, 96, 128, 160, 200, 256, 320]
    bs_weights = np.array([1.0 / bs for bs in block_sizes])
    bs_weights /= bs_weights.sum()
    
    T_start = 0.5
    T_end = 0.0001
    
    n_passes = 16
    iters_per_pass = 3000
    
    for pass_idx in range(n_passes):
        if time.time() > _DEADLINE:
            break
        
        frac = pass_idx / n_passes
        T = T_start * (T_end / T_start) ** frac
        
        for it in range(iters_per_pass):
            if time.time() > _DEADLINE:
                break
            
            # Choose block size
            bs_idx = rng.choices(range(len(block_sizes)), weights=bs_weights, k=1)[0]
            bs = block_sizes[bs_idx]
            
            start = rng.randint(0, N - bs)
            
            # Save and flip
            orig = s[start:start+bs].copy()
            s[start:start+bs] *= -1
            
            new_val = _evaluate(s)
            
            # Metropolis criterion
            if new_val <= current_val:
                current_val = new_val
                if new_val < best_val:
                    best_val = new_val
                    best_s = s.copy()
            else:
                delta = new_val - current_val
                if delta > 0:
                    accept_prob = np.exp(-delta / max(T, 1e-10))
                    if rng.random() < accept_prob:
                        current_val = new_val
                    else:
                        s[start:start+bs] = orig
                else:
                    current_val = new_val
                    if new_val < best_val:
                        best_val = new_val
                        best_s = s.copy()
        
        # Periodic perturbation from best to escape local minima
        if pass_idx % 4 == 3 and time.time() < _DEADLINE:
            s = best_s.copy()
            current_val = best_val
            n_perturb = rng.randint(3, 8)
            for _ in range(n_perturb):
                bs = rng.choice([4, 8, 16, 32, 64])
                start = rng.randint(0, N - bs)
                s[start:start+bs] *= -1
            current_val = _evaluate(s)
    
    return best_s
# EVOLVE-BLOCK-END
