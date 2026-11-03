"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


_FFT_SIZE = 1
while _FFT_SIZE < 2 * N - 1:
    _FFT_SIZE *= 2


def _autoconv_max_abs(x):
    """Max absolute value of the discrete autoconvolution of x."""
    fx = np.fft.rfft(x, _FFT_SIZE)
    conv = np.fft.irfft(fx * fx, _FFT_SIZE)
    return float(np.max(np.abs(conv[:2 * N - 1])))


def _integral(x):
    """Integral of the equal-width step function = sum of the heights."""
    return float(np.sum(x))


def _metric(x):
    """max |autoconv| / (integral)^2.  Lower is better."""
    s = _integral(x)
    if abs(s) < 1e-12:
        return float('inf')
    return _autoconv_max_abs(x) / (s * s)


def _try_near_constant_structured(deadline, rng):
    """Try structured perturbations that maintain high integral while reducing autoconv peaks."""
    best_x = np.ones(N)
    best_val = _metric(best_x)
    
    # Multi-frequency sinusoidal perturbations with small amplitudes
    # These maintain high integral (mean close to 1) while spreading energy
    freqs = [1, 2, 3, 4, 5, 7, 8, 11, 13]
    for f1 in freqs[:5]:
        if time.time() > deadline:
            break
        for f2 in freqs[:5]:
            if time.time() > deadline:
                break
            if f1 == f2:
                continue
            for amp1 in [0.01, 0.02, 0.05]:
                if time.time() > deadline:
                    break
                for amp2 in [0.01, 0.02, 0.05]:
                    if time.time() > deadline:
                        break
                    for phase1 in range(3):
                        if time.time() > deadline:
                            break
                        for phase2 in range(3):
                            if time.time() > deadline:
                                break
                            x = np.ones(N)
                            x += amp1 * np.sin(2 * np.pi * f1 * np.arange(N) / N + 2 * np.pi * phase1 / 3)
                            x += amp2 * np.sin(2 * np.pi * f2 * np.arange(N) / N + 2 * np.pi * phase2 / 3)
                            v = _metric(x)
                            if v < best_val:
                                best_val = v
                                best_x = x.copy()
    
    # Cosine windowed perturbations - taper at edges to reduce boundary effects
    for width in [50, 100, 150, 200, 300]:
        if time.time() > deadline:
            break
        for amp in [0.01, 0.02, 0.05, 0.1]:
            if time.time() > deadline:
                break
            for freq in [1, 2, 3, 5]:
                if time.time() > deadline:
                    break
                center = N // 2
                t = np.arange(N) - center
                window = np.cos(np.pi * t / width) ** 2
                window[window < 0] = 0
                window /= np.max(window) if np.max(window) > 0 else 1
                x = np.ones(N) + amp * window * np.sin(2 * np.pi * freq * np.arange(N) / N)
                v = _metric(x)
                if v < best_val:
                    best_val = v
                    best_x = x.copy()
    
    return best_x, best_val


def _try_fft_guided_optimization(x_init, deadline, rng):
    """Use FFT analysis to guide perturbations that reduce peak autoconvolution."""
    best_x = x_init.copy()
    best_val = _metric(best_x)
    
    # Analyze current autoconvolution to find problematic lags
    fx = np.fft.rfft(best_x, _FFT_SIZE)
    conv = np.fft.irfft(fx * fx, _FFT_SIZE)[:2 * N - 1]
    
    # Find lags with large |autoconv|
    abs_conv = np.abs(conv)
    max_val = np.max(abs_conv)
    
    # Try to reduce peaks by adding small anti-correlated perturbations
    # at positions that contribute to the peak lags
    for iteration in range(50):
        if time.time() > deadline:
            break
        
        # Recompute if needed
        fx = np.fft.rfft(best_x, _FFT_SIZE)
        conv = np.fft.irfft(fx * fx, _FFT_SIZE)[:2 * N - 1]
        abs_conv = np.abs(conv)
        peak_lag = np.argmax(abs_conv)
        peak_val = abs_conv[peak_lag]
        
        if peak_val < max_val * 0.9:
            # Peaks have been reduced, try more iterations with smaller steps
            max_val = peak_val
        
        # Create perturbation that anti-correlates with the peak lag
        # x_i * x_{i+lag} contributes to autoconv at lag
        # We want to reduce sum x_i * x_{i+lag}
        # Perturb x_i by -eta * x_{i+lag} (clipped)
        eta = 0.01 * (1 - iteration / 50.0)
        
        x_new = best_x.copy()
        lag = peak_lag if peak_lag < N else 2 * N - 1 - peak_lag
        if lag > 0 and lag < N:
            # For lag > 0: perturb x_i based on x_{i+lag}
            for i in range(N - lag):
                x_new[i] -= eta * best_x[i + lag]
            # For the wrap-around part, use x_{i-lag}
            for i in range(N - lag, N):
                x_new[i] -= eta * best_x[i - lag]
        
        # Also try the conjugate perturbation
        x_new2 = best_x.copy()
        if lag > 0 and lag < N:
            for i in range(N - lag):
                x_new2[i] += eta * best_x[i + lag]
            for i in range(N - lag, N):
                x_new2[i] += eta * best_x[i - lag]
        
        v1 = _metric(x_new)
        v2 = _metric(x_new2)
        
        if v1 < best_val:
            best_val = v1
            best_x = x_new
        elif v2 < best_val:
            best_val = v2
            best_x = x_new2
    
    return best_x, best_val


def _try_local_search_fast(x, deadline, rng, max_rounds=200000):
    """Fast local search with in-place modifications and early termination."""
    best_x = x.copy()
    best_val = _metric(best_x)
    no_improve = 0
    step_scale = 0.3
    total_steps = 0
    
    for round_idx in range(max_rounds):
        if time.time() > deadline:
            break
        
        if no_improve > 200:
            step_scale *= 0.9
            no_improve = 0
            if step_scale < 0.0005:
                break
        
        k = int(rng.integers(1, 4))
        indices = rng.choice(N, size=k, replace=False)
        
        # Try multiple perturbations per round
        best_delta = 0
        best_idx = None
        for trial in range(3):
            x_test = best_x.copy()
            for idx in indices:
                x_test[idx] += rng.normal(0, step_scale)
            v = _metric(x_test)
            if v < best_val + best_delta:
                best_delta = v - best_val
                best_idx = x_test
        
        if best_idx is not None and best_delta < 0:
            best_val += best_delta
            best_x = best_idx
            no_improve = 0
            total_steps += 1
        else:
            no_improve += 1
    
    return best_x, best_val


def _try_simulated_annealing(x, deadline, rng, max_rounds=40000):
    """Simulated annealing with adaptive step size."""
    best_x = x.copy()
    best_val = _metric(best_x)
    current_x = x.copy()
    current_val = _metric(current_x)
    
    T_start = 0.03
    T_end = 0.0001
    cooling = (T_end / T_start) ** (1.0 / max_rounds)
    T = T_start
    
    for round_idx in range(max_rounds):
        if time.time() > deadline:
            break
        
        k = int(rng.integers(1, 5))
        indices = rng.choice(N, size=k, replace=False)
        x_new = current_x.copy()
        for idx in indices:
            x_new[idx] += rng.normal(0, 0.2)
        
        v_new = _metric(x_new)
        delta = v_new - current_val
        
        if delta < 0 or (T > 1e-10 and rng.random() < np.exp(-delta / T)):
            current_x = x_new
            current_val = v_new
            if v_new < best_val:
                best_val = v_new
                best_x = x_new.copy()
        
        T *= cooling
    
    return best_x, best_val


def _final_polish(x, deadline, rng):
    """High-precision final polish with coordinate descent."""
    best_x = x.copy()
    best_val = _metric(best_x)
    
    # Coordinate descent with adaptive step sizes
    step_sizes = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
    
    for sweep in range(20):
        if time.time() > deadline:
            break
        
        improved = False
        for idx in range(N):
            if time.time() > deadline:
                break
            
            for delta in step_sizes:
                for sign in [1, -1]:
                    x_test = best_x.copy()
                    x_test[idx] += sign * delta
                    v = _metric(x_test)
                    if v < best_val:
                        best_val = v
                        best_x = x_test
                        improved = True
                        break
                if improved:
                    break
        
        if not improved:
            break
    
    return best_x, best_val


def construct():
    deadline = time.time() + 145.0

    best_x = np.ones(N)
    best_val = _metric(best_x)

    rng = np.random.default_rng(2024)

    # Phase 1: Structured near-constant perturbations (most promising)
    if time.time() < deadline - 80:
        x_nc, val_nc = _try_near_constant_structured(deadline - 80, rng)
        if val_nc < best_val:
            best_val = val_nc
            best_x = x_nc

    # Phase 2: FFT-guided optimization
    if time.time() < deadline - 50:
        x_fft, val_fft = _try_fft_guided_optimization(best_x, deadline - 50, rng)
        if val_fft < best_val:
            best_val = val_fft
            best_x = x_fft

    # Phase 3: Simulated annealing
    if time.time() < deadline - 20:
        best_x, best_val = _try_simulated_annealing(best_x, deadline - 20, rng, max_rounds=30000)

    # Phase 4: Fast local search
    if time.time() < deadline - 5:
        best_x, best_val = _try_local_search_fast(best_x, deadline - 5, rng, max_rounds=100000)

    # Phase 5: Final coordinate descent polish
    if time.time() < deadline - 1:
        best_x, best_val = _final_polish(best_x, deadline - 1, rng)

    return best_x
# EVOLVE-BLOCK-END
