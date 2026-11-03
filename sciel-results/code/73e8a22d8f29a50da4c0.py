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


def _generate_m_sequence(n, poly, init):
    """Generate an m-sequence of length n using a linear feedback shift register."""
    seq = []
    state = init
    for _ in range(n):
        seq.append(1.0 if (state & 1) == 0 else -1.0)
        feedback = 0
        for tap in poly:
            feedback ^= (state >> tap) & 1
        state = (state >> 1) | (feedback << (len(poly) - 1))
    return np.array(seq[:n])


def _try_m_sequences(deadline, rng):
    """Try various m-sequence based patterns."""
    best_x = np.ones(N)
    best_val = _metric(best_x)
    
    polys = [
        [1, 0], [2, 0], [3, 1], [4, 1], [5, 2],
        [6, 1], [7, 3], [8, 4], [9, 5], [10, 7],
    ]
    
    for poly in polys:
        if time.time() > deadline:
            break
        for init in range(min(10, (1 << len(poly)) - 1)):
            if time.time() > deadline:
                break
            seq = _generate_m_sequence(N, poly, init + 1)
            v = _metric(seq)
            if v < best_val:
                best_val = v
                best_x = seq.copy()
            seq_neg = -seq
            v = _metric(seq_neg)
            if v < best_val:
                best_val = v
                best_x = seq_neg.copy()
    
    return best_x, best_val


def _try_near_constant(deadline, rng):
    """Try sequences close to constant with small structured perturbations."""
    best_x = np.ones(N)
    best_val = _metric(best_x)
    
    # Small sinusoidal perturbations
    for freq in range(1, 20):
        if time.time() > deadline:
            break
        for amp in [0.01, 0.05, 0.1, 0.2, 0.5]:
            if time.time() > deadline:
                break
            for phase in range(freq):
                x = np.ones(N) + amp * np.sin(2 * np.pi * freq * np.arange(N) / N + 2 * np.pi * phase / freq)
                v = _metric(x)
                if v < best_val:
                    best_val = v
                    best_x = x.copy()
    
    # Small random perturbations on constant
    for amp in [0.01, 0.05, 0.1, 0.2]:
        if time.time() > deadline:
            break
        for _ in range(20):
            if time.time() > deadline:
                break
            x = np.ones(N) + amp * rng.standard_normal(N)
            v = _metric(x)
            if v < best_val:
                best_val = v
                best_x = x.copy()
    
    return best_x, best_val


def _try_block_patterns(deadline, rng):
    """Try patterns with blocks of +1 and -1 of various sizes."""
    best_x = np.ones(N)
    best_val = _metric(best_x)
    
    for block_size in [2, 3, 4, 5, 7, 8, 10, 13, 16, 20, 25, 32, 40, 50, 64, 80, 100, 128, 160, 200]:
        if time.time() > deadline:
            break
        if block_size > N:
            continue
        
        n_blocks = N // block_size
        remainder = N % block_size
        
        x = np.zeros(N)
        for i in range(n_blocks):
            start = i * block_size
            end = start + block_size
            x[start:end] = 1.0 if i % 2 == 0 else -1.0
        if remainder > 0:
            x[n_blocks * block_size:] = 1.0 if (n_blocks % 2 == 0) else -1.0
        
        v = _metric(x)
        if v < best_val:
            best_val = v
            best_x = x.copy()
        
        for _ in range(3):
            if time.time() > deadline:
                break
            x = np.zeros(N)
            for i in range(n_blocks):
                start = i * block_size
                end = start + block_size
                x[start:end] = rng.choice([-1.0, 1.0])
            if remainder > 0:
                x[n_blocks * block_size:] = rng.choice([-1.0, 1.0])
            v = _metric(x)
            if v < best_val:
                best_val = v
                best_x = x.copy()
    
    return best_x, best_val


def _try_gaussian_smoothed(deadline, rng):
    """Try sequences that are smoothed versions of random patterns."""
    best_x = np.ones(N)
    best_val = _metric(best_x)
    
    for sigma in [2, 5, 10, 20, 40]:
        if time.time() > deadline:
            break
        for _ in range(8):
            if time.time() > deadline:
                break
            x_raw = rng.choice([-1.0, 1.0], size=N)
            from scipy.ndimage import gaussian_filter1d
            x_smooth = gaussian_filter1d(x_raw, sigma=sigma)
            rms = np.sqrt(np.mean(x_smooth**2))
            if rms > 1e-12:
                x_smooth = x_smooth / rms
            v = _metric(x_smooth)
            if v < best_val:
                best_val = v
                best_x = x_smooth.copy()
    
    return best_x, best_val


def _try_local_search(x, deadline, rng, max_rounds=100000):
    """Local search with adaptive perturbation magnitude."""
    best_x = x.copy()
    best_val = _metric(best_x)
    no_improve = 0
    step_scale = 0.5
    
    for round_idx in range(max_rounds):
        if time.time() > deadline:
            break
        
        if no_improve > 500:
            step_scale *= 0.85
            no_improve = 0
            if step_scale < 0.001:
                break
        
        k = rng.integers(1, 4)
        indices = rng.choice(N, size=k, replace=False)
        x_flipped = best_x.copy()
        
        for idx in indices:
            delta = rng.normal(0, step_scale)
            x_flipped[idx] += delta
        
        v = _metric(x_flipped)
        if v < best_val:
            best_val = v
            best_x = x_flipped
            no_improve = 0
        else:
            no_improve += 1
    
    return best_x, best_val


def _try_simulated_annealing(x, deadline, rng, max_rounds=50000):
    """Simulated annealing."""
    best_x = x.copy()
    best_val = _metric(best_x)
    current_x = x.copy()
    current_val = _metric(current_x)
    
    T_start = 0.05
    T_end = 0.0005
    cooling = (T_end / T_start) ** (1.0 / max_rounds)
    T = T_start
    
    for round_idx in range(max_rounds):
        if time.time() > deadline:
            break
        
        k = rng.integers(1, 4)
        indices = rng.choice(N, size=k, replace=False)
        x_new = current_x.copy()
        for idx in indices:
            x_new[idx] += rng.normal(0, 0.3)
        
        v_new = _metric(x_new)
        delta = v_new - current_val
        
        if delta < 0 or rng.random() < np.exp(-delta / T):
            current_x = x_new
            current_val = v_new
            if v_new < best_val:
                best_val = v_new
                best_x = x_new.copy()
        
        T *= cooling
    
    return best_x, best_val


def _final_polish(x, deadline, rng):
    """High-precision final polish with very small steps."""
    best_x = x.copy()
    best_val = _metric(best_x)
    
    # Try single-element adjustments with very small steps
    for _ in range(200000):
        if time.time() > deadline:
            break
        
        idx = int(rng.integers(0, N))
        for delta in [0.001, -0.001, 0.01, -0.01, 0.05, -0.05]:
            x_test = best_x.copy()
            x_test[idx] += delta
            v = _metric(x_test)
            if v < best_val:
                best_val = v
                best_x = x_test
                break
    
    return best_x, best_val


def construct():
    deadline = time.time() + 130.0

    best_x = np.ones(N)
    best_val = _metric(best_x)

    rng = np.random.default_rng(2024)

    # Phase 1: Near-constant perturbations (most promising)
    if time.time() < deadline - 90:
        x_nc, val_nc = _try_near_constant(deadline - 90, rng)
        if val_nc < best_val:
            best_val = val_nc
            best_x = x_nc

    # Phase 2: M-sequences
    if time.time() < deadline - 70:
        x_m, val_m = _try_m_sequences(deadline - 70, rng)
        if val_m < best_val:
            best_val = val_m
            best_x = x_m

    # Phase 3: Block patterns
    if time.time() < deadline - 50:
        x_block, val_block = _try_block_patterns(deadline - 50, rng)
        if val_block < best_val:
            best_val = val_block
            best_x = x_block

    # Phase 4: Gaussian smoothed
    if time.time() < deadline - 35:
        x_gauss, val_gauss = _try_gaussian_smoothed(deadline - 35, rng)
        if val_gauss < best_val:
            best_val = val_gauss
            best_x = x_gauss

    # Phase 5: Random ±1
    if time.time() < deadline - 20:
        for _ in range(40):
            if time.time() > deadline - 20:
                break
            cand = rng.choice([-1.0, 1.0], size=N)
            v = _metric(cand)
            if v < best_val:
                best_val = v
                best_x = cand.copy()

    # Phase 6: Simulated annealing
    if time.time() < deadline - 10:
        best_x, best_val = _try_simulated_annealing(best_x, deadline - 10, rng, max_rounds=30000)

    # Phase 7: Local search
    if time.time() < deadline - 4:
        best_x, best_val = _try_local_search(best_x, deadline - 4, rng, max_rounds=50000)

    # Phase 8: Final high-precision polish
    if time.time() < deadline - 1:
        best_x, best_val = _final_polish(best_x, deadline - 1, rng)

    return best_x
# EVOLVE-BLOCK-END
