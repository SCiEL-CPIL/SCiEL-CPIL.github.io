"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import scipy.optimize

N = 400

# EVOLVE-BLOCK-START

def _autoconv_full(x):
    """Full linear autoconvolution of x (length 2N-1)."""
    n = len(x)
    size = 1 << (2 * n - 1).bit_length()
    F = np.fft.rfft(x, n=size)
    conv = np.fft.irfft(F * F, n=size)
    return conv[:2 * n - 1]


def _score_offdiag(x):
    """
    Compute the metric: max(|autoconv|) for k != 0 / (sum(x))^2
    We want to minimize this.
    """
    integral = np.sum(x)
    if abs(integral) < 1e-12:
        return 1e12
    conv = _autoconv_full(x)
    if len(conv) <= 1:
        return 0.0
    max_abs_offdiag = np.max(np.abs(conv[1:]))
    return max_abs_offdiag / (integral * integral)


def _score_std(x):
    """
    Compute the metric: max(|autoconv|) for all k / (sum(x))^2
    """
    integral = np.sum(x)
    if abs(integral) < 1e-12:
        return 1e12
    conv = _autoconv_full(x)
    max_abs = np.max(np.abs(conv))
    return max_abs / (integral * integral)


def _spectral_phase_optimize(x_init, deadline, rng):
    """
    Optimize the phase of the Fourier transform to maximize spectral flatness.
    The idea: if the magnitude spectrum is flat, the autoconvolution will be
    more evenly distributed, reducing peaks.
    
    We parameterize the signal as:
    x[n] = sum_k A[k] * exp(i * (phi[k] + 2*pi*k*n/N))
    
    where A[k] is the magnitude (fixed from initial signal) and phi[k] is the
    phase to be optimized.
    """
    n = len(x_init)
    size = 1 << (2 * n - 1).bit_length()
    
    # Get the FFT of the initial signal
    X = np.fft.rfft(x_init, n=size)
    
    # Extract magnitudes (keep them fixed)
    magnitudes = np.abs(X)
    
    # Initialize phases randomly
    num_freqs = len(X)
    phases_init = rng.uniform(0, 2 * np.pi, size=num_freqs)
    
    # Objective: minimize the peak-to-mean ratio of autoconvolution
    def objective(phases):
        # Reconstruct signal from magnitudes and phases
        X_new = magnitudes * np.exp(1j * phases)
        x_new = np.fft.irfft(X_new, n=size)[:n]
        
        # Compute autoconvolution
        F_new = np.fft.rfft(x_new, n=size)
        ac = np.fft.irfft(F_new * F_new, n=size)[:2 * n - 1]
        
        # Peak-to-mean ratio (off-diagonal)
        if len(ac) > 1:
            peak = np.max(np.abs(ac[1:]))
            mean = np.mean(np.abs(ac[1:]))
        else:
            peak = np.max(np.abs(ac))
            mean = np.mean(np.abs(ac))
        
        if mean < 1e-15:
            return 1e10
        return peak / mean
    
    # Use differential evolution to optimize phases
    bounds = [(0, 2 * np.pi)] * num_freqs
    
    try:
        result = scipy.optimize.differential_evolution(
            objective,
            bounds=bounds,
            maxiter=30,
            popsize=10,
            tol=1e-4,
            seed=42,
            polish=True,
            updating='immediate'
        )
        
        # Reconstruct optimized signal
        X_opt = magnitudes * np.exp(1j * result.x)
        x_opt = np.fft.irfft(X_opt, n=size)[:n]
        
        # Make it real and clean
        x_opt = np.real(x_opt)
        
        # Score the result
        score_opt = _score_offdiag(x_opt)
        return x_opt, score_opt
        
    except Exception:
        pass
    
    return x_init, _score_offdiag(x_init)


def _phase_local_search(x, current_score, best_score, best_h, rng, deadline, n):
    """
    Local search in the phase domain: perturb phases and accept improvements.
    """
    size = 1 << (2 * n - 1).bit_length()
    X = np.fft.rfft(x, n=size)
    magnitudes = np.abs(X)
    phases = np.angle(X)
    
    no_improve = 0
    max_no_improve = 200
    
    while time.time() < deadline:
        if no_improve > max_no_improve:
            break
        
        # Perturb a random subset of phases
        num_perturb = rng.integers(1, max(2, n // 10))
        idx = rng.choice(len(phases), size=num_perturb, replace=False)
        delta = rng.uniform(-0.5, 0.5, size=num_perturb)
        
        old_phases = phases[idx].copy()
        phases[idx] += delta
        
        # Reconstruct
        X_new = magnitudes * np.exp(1j * phases)
        x_new = np.fft.irfft(X_new, n=size)[:n]
        x_new = np.real(x_new)
        
        # Score
        s = np.sum(x_new)
        if abs(s) < 1e-12:
            new_score = np.inf
        else:
            F_new = np.fft.rfft(x_new, n=size)
            ac = np.fft.irfft(F_new * F_new, n=size)[:2 * n - 1]
            if len(ac) > 1:
                max_abs = np.max(np.abs(ac[1:]))
            else:
                max_abs = np.max(np.abs(ac))
            new_score = max_abs / (s * s)
        
        if new_score <= current_score:
            current_score = new_score
            no_improve = 0
            x = x_new
            if new_score < best_score:
                best_score = new_score
                best_h = x.copy()
        else:
            phases[idx] = old_phases
            no_improve += 1
    
    return x, current_score, best_score, best_h


def _m_sequence(n, k):
    """Generate an m-sequence of length n using LFSR with k stages."""
    polys = {
        4: [3, 0],
        5: [4, 3, 0],
        6: [5, 3, 0],
        7: [6, 5, 3, 0],
        8: [7, 5, 3, 2, 0],
    }
    if k not in polys:
        return np.sign(np.sin(np.arange(n) * 2 * np.pi / n * 3))[:n]
    
    taps = polys[k]
    state = [0] * k
    state[0] = 1
    seq = []
    for _ in range(n):
        fb = 0
        for t in taps:
            fb ^= state[t]
        for i in range(k - 1, 0, -1):
            state[i] = state[i - 1]
        state[0] = fb
        seq.append(1.0 if state[0] else -1.0)
    return np.array(seq, dtype=np.float64)


def _block_pattern(n, block_size, neg_frac):
    """Generate block pattern with alternating +1/-1 blocks."""
    h = np.zeros(n, dtype=np.float64)
    for i in range(n):
        block_idx = i // block_size
        if block_idx % 2 == 0:
            pos_in_block = i % block_size
            n_neg = int(neg_frac * block_size)
            h[i] = -1.0 if pos_in_block < n_neg else 1.0
        else:
            pos_in_block = i % block_size
            n_neg = int(neg_frac * block_size)
            h[i] = 1.0 if pos_in_block < n_neg else -1.0
    return h


def _chunk_flip_search(current, current_score, best_score, best_h, rng, deadline, n, use_offdiag=True):
    """
    Chunk-based local search with peak-targeting.
    """
    size = 1 << (2 * n - 1).bit_length()
    
    ac = _autoconv_full(current)
    if use_offdiag and len(ac) > 1:
        peak_idx = np.argmax(np.abs(ac[1:])) + 1
    else:
        peak_idx = np.argmax(np.abs(ac))
    
    no_improve_count = 0
    max_no_improve = 5000
    
    while time.time() < deadline:
        if no_improve_count > max_no_improve:
            break
        
        r = rng.random()
        if r < 0.4:
            chunk_sz = 1
        elif r < 0.6:
            chunk_sz = rng.choice([2, 3, 4])
        elif r < 0.8:
            chunk_sz = rng.choice([5, 8, 12])
        else:
            chunk_sz = rng.choice([16, 24, 32])
        
        if rng.random() < 0.6:
            target = peak_idx % n
            spread = max(chunk_sz, n // 8)
            start = int(rng.integers(max(0, target - spread), min(n - chunk_sz, target + spread) + 1))
        else:
            start = int(rng.integers(0, max(1, n - chunk_sz + 1)))
        
        start = max(0, min(start, n - chunk_sz))
        end = start + chunk_sz
        
        old_vals = current[start:end].copy()
        current[start:end] = -old_vals
        
        s = np.sum(current)
        if abs(s) < 1e-12:
            new_score = np.inf
        else:
            H = np.fft.rfft(current, size)
            ac_new = np.fft.irfft(H * H, size)[:2 * n - 1]
            if use_offdiag and len(ac_new) > 1:
                max_abs = np.max(np.abs(ac_new[1:]))
            else:
                max_abs = np.max(np.abs(ac_new))
            new_score = max_abs / (s * s)
        
        if new_score <= current_score:
            current_score = new_score
            no_improve_count = 0
            if new_score < best_score:
                best_score = new_score
                best_h = current.copy()
                if use_offdiag and len(ac_new) > 1:
                    peak_idx = np.argmax(np.abs(ac_new[1:])) + 1
                else:
                    peak_idx = np.argmax(np.abs(ac_new))
        else:
            current[start:end] = old_vals
            no_improve_count += 1
    
    return current, current_score, best_score, best_h


def construct():
    t0 = time.time()
    deadline = t0 + 130.0
    
    rng = np.random.default_rng(42)
    
    # Diverse initializations
    candidates = []
    
    # 1. Hanning window (good off-diagonal behavior)
    candidates.append(np.hanning(N))
    
    # 2. Hamming window
    candidates.append(np.hamming(N))
    
    # 3. M-sequences
    for k in [6, 7, 8]:
        candidates.append(_m_sequence(N, k))
    
    # 4. Block patterns
    for bs in [8, 16, 32, 64]:
        for nf in [0.0, 0.2, 0.3, 0.5]:
            candidates.append(_block_pattern(N, bs, nf))
    
    # 5. Random with varying bias
    for bias in [0.0, 0.1, 0.2, 0.3]:
        p_pos = 0.5 + bias / 2
        h = rng.choice([-1.0, 1.0], size=N, p=[1 - p_pos, p_pos])
        candidates.append(h)
    
    # 6. Sinusoidal modulation
    for freq in [1, 2, 3, 5]:
        h = np.sign(np.sin(2 * np.pi * freq * np.arange(N) / N))
        candidates.append(h)
    
    # 7. Walsh-Hadamard-like patterns
    for level in range(4):
        h = np.ones(N)
        for i in range(N):
            val = 1.0
            for bit in range(level + 1):
                if (i >> bit) & 1:
                    val *= -1.0
            h[i] = val
        candidates.append(h)
    
    # 8. Gaussian window
    x = np.arange(N)
    sigma = N / 4
    candidates.append(np.exp(-0.5 * ((x - N/2) / sigma) ** 2))
    
    # Evaluate all candidates using off-diagonal metric
    best_score = np.inf
    best_h = np.ones(N)
    
    for h in candidates:
        if time.time() > deadline - 60:
            break
        s = _score_offdiag(h)
        if s < best_score:
            best_score = s
            best_h = h.copy()
    
    n = N
    
    # Phase 1: Spectral phase optimization (NEW - structural change)
    if time.time() < deadline - 50:
        spectral_deadline = time.time() + 30
        best_h, best_score = _spectral_phase_optimize(best_h, spectral_deadline, rng)
    
    # Phase 2: Phase local search (NEW - structural change)
    if time.time() < deadline - 20:
        phase2_deadline = min(t0 + 90, deadline - 15)
        best_h, best_score, best_score, best_h = _phase_local_search(
            best_h.copy(), best_score, best_score, best_h.copy(), rng, phase2_deadline, n
        )
    
    # Phase 3: Final refinement with chunk flips
    if time.time() < deadline - 5:
        best_h, best_score, best_score, best_h = _chunk_flip_search(
            best_h.copy(), best_score, best_score, best_h.copy(), rng, deadline - 2, n, use_offdiag=True
        )
    
    return best_h

# EVOLVE-BLOCK-END
