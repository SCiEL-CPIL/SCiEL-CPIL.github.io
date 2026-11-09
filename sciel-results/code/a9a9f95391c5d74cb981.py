import numpy as np
import time
import math

N = 400

def compute_autocorrelation(h):
    """
    Compute the autocorrelation of h using FFT.
    Returns the full autocorrelation array of length N (positive lags only).
    """
    n_fft = 1 << (2 * N - 1).bit_length()
    f = np.zeros(n_fft, dtype=np.float64)
    f[:N] = h
    F = np.fft.fft(f)
    R = np.real(np.fft.ifft(F * np.conj(F)))
    return R[:N]

def peak_sidelobe(h):
    """
    Compute the peak sidelobe level (normalized by N).
    h: array of shape (N,) with values +/- 1
    """
    R = compute_autocorrelation(h)
    # R[0] is the DC component (N), R[1:] are sidelobes
    return np.max(np.abs(R[1:])) / N

def sinusoidal_sequence(frequency, phase, length=N):
    """
    Generate a sequence of +/- 1 from a sinusoidal waveform.
    h[i] = sign(sin(frequency * i + phase))
    """
    x = np.arange(length)
    waveform = np.sin(frequency * x + phase)
    # Handle zeros by assigning +1 (arbitrary choice)
    h = np.sign(waveform)
    h[h == 0] = 1.0
    return h

def objective(params):
    """
    Objective function for L-BFGS optimization.
    params: [frequency, phase]
    Returns: peak sidelobe level
    """
    freq = params[0]
    phase = params[1]
    h = sinusoidal_sequence(freq, phase)
    return peak_sidelobe(h)

def find_best_sinusoidal(deadline):
    """
    Use L-BFGS to find the best sinusoidal parameters.
    """
    rng = np.random.default_rng(42)
    
    best_h = None
    best_score = float('inf')
    
    # Try multiple random starting points
    n_starts = 20
    
    for i in range(n_starts):
        if time.time() > deadline:
            break
        
        # Random initial frequency and phase
        # Frequency should be in a reasonable range to get ~N/2 sign changes
        # For N=400, we want roughly 200 sign changes, so freq ~ pi/2 to pi
        freq_init = rng.uniform(0.5, 3.0)
        phase_init = rng.uniform(0, 2 * np.pi)
        
        x0 = np.array([freq_init, phase_init])
        
        # Run L-BFGS
        from scipy.optimize import minimize
        result = minimize(
            objective,
            x0,
            method='L-BFGS-B',
            options={'maxiter': 50, 'maxfun': 200}
        )
        
        if result.success or result.fun < best_score:
            h = sinusoidal_sequence(result.x[0], result.x[1])
            score = result.fun
            if score < best_score:
                best_score = score
                best_h = h.copy()
        
        # Also check the initial point
        h_init = sinusoidal_sequence(freq_init, phase_init)
        score_init = peak_sidelobe(h_init)
        if score_init < best_score:
            best_score = score_init
            best_h = h_init.copy()
    
    return best_h, best_score

def local_search_from_h(h, deadline, rng):
    """
    Perform local search starting from h.
    """
    current_score = peak_sidelobe(h)
    best_h = h.copy()
    best_score = current_score
    
    while time.time() < deadline:
        # Find best single flip using batch FFT
        n_fft = 1 << (2 * N - 1).bit_length()
        
        # Batch all flips
        h_batch = np.tile(h, (N, 1))
        for i in range(N):
            h_batch[i, i] *= -1
        
        F_batch = np.fft.fft(h_batch, axis=1)
        R_batch = np.real(np.fft.ifft(F_batch * np.conj(F_batch), axis=1))
        R_batch = R_batch[:, :N]
        R_batch[:, 0] = 0
        
        scores = np.max(np.abs(R_batch[:, 1:]), axis=1) / N
        
        min_idx = np.argmin(scores)
        min_score = scores[min_idx]
        
        if min_score < current_score:
            h[min_idx] *= -1
            current_score = min_score
            
            if current_score < best_score:
                best_score = current_score
                best_h = h.copy()
        else:
            # Local minimum reached, perturb
            num_flips = rng.integers(5, 11)
            flip_indices = rng.choice(N, size=num_flips, replace=False)
            h[flip_indices] *= -1
            current_score = peak_sidelobe(h)
    
    return best_h, best_score

def simulated_annealing(h, deadline, rng):
    """
    Optimize h using Simulated Annealing.
    """
    current_score = peak_sidelobe(h)
    best_h = h.copy()
    best_score = current_score
    
    # Initial temperature
    T = 0.5
    # Cooling rate
    alpha = 0.999
    # Minimum temperature
    T_min = 1e-5
    
    # We will perform steps until deadline is reached
    # Each step involves a random flip and an FFT.
    # FFT size is 1024. Cost is approx 1024*log2(1024) ~ 10k ops.
    # Python overhead is significant. Let's estimate ~1000 steps/sec.
    # 115 seconds -> ~115,000 steps.
    
    while time.time() < deadline and T > T_min:
        # Randomly select an index to flip
        i = rng.integers(0, N)
        h[i] *= -1
        
        new_score = peak_sidelobe(h)
        
        # Acceptance criterion
        delta = new_score - current_score
        if delta < 0 or rng.random() < np.exp(-delta / T):
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_h = h.copy()
        else:
            # Reject: revert
            h[i] *= -1
            
        # Cool down
        T *= alpha
        
    return best_h, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    rng = np.random.default_rng(42)
    
    candidates = []
    best_h = None
    best_score = float('inf')
    
    # 1. Sinusoidal parameterization with L-BFGS
    h_sin, score_sin = find_best_sinusoidal(deadline - 30)  # Leave 30s for local search
    if h_sin is not None and score_sin < best_score:
        best_score = score_sin
        best_h = h_sin.copy()
    
    # 2. Thue-Morse
    def generate_thue_morse(length):
        seq = np.array([0])
        while len(seq) < length:
            inv = 1 - seq
            seq = np.concatenate([seq, inv])
        return (1 - 2 * seq[:length]).astype(np.float64)
    
    h_tm = generate_thue_morse(N)
    candidates.append(h_tm)
    
    # 3. Random
    h_rand = rng.choice([-1.0, 1.0], N)
    candidates.append(h_rand)
    
    # 4. M-Sequence
    def generate_mseq_length(m, taps, length):
        state = 1
        seq = []
        for i in range(length):
            seq.append(1.0 if (state & 1) else -1.0)
            fb = 0
            for t in taps:
                fb ^= (state >> t) & 1
            state = (state >> 1) | (fb << (m - 1))
        return np.array(seq)
    
    h_mseq = generate_mseq_length(9, [0, 4], 511)
    h_mseq = h_mseq[:N]
    candidates.append(h_mseq)
    
    # 5. Add a second random candidate with different seed
    rng2 = np.random.default_rng(123)
    h_rand2 = rng2.choice([-1.0, 1.0], N)
    candidates.append(h_rand2)
    
    # Local search on each candidate
    remaining_time = deadline - time.time()
    if remaining_time > 5:
        time_per_candidate = remaining_time / len(candidates)
        
        for idx, h in enumerate(candidates):
            if time.time() > deadline:
                break
            
            h_work = h.copy()
            local_deadline = time.time() + min(time_per_candidate, deadline - time.time())
            
            # Alternate between local search and simulated annealing
            if idx % 2 == 0:
                h_opt, score = local_search_from_h(h_work, local_deadline, rng)
            else:
                h_opt, score = simulated_annealing(h_work, local_deadline, rng)
            
            if score < best_score:
                best_score = score
                best_h = h_opt
    
    if best_h is None:
        return np.ones(N)
    
    return best_h
