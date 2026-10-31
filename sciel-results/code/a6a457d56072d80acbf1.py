import numpy as np
import time

N = 400

def _autoconv(x):
    """Compute full autoconvolution of x via FFT."""
    n = len(x)
    size = 1
    while size < 2 * n - 1:
        size *= 2
    X = np.fft.rfft(x, size)
    ac = np.fft.irfft(X * np.conj(X), size)[:2 * n - 1]
    return ac

def _score(x):
    """Compute max|autoconv| / (sum(x))^2. Lower is better."""
    s = np.sum(x)
    if abs(s) < 1e-12:
        return float('inf')
    ac = _autoconv(x)
    max_ac = np.max(np.abs(ac))
    return max_ac / (s * s)

def _van_der_corput(n, base=2):
    """Generate first n terms of Van der Corput sequence in [0,1)."""
    result = np.zeros(n)
    for i in range(n):
        v = 0.0
        f = 1.0 / base
        num = i
        while num > 0:
            v += f * (num % base)
            num //= base
            f /= base
        result[i] = v
    return result

def _kronecker(n, alpha):
    """Generate first n terms of Kronecker sequence {i*alpha} mod 1."""
    return (np.arange(n) * alpha) % 1.0

def _from_fractional_parts(fracs, n, num_levels=4):
    """Map fractional parts to discrete signed levels."""
    # Quantize into num_levels bins, map to signed values
    bins = np.digitize(fracs, np.linspace(0, 1, num_levels + 1)[1:-1])
    # Map bins 0..num_levels-1 to signed values
    # Center at 0: values like -1.5, -0.5, 0.5, 1.5 for 4 levels
    levels = np.linspace(-1, 1, num_levels)
    return levels[bins]

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin

    # Baseline: All ones
    x_best = np.ones(N, dtype=np.float64)
    best_score = _score(x_best)
    
    # Generate various number-theoretic sequences
    candidates = []
    
    # Van der Corput with different bases
    for base in [2, 3, 5, 7]:
        vdc = _van_der_corput(N, base)
        for num_levels in [2, 3, 4, 5, 8]:
            x = _from_fractional_parts(vdc, N, num_levels)
            candidates.append(x)
    
    # Kronecker sequences with various irrational alpha
    alphas = [
        (1 + np.sqrt(5)) / 2,  # golden ratio
        np.sqrt(2),
        np.sqrt(3),
        np.pi,
        np.e,
        1.0 / ((1 + np.sqrt(5)) / 2),  # inverse golden ratio
        0.38196601125,  # 1/phi
        0.70710678118,  # 1/sqrt(2)
        0.61803398875,  # phi - 1
        0.41421356237,  # sqrt(2) - 1
    ]
    
    for alpha in alphas:
        kr = _kronecker(N, alpha)
        for num_levels in [2, 3, 4, 5, 8]:
            x = _from_fractional_parts(kr, N, num_levels)
            candidates.append(x)
    
    # Also try some structured patterns
    # Alternating blocks
    for block_size in [2, 3, 4, 5, 7, 10, 13, 17, 20]:
        pattern = np.tile(np.array([1.0, -1.0][:block_size % 2 + 1]), N // (block_size % 2 + 1) + 1)[:N]
        candidates.append(pattern)
    
    # Evaluate all candidates
    for x in candidates:
        if time.time() > deadline - 30:
            break
        s = np.sum(x)
        if abs(s) < 1e-12:
            continue
        ac = _autoconv(x)
        max_ac = np.max(np.abs(ac))
        score = max_ac / (s * s)
        if score < best_score:
            best_score = score
            x_best = x.copy()
    
    # Local search: sign flips on the best candidate
    if time.time() < deadline - 10:
        x = x_best.copy()
        rng = np.random.default_rng(123)
        
        # Precompute FFT size
        fft_size = 1
        while fft_size < 2 * N - 1:
            fft_size *= 2
        
        no_improve = 0
        max_no_improve = 50
        step_size = 0.01
        
        for _iter in range(5000):
            if time.time() > deadline - 5:
                break
            
            s = np.sum(x)
            if abs(s) < 1e-12:
                break
            ac = _autoconv(x)
            max_ac = np.max(np.abs(ac))
            cur_score = max_ac / (s * s)
            
            if cur_score < best_score:
                best_score = cur_score
                x_best = x.copy()
                no_improve = 0
            else:
                no_improve += 1
                if no_improve > max_no_improve:
                    break
            
            # Try flipping a random position
            idx = rng.integers(N)
            x_trial = x.copy()
            x_trial[idx] = -x_trial[idx]
            
            s_trial = np.sum(x_trial)
            if abs(s_trial) < 1e-12:
                continue
            ac_trial = _autoconv(x_trial)
            max_ac_trial = np.max(np.abs(ac_trial))
            trial_score = max_ac_trial / (s_trial * s_trial)
            
            if trial_score < cur_score:
                x = x_trial
            else:
                # Occasionally accept worse moves (simulated annealing style)
                if rng.random() < 0.05:
                    x = x_trial
    
    # Final validation
    final_score = _score(x_best)
    if final_score < best_score:
        best_score = final_score
    
    return x_best
