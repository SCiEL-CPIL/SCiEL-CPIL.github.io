import numpy as np
import time
from scipy.optimize import minimize

N = 400
TIME_LIMIT = 120.0
SAFETY_MARGIN = 5.0
DEADLINE = time.time() + TIME_LIMIT - SAFETY_MARGIN

# Precompute FFT size for autoconvolution
FFT_SIZE = 1
while FFT_SIZE < 2 * N - 1:
    FFT_SIZE *= 2


def _autoconv(f):
    """Compute the linear (full) autoconvolution of f using FFT."""
    F = np.fft.rfft(f, n=FFT_SIZE)
    C = np.fft.irfft(F * F, n=FFT_SIZE)
    return C[:2 * N - 1]


def _score(f):
    """Compute max(|autoconv(f)|) / (sum(f))^2."""
    S = np.sum(f)
    if abs(S) < 1e-12:
        return 1e10
    C = _autoconv(f)
    return np.max(np.abs(C)) / (S * S)


def _objective_smooth(f, t=0.01):
    """
    Smooth surrogate using log-sum-exp for the max.
    """
    S = np.sum(f)
    if abs(S) < 1e-12:
        return 1e10
    C = _autoconv(f)
    absC = np.abs(C)
    max_val = np.max(absC)
    lse = max_val + np.log(np.sum(np.exp(t * (absC - max_val))))
    return (lse / t) / (S * S)


def _gradient_smooth(f, t=0.01):
    """
    Gradient of the smooth objective using FFT-based approach.
    """
    S = np.sum(f)
    if abs(S) < 1e-12:
        return np.zeros_like(f)
    
    C = _autoconv(f)
    absC = np.abs(C)
    signC = np.sign(C)
    max_val = np.max(absC)
    
    exp_term = np.exp(t * (absC - max_val))
    sum_exp = np.sum(exp_term)
    weights = exp_term / sum_exp
    
    A = weights * signC  # length 2N-1
    
    # Compute gradient efficiently
    grad = np.zeros(N)
    for k in range(2 * N - 1):
        a_val = A[k]
        if abs(a_val) < 1e-15:
            continue
        k_offset = k - (N - 1)  # actual lag
        if k_offset >= 0:
            n_valid = N - k_offset
            if n_valid > 0:
                grad[:n_valid] += a_val * f[k_offset:k_offset + n_valid]
                grad[k_offset:] += a_val * f[:n_valid]
        else:
            k_pos = -k_offset
            n_valid = N - k_pos
            if n_valid > 0:
                grad[:n_valid] += a_val * f[k_pos:k_pos + n_valid]
                grad[k_pos:] += a_val * f[:n_valid]
    
    LSE_val = max_val + np.log(np.sum(np.exp(t * (absC - max_val))))
    G_val = LSE_val / t
    
    return (1.0 / (S * S)) * grad - (G_val / (S * S * S)) * 2.0 * S


def _generate_legendre_sequence(N):
    """
    Generate a Legendre sequence based on the Legendre symbol.
    For prime p, the Legendre sequence has length p with values ±1.
    We adapt it to length N by truncation or padding.
    """
    # Find a prime >= N
    def is_prime(n):
        if n < 2:
            return False
        for i in range(2, int(np.sqrt(n)) + 1):
            if n % i == 0:
                return False
        return True
    
    p = N
    while not is_prime(p):
        p += 1
    
    # Legendre symbol: (a|p) = a^((p-1)/2) mod p, mapped to ±1
    seq = np.zeros(p)
    for a in range(p):
        if a == 0:
            seq[a] = 0.0
        else:
            val = pow(a, (p - 1) // 2, p)
            seq[a] = 1.0 if val == 1 else -1.0
    
    # Truncate to N
    seq = seq[:N]
    
    # Ensure non-zero sum
    S = np.sum(seq)
    if abs(S) < N * 0.05:
        # Flip some signs to increase |S|
        rng = np.random.default_rng(42)
        for _ in range(N // 2):
            idx = rng.integers(0, N)
            if seq[idx] * S < 0:
                seq[idx] = -seq[idx]
                S = np.sum(seq)
            if abs(S) > N * 0.1:
                break
    
    return seq


def _generate_chirp_sequence(N, freq=0.01):
    """
    Generate a chirp (quadratic phase) sequence.
    f[n] = sign(sin(2*pi*freq*n^2))
    """
    n = np.arange(N)
    phase = 2 * np.pi * freq * n * n
    f = np.sign(np.sin(phase))
    f[f == 0] = 1.0
    
    # Ensure non-zero sum
    S = np.sum(f)
    if abs(S) < N * 0.05:
        rng = np.random.default_rng(42)
        for _ in range(N // 2):
            idx = rng.integers(0, N)
            if f[idx] * S < 0:
                f[idx] = -f[idx]
                S = np.sum(f)
            if abs(S) > N * 0.1:
                break
    
    return f


def _optimize_chirp_freq(N, max_iter=20, time_budget=3.0):
    """
    Optimize the chirp frequency to minimize the score.
    """
    start_time = time.time()
    best_score = 1e10
    best_freq = 0.01
    
    # Try a range of frequencies
    freqs = np.linspace(0.001, 0.1, 50)
    for freq in freqs:
        if time.time() - start_time > time_budget:
            break
        f = _generate_chirp_sequence(N, freq)
        s = _score(f)
        if s < best_score:
            best_score = s
            best_freq = freq
    
    return best_freq


def _local_search_binary(f, max_iter=3000, time_budget=8.0):
    """
    Local search on binary {-1, 1} sequences.
    Tries single flips and pair flips, accepting if it improves the objective.
    """
    f = f.copy()
    S = np.sum(f)
    C = _autoconv(f)
    score = np.max(np.abs(C)) / (S * S)
    
    start_time = time.time()
    no_improve_count = 0
    
    for it in range(max_iter):
        if time.time() - start_time > time_budget:
            break
        if time.time() > DEADLINE:
            break
        
        k_max = np.argmax(np.abs(C))
        
        # Collect candidate positions for single flips
        candidates_single = set()
        for _ in range(15):
            candidates_single.add(np.random.randint(N))
        
        for j in range(N):
            if 0 <= k_max - j < N:
                candidates_single.add(j)
        
        # Collect candidate pairs for pair flips
        candidates_pair = set()
        for _ in range(8):
            i1 = np.random.randint(N)
            i2 = np.random.randint(N)
            if i1 != i2:
                candidates_pair.add((min(i1, i2), max(i1, i2)))
        
        # Add pairs near the peak
        for j in range(min(20, N)):
            if 0 <= k_max - j < N and j != k_max - j:
                candidates_pair.add((min(j, k_max - j), max(j, k_max - j)))
        
        best_improvement = 0
        best_move = None
        
        # Try single flips
        for i in candidates_single:
            v = f[i]
            k_start = i
            k_end = min(i + N, 2 * N - 1)
            k_end_clamped = min(k_end, len(C))
            f_slice = f[:k_end_clamped - k_start]
            
            delta_C = np.zeros_like(C)
            delta_C[k_start:k_end_clamped] = -2.0 * v * f_slice
            
            new_C = C + delta_C
            new_S = S - 2.0 * v
            
            if abs(new_S) < 1e-6:
                continue
            
            new_score = np.max(np.abs(new_C)) / (new_S * new_S)
            
            if new_score < score - best_improvement:
                best_improvement = score - new_score
                best_move = ('single', (i,))
        
        # Try pair flips
        for (i, j) in candidates_pair:
            vi, vj = f[i], f[j]
            
            delta_C = np.zeros_like(C)
            k_start_i = i
            k_end_i = min(i + N, 2 * N - 1)
            delta_C[k_start_i:k_end_i] += -2.0 * vi * f[:k_end_i - k_start_i]
            k_start_j = j
            k_end_j = min(j + N, 2 * N - 1)
            delta_C[k_start_j:k_end_j] += -2.0 * vj * f[:k_end_j - k_start_j]
            
            new_C = C + delta_C
            new_S = S - 2.0 * vi - 2.0 * vj
            
            if abs(new_S) < 1e-6:
                continue
            
            new_score = np.max(np.abs(new_C)) / (new_S * new_S)
            
            if new_score < score - best_improvement:
                best_improvement = score - new_score
                best_move = ('pair', (i, j))
        
        if best_move is not None:
            if best_move[0] == 'single':
                i = best_move[1][0]
                f[i] = -f[i]
            else:
                i, j = best_move[1]
                f[i] = -f[i]
                f[j] = -f[j]
            
            S = np.sum(f)
            C = _autoconv(f)
            score = np.max(np.abs(C)) / (S * S)
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count > 50:
                break
    
    return f, score


def _polish_search(f, time_budget=5.0):
    """
    High-precision polish pass with small perturbations.
    """
    f = f.copy()
    S = np.sum(f)
    C = _autoconv(f)
    score = np.max(np.abs(C)) / (S * S)
    
    start_time = time.time()
    rng = np.random.default_rng(123)
    
    for _ in range(500):
        if time.time() - start_time > time_budget:
            break
        if time.time() > DEADLINE:
            break
        
        idx = rng.integers(0, N)
        delta = rng.choice([-0.05, 0.05, -0.1, 0.1])
        f[idx] += delta
        new_S = np.sum(f)
        if abs(new_S) < 1e-6:
            f[idx] -= delta
            continue
        new_C = _autoconv(f)
        new_score = np.max(np.abs(new_C)) / (new_S * new_S)
        if new_score < score:
            score = new_score
        else:
            f[idx] -= delta
    
    return f, score


def _spectral_init(rng, N):
    """
    Initialize using windowed sinc in frequency domain to structurally
    suppress peak autoconvolution sidelobes.
    """
    n = N
    w = np.zeros(n)
    for i in range(n):
        x = i / (n - 1)
        w[i] = (0.35875 - 0.48829 * np.cos(2 * np.pi * x) +
                0.14128 * np.cos(4 * np.pi * x) - 0.01168 * np.cos(6 * np.pi * x))
    
    phases = rng.uniform(0, 2 * np.pi, n)
    freq_signal = w * np.exp(1j * phases)
    time_signal = np.fft.ifft(freq_signal).real
    
    f = np.sign(time_signal)
    f[f == 0] = 1.0
    
    S = np.sum(f)
    if abs(S) < N * 0.05:
        for _ in range(N // 4):
            idx = rng.integers(0, N)
            if f[idx] * S < 0:
                f[idx] = -f[idx]
                S = np.sum(f)
            if abs(S) > N * 0.2:
                break
    
    return f


def construct():
    start_time = time.time()
    
    best_f = np.ones(N)
    best_score = _score(best_f)
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initializations
    initializations = []
    
    # Legendre sequence
    f_leg = _generate_legendre_sequence(N)
    initializations.append(f_leg)
    
    # Chirp sequences with optimized frequency
    best_freq = _optimize_chirp_freq(N, time_budget=2.0)
    for freq in [best_freq, best_freq * 0.5, best_freq * 2.0, 0.005, 0.02, 0.05]:
        f_chirp = _generate_chirp_sequence(N, freq)
        initializations.append(f_chirp)
    
    # Spectral initializations
    for _ in range(4):
        if time.time() > DEADLINE:
            break
        f_spec = _spectral_init(rng, N)
        initializations.append(f_spec)
    
    # Random binary initializations
    for _ in range(6):
        if time.time() > DEADLINE:
            break
        f0 = rng.choice([-1.0, 1.0], size=N)
        S0 = np.sum(f0)
        if abs(S0) < 5:
            for _ in range(N // 2):
                idx = rng.integers(0, N)
                if f0[idx] * S0 < 0:
                    f0[idx] = -f0[idx]
                    S0 = np.sum(f0)
            if abs(S0) < 5:
                continue
        initializations.append(f0)
    
    # Continuous optimization from some starts
    for restart in range(min(3, len(initializations))):
        if time.time() > DEADLINE:
            break
        
        f0 = initializations[restart].copy()
        if abs(np.sum(f0)) > 1e-6:
            f0 = f0 / np.sum(f0)
        else:
            f0 = np.ones(N)
        
        ts = [0.1, 0.05, 0.01]
        current_f = f0
        
        for t in ts:
            if time.time() > DEADLINE:
                break
            
            try:
                res = minimize(
                    _objective_smooth,
                    current_f,
                    args=(t,),
                    method='L-BFGS-B',
                    jac=_gradient_smooth,
                    options={'maxiter': 30, 'maxls': 30, 'ftol': 1e-9, 'gtol': 1e-7}
                )
                
                if res.fun < best_score * 1.01:
                    true_score = _score(res.x)
                    if true_score < best_score:
                        best_score = true_score
                        best_f = res.x.copy()
                    current_f = res.x
            except Exception:
                continue
    
    # Binary local search from all initializations
    for f0 in initializations:
        if time.time() > DEADLINE:
            break
        
        remaining_time = DEADLINE - time.time()
        time_budget = min(5.0, remaining_time * 0.6)
        
        if time_budget < 0.5:
            break
        
        f_local, score_local = _local_search_binary(f0, max_iter=2000, time_budget=time_budget)
        
        if score_local < best_score:
            best_score = score_local
            best_f = f_local.copy()
    
    # Final polish
    if time.time() < DEADLINE:
        remaining_time = DEADLINE - time.time()
        if remaining_time > 3:
            time_budget = min(3.0, remaining_time * 0.8)
            f_polish, score_polish = _polish_search(best_f, time_budget=time_budget)
            if score_polish < best_score:
                best_score = score_polish
                best_f = f_polish.copy()
    
    return best_f
