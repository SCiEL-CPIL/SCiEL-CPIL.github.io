import numpy as np
import time
from scipy.optimize import minimize

N = 400
TIME_LIMIT = 150.0
SAFETY_MARGIN = 10.0
DEADLINE = time.time() + TIME_LIMIT - SAFETY_MARGIN

# Precompute FFT size
FFT_SIZE = 1
while FFT_SIZE < 2 * N - 1:
    FFT_SIZE *= 2

def _autoconv_fft(f):
    """Compute linear autoconvolution using FFT."""
    F = np.fft.rfft(f, n=FFT_SIZE)
    C = np.fft.irfft(F * F, n=FFT_SIZE)
    return C[:2 * N - 1]

def _score(f):
    S = np.sum(f)
    if abs(S) < 1e-12:
        return 1e10
    C = _autoconv_fft(f)
    return np.max(np.abs(C)) / (S * S)

def _local_search_binary_fast(f, max_iter=1000, time_budget=5.0):
    """
    Local search on binary {-1, 1} sequences using incremental convolution updates.
    """
    f = np.sign(f).astype(np.float64)
    S = np.sum(f)
    if abs(S) < 1e-6:
        return f, 1e10
        
    C = _autoconv_fft(f)
    score = np.max(np.abs(C)) / (S * S)
    
    start_time = time.time()
    no_improve_count = 0
    
    for it in range(max_iter):
        if time.time() - start_time > time_budget:
            break
        if time.time() > DEADLINE:
            break
            
        best_improvement = 0
        best_flip_idx = -1
        best_new_C = None
        best_new_S = S
        
        for i in range(N):
            v = f[i]
            # Flip f[i]
            # delta = -2 * v
            delta = -2.0 * v
            
            # Update C
            # C_new[k] = C[k] + 2 * delta * f[k-i] + delta^2 * (1 if k==2i else 0)
            
            C_new = C.copy()
            
            # Apply linear change: 2 * delta * f[k-i]
            # k ranges from 0 to 2N-2.
            # f[k-i] is valid if 0 <= k-i < N => i <= k < i+N.
            
            k_start = i
            k_end = i + N
            
            if k_start < len(C_new):
                # f[0 : N-k_start] corresponds to f[k-i] for k in [i, i+N)
                # Wait.
                # k = i + j. f[k-i] = f[j].
                # j ranges from 0 to N-1.
                # k ranges from i to i+N-1.
                
                # Check bounds
                end_idx = min(k_end, len(C_new))
                start_idx = k_start
                
                # The segment of f is f[0 : end_idx - start_idx]
                # The segment of C is C[start_idx : end_idx]
                
                f_segment = f[:end_idx - start_idx]
                
                C_new[start_idx:end_idx] += 2.0 * delta * f_segment
                
            # Apply quadratic change: delta^2 at k=2i
            if 2 * i < len(C_new):
                C_new[2 * i] += delta * delta
                
            new_S = S - 2.0 * v
            if abs(new_S) < 1e-6:
                continue
                
            new_score = np.max(np.abs(C_new)) / (new_S * new_S)
            
            if new_score < score - best_improvement:
                best_improvement = score - new_score
                best_flip_idx = i
                best_new_C = C_new
                best_new_S = new_S
        
        if best_flip_idx >= 0:
            v = f[best_flip_idx]
            f[best_flip_idx] = -v
            S = best_new_S
            C = best_new_C
            score = np.max(np.abs(C)) / (S * S)
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count > 10:
                break
                
    return f, score

def _legendre_symbol(a, p):
    """Compute Legendre symbol (a/p)."""
    if a % p == 0:
        return 0
    return pow(a, (p - 1) // 2, p)

def _generate_legendre_sequence(p):
    """
    Generate a sequence based on Legendre symbols modulo p.
    If p > N, we take the first N symbols.
    If p < N, we tile or repeat? 
    For p close to N, we take the first N-1 non-zero symbols and pad?
    Let's just generate a sequence of length N using the first N values.
    Note: Legendre symbol is 0 for a=p. We map 0 to 1 to keep binary.
    """
    seq = np.zeros(N)
    for i in range(N):
        val = _legendre_symbol(i, p)
        if val == 0:
            seq[i] = 1.0
        elif val == 1:
            seq[i] = 1.0
        else: # -1
            seq[i] = -1.0
    return seq

def _quadratic_phase_seq(N, k):
    """
    Generate a sequence based on quadratic phase e^(i*pi*k*i^2/N).
    Map to signs.
    """
    i = np.arange(N)
    phase = np.pi * k * i * i / N
    # Map complex phase to sign
    # cos(phase) > 0 -> 1, else -1
    return np.sign(np.cos(phase)).astype(np.float64)

def construct():
    start_time = time.time()
    
    best_f = np.ones(N)
    best_score = _score(best_f)
    
    rng = np.random.default_rng(42)
    
    # 1. Structured Initializations
    # Generate a set of diverse binary sequences
    
    candidates = []
    
    # a) All ones
    candidates.append(np.ones(N))
    
    # b) Legendre sequences for primes near N
    # Find primes near N
    def is_prime(n):
        if n < 2: return False
        for i in range(2, int(np.sqrt(n)) + 1):
            if n % i == 0:
                return False
        return True
        
    primes = []
    # Search for primes in a range around N
    for p in range(max(3, N - 20), N + 20):
        if is_prime(p):
            primes.append(p)
            
    for p in primes:
        candidates.append(_generate_legendre_sequence(p))
        
    # c) Quadratic phase sequences
    for k in [1, 2, 3, 5, 7, 11]:
        candidates.append(_quadratic_phase_seq(N, k))
        
    # d) Random binary sequences
    for _ in range(10):
        candidates.append(rng.choice([-1.0, 1.0], size=N))
        
    # e) Alternating patterns
    candidates.append(np.array([1.0 if i % 2 == 0 else -1.0 for i in range(N)]))
    candidates.append(np.array([1.0 if i % 3 < 2 else -1.0 for i in range(N)]))

    # 2. Local Search from each candidate
    for f0 in candidates:
        if time.time() > DEADLINE:
            break
            
        # Ensure binary
        f_bin = np.sign(f0)
        if np.sum(f_bin) == 0:
            f_bin = np.ones(N)
            
        # Run local search
        # Allocate more time to structured candidates? 
        # Let's use a fixed budget per candidate to ensure coverage.
        # 150s total. ~30 candidates. 5s each = 150s. 
        # Let's use 4s per candidate to be safe.
        f_local, score_local = _local_search_binary_fast(f_bin, max_iter=2000, time_budget=4.0)
        
        if score_local < best_score:
            best_score = score_local
            best_f = f_local.copy()

    # 3. If we have time left, do a few more random restarts with longer local search
    time_elapsed = time.time() - start_time
    time_remaining = TIME_LIMIT - SAFETY_MARGIN - time_elapsed
    
    if time_remaining > 10:
        num_extra = int(time_remaining / 5)
        for _ in range(num_extra):
            if time.time() > DEADLINE:
                break
            f_rand = rng.choice([-1.0, 1.0], size=N)
            if abs(np.sum(f_rand)) < N * 0.2:
                continue
            f_local, score_local = _local_search_binary_fast(f_rand, max_iter=2000, time_budget=5.0)
            if score_local < best_score:
                best_score = score_local
                best_f = f_local.copy()

    return best_f
