import numpy as np
import time
from scipy.signal import fftconvolve

N = 400
TIME_LIMIT = 120.0
SAFETY_MARGIN = 5.0
DEADLINE = time.time() + TIME_LIMIT - SAFETY_MARGIN

# Precompute FFT size for linear convolution of two length N sequences
# Result length is 2N - 1
FFT_SIZE = 1
while FFT_SIZE < 2 * N - 1:
    FFT_SIZE *= 2

def _autoconv(f):
    """Compute the linear (full) autoconvolution of f using FFT."""
    F = np.fft.rfft(f, n=FFT_SIZE)
    C = np.fft.irfft(F * F, n=FFT_SIZE)
    return C[:2 * N - 1]

def _objective_vectorized(f):
    """
    Compute max(|autoconv(f)|) / (sum(f))^2.
    If sum(f) == 0, return a large penalty.
    """
    S = np.sum(f)
    if abs(S) < 1e-12:
        return 1e10
    C = _autoconv(f)
    return np.max(np.abs(C)) / (S * S)

def _thue_morse_sequence(n):
    """Generate Thue-Morse sequence of length n as +/-1."""
    seq = np.zeros(n)
    for i in range(n):
        x = i
        count = 0
        while x:
            count += x & 1
            x >>= 1
        seq[i] = 1.0 if count % 2 == 0 else -1.0
    return seq

def _welch_costas_prime(p):
    """
    Generate a Welch Costas permutation for prime p.
    Returns a sequence of length p where perm[i] = i^2 mod p.
    """
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

    perm = np.zeros(p, dtype=int)
    for i in range(1, p):
        perm[i] = (i * i) % p

    b = np.zeros(p)
    b[perm < p // 2] = 1.0
    b[perm >= p // 2] = -1.0
    
    return b[:N].copy()

def _legendre_sequence(n):
    """Generate a Legendre sequence of length n based on quadratic residues."""
    # Find a prime p >= n
    def is_prime(n):
        if n < 2:
            return False
        for i in range(2, int(np.sqrt(n)) + 1):
            if n % i == 0:
                return False
        return True

    p = n
    while not is_prime(p):
        p += 1
        
    seq = np.zeros(n)
    for i in range(1, n):
        # Legendre symbol (i/p)
        if pow(i, (p-1)//2, p) == 1:
            seq[i] = 1.0
        else:
            seq[i] = -1.0
    seq[0] = 1.0 # Convention
    return seq

def _local_search_binary(f, time_budget=10.0):
    """
    Local search on binary {-1, 1} sequences.
    Uses vectorized evaluation of candidate flips for speed.
    """
    f = f.copy()
    S = np.sum(f)
    C = _autoconv(f)
    score = np.max(np.abs(C)) / (S * S)
    
    start_time = time.time()
    no_improve_count = 0
    
    while time.time() - start_time < time_budget:
        if time.time() > DEADLINE:
            break
            
        k_max = np.argmax(np.abs(C))
        
        candidates = set()
        # Focus on indices that contribute to the peak autocorrelation
        for j in range(N):
            if 0 <= k_max - j < N:
                candidates.add(j)
        
        # Add some random candidates to escape local minima
        for _ in range(20):
            candidates.add(np.random.randint(N))
            
        candidates = list(candidates)
        M = len(candidates)
        
        if M == 0:
            break
            
        # Create delta matrix
        D = np.zeros((M, 2*N - 1))
        S_new_arr = np.zeros(M)
        
        for m, i in enumerate(candidates):
            v = f[i]
            S_new_arr[m] = S - 2.0 * v
            
            k_start = i
            k_end = min(i + N, 2 * N - 1)
            k_end_clamped = min(k_end, 2 * N - 1)
            
            f_slice = f[:k_end_clamped - k_start]
            delta = -2.0 * v * f_slice
            
            D[m, k_start:k_end_clamped] = delta
            
            # Fix k=2i case where change is 0
            k_center = 2 * i
            if k_start <= k_center < k_end_clamped:
                D[m, k_center] = 0.0
                
        # Compute new scores
        C_new = C[None, :] + D
        max_abs_C = np.max(np.abs(C_new), axis=1)
        
        # Avoid division by zero
        valid = np.abs(S_new_arr) > 1e-6
        scores = np.full(M, 1e10)
        scores[valid] = max_abs_C[valid] / (S_new_arr[valid] ** 2)
        
        min_idx = np.argmin(scores)
        min_score = scores[min_idx]
        
        if min_score < score:
            i = candidates[min_idx]
            v = f[i]
            f[i] = -v
            S = S_new_arr[min_idx]
            C = C_new[min_idx].copy()
            score = min_score
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count > 50:
                break
                
    return f, score

def _gradient_descent_smooth(f, deadline, lr_init=0.05):
    """
    Gradient descent on a smooth surrogate of the objective.
    Objective: minimize max|ac(k)| / S^2.
    """
    f = f.copy()
    S = np.sum(f)
    if abs(S) < 1e-9:
        f = f + 0.1
        S = np.sum(f)
        
    size = 2 * N - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
        
    lr = lr_init
    best_val = _objective_vectorized(f)
    
    for step in range(500):
        if time.time() > deadline:
            break
        
        # Compute ac via FFT
        H = np.fft.rfft(f, fsize)
        ac = np.fft.irfft(H * H, fsize)[:size]
        
        abs_ac = np.abs(ac)
        k_star = np.argmax(abs_ac)
        sign_ac = np.sign(ac[k_star])
        
        # Gradient of max|ac| w.r.t f
        grad_M = np.zeros(N)
        if k_star == 0:
            grad_M = 2.0 * f
        else:
            idx1 = np.arange(N - k_star)
            grad_M[idx1] += sign_ac * f[idx1 + k_star]
            idx2 = np.arange(k_star, N)
            grad_M[idx2] += sign_ac * f[idx2 - k_star]
            
        # Objective is max|ac| / S^2
        # grad (max|ac| / S^2) = (1/S^2) * grad(max|ac|) - (2 * max|ac| / S^3) * grad(S)
        # grad(S) = 1 for all i
        max_abs_ac = np.max(abs_ac)
        grad_obj = (1.0 / (S * S)) * grad_M - (2.0 * max_abs_ac / (S * S * S)) * np.ones(N)
        
        f_new = f - lr * grad_obj
        
        # Normalize to keep sum close to original or fixed?
        s_new = np.sum(f_new)
        if abs(s_new) > 1e-9:
            f_new = f_new * (S / s_new)
        else:
            f_new = f
            
        val_new = _objective_vectorized(f_new)
        
        if val_new < best_val:
            best_val = val_new
            f = f_new
            lr *= 0.99
        else:
            lr *= 0.95
            if lr < 1e-6:
                break
                
    return f, best_val

def construct():
    start_time = time.time()
    
    best_f = np.ones(N)
    best_score = _objective_vectorized(best_f)
    
    rng = np.random.default_rng(42)
    
    # Strategy: 
    # 1. Start with Thue-Morse (strong baseline for low autocorrelation).
    # 2. Start with Costas-derived sequence.
    # 3. Start with Legendre sequence.
    # 4. Start with random binary.
    # 5. Refine each with gradient descent (smooth) then binary local search.
    
    initializations = []
    
    # Thue-Morse
    f_tm = _thue_morse_sequence(N)
    initializations.append(f_tm)
    
    # Costas-derived
    try:
        f_costas = _welch_costas_prime(N)
        # Ensure it's binary
        f_costas = np.sign(f_costas)
        if np.all(f_costas == 0):
            f_costas = np.ones(N)
        initializations.append(f_costas)
    except:
        pass
        
    # Legendre
    try:
        f_leg = _legendre_sequence(N)
        initializations.append(f_leg)
    except:
        pass
        
    # Random binary
    for _ in range(5):
        f_rand = rng.choice([-1.0, 1.0], size=N)
        # Ensure sum is not too small
        s = np.sum(f_rand)
        if abs(s) < N * 0.1:
            f_rand = f_rand + 0.5
            f_rand = np.sign(f_rand)
        initializations.append(f_rand)
        
    # Perturbed Thue-Morse
    for _ in range(5):
        f_ptm = _thue_morse_sequence(N)
        flip_mask = rng.random(N) < 0.1
        f_ptm[flip_mask] *= -1.0
        initializations.append(f_ptm)
        
    # Evaluate initial scores
    for f_init in initializations:
        val = _objective_vectorized(f_init)
        if val < best_score:
            best_score = val
            best_f = f_init.copy()
            
    # Refine each initialization
    for i, f_init in enumerate(initializations):
        if time.time() > DEADLINE:
            break
            
        remaining_time = DEADLINE - time.time()
        if remaining_time < 2.0:
            break
            
        # Phase 1: Gradient Descent (smooth)
        time_budget_grad = min(10.0, remaining_time * 0.5)
        f_grad, val_grad = _gradient_descent_smooth(f_init, start_time + time_budget_grad)
        
        if val_grad < best_score:
            best_score = val_grad
            best_f = f_grad.copy()
            
        # Phase 2: Binary Local Search
        # Convert to binary if it drifted
        f_bin = np.sign(f_grad)
        if np.all(f_bin == 0):
            f_bin = np.ones(N)
            
        time_budget_local = min(5.0, remaining_time * 0.3)
        f_local, val_local = _local_search_binary(f_bin, time_budget=time_budget_local)
        
        if val_local < best_score:
            best_score = val_local
            best_f = f_local.copy()
            
    return best_f
