"""Rewritten program: local search for signed step heights minimizing
max absolute off-zero autoconvolution divided by squared integral."""

import time
import numpy as np

N = 400
DEADLINE = 110.0  # seconds wall-clock safety margin before 120s kill


def _fft_size(n):
    size = 1
    while size < 2 * n - 1:
        size <<= 1
    return size


def _autocorr_offmax(f, size):
    """Compute max absolute off-zero autocorrelation."""
    n = len(f)
    F = np.fft.rfft(f, size)
    power = F * np.conj(F)
    ac = np.fft.irfft(power, size)[:2 * n - 1]
    off = np.concatenate([ac[:n - 1], ac[n:]])
    return float(np.max(np.abs(off)))


def _objective(f):
    """max absolute off-zero autoconvolution divided by (sum f)^2."""
    n = len(f)
    size = _fft_size(n)
    s = f.sum()
    s2 = s * s
    if abs(s2) < 1e-9:
        return float('inf')
    offmax = _autocorr_offmax(f, size)
    return offmax / s2


def _logsumexp_max_abs(off, beta=100.0):
    """Smooth approximation to max|off| using log-sum-exp."""
    x = np.abs(off)
    # Avoid log(0) issues
    x = np.maximum(x, 1e-30)
    m = np.max(x)
    if m < 1e-15:
        return 0.0
    # log-sum-exp: m + log(sum(exp((x-m)/beta))) / beta
    # This approximates max when beta is small
    scaled = (x - m) / beta
    log_sum = np.log(np.sum(np.exp(scaled)))
    return m + beta * log_sum


def _soft_objective(f, beta=100.0):
    """Smooth surrogate: soft-max of |off| / sum^2."""
    n = len(f)
    size = _fft_size(n)
    s = f.sum()
    s2 = s * s
    if abs(s2) < 1e-9:
        return float('inf')
    F = np.fft.rfft(f, size)
    power = F * np.conj(F)
    ac = np.fft.irfft(power, size)[:2 * n - 1]
    off = np.concatenate([ac[:n - 1], ac[n:]])
    soft_max = _logsumexp_max_abs(off, beta)
    return soft_max / s2


def _gradient_soft(f, beta=100.0):
    """Compute gradient of soft objective w.r.t. f (for continuous relaxation)."""
    n = len(f)
    size = _fft_size(n)
    s = f.sum()
    s2 = s * s
    if abs(s2) < 1e-9:
        return None
    
    F = np.fft.rfft(f, size)
    power = F * np.conj(F)
    ac = np.fft.irfft(power, size)[:2 * n - 1]
    off = np.concatenate([ac[:n - 1], ac[n:]])
    
    x = np.abs(off)
    x_safe = np.maximum(x, 1e-30)
    m = np.max(x_safe)
    if m < 1e-15:
        return None
    
    # weights for log-sum-exp
    scaled = (x_safe - m) / beta
    exp_scaled = np.exp(scaled)
    weights = exp_scaled / np.sum(exp_scaled)
    
    # d(soft_max)/d(off_k) = weights[k] * sign(off_k)
    d_soft = weights * np.sign(off)
    
    # d(soft_max / s^2)/d(f_j) = (d_soft_k * d(off_k)/d(f_j)) / s^2 - 2*soft_max * s / s^4 * 1
    # This is complex; instead use finite differences for gradient
    
    # Use central differences for gradient estimation
    grad = np.zeros(n)
    eps = 1e-5
    f0 = f.copy()
    obj0 = _soft_objective(f0, beta)
    for j in range(n):
        if time.time() - _gradient_soft.start_time > DEADLINE - 5:
            return None
        f_plus = f0.copy()
        f_plus[j] += eps
        f_minus = f0.copy()
        f_minus[j] -= eps
        obj_plus = _soft_objective(f_plus, beta)
        obj_minus = _soft_objective(f_minus, beta)
        grad[j] = (obj_plus - obj_minus) / (2 * eps)
    
    return grad


def construct():
    start = time.time()
    _gradient_soft.start_time = start
    
    n = N
    size = _fft_size(n)
    
    best = None
    best_obj = float('inf')
    
    def try_candidate(cand):
        nonlocal best, best_obj
        s = cand.sum()
        if abs(s) < 1e-9:
            return
        offmax = _autocorr_offmax(cand, size)
        obj = offmax / (s * s)
        if obj < best_obj:
            best_obj = obj
            best = cand.copy()
    
    # Phase 1: Generate diverse starting points
    rng = np.random.default_rng(42)
    
    # Start with all ones
    f0 = np.ones(n, dtype=np.float64)
    try_candidate(f0)
    
    # Random binary vectors
    for _ in range(50):
        if time.time() - start > DEADLINE:
            break
        cand = rng.choice([-1.0, 1.0], size=n)
        try_candidate(cand)
    
    # Structured patterns: low-frequency Fourier modes
    for freq in [1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]:
        if time.time() - start > DEADLINE:
            break
        t = np.arange(n)
        for phase in [0, np.pi/2, np.pi, 3*np.pi/2]:
            cand = np.sign(np.cos(2*np.pi*freq*t/n + phase))
            cand[cand == 0] = 1.0
            try_candidate(cand)
    
    # Random low-pass filtered
    for _ in range(30):
        if time.time() - start > DEADLINE:
            break
        # Generate in frequency domain with low-pass
        F_rand = np.random.randn(size // 2 + 1) + 1j * np.random.randn(size // 2 + 1)
        # Keep only low frequencies
        cutoff = size // 8
        F_rand[cutoff:] = 0
        cand = np.real(np.fft.irfft(F_rand, size))[:n]
        cand = np.sign(cand)
        cand[cand == 0] = 1.0
        try_candidate(cand)
    
    if best is None:
        return np.ones(n, dtype=np.float64)
    
    # Phase 2: Local search with efficient FFT updates
    def local_search(cur, cur_sum, max_time=30.0):
        nonlocal best, best_obj
        start_ls = time.time()
        cur = cur.copy()
        cur_sum = cur_sum
        cur_offmax = _autocorr_offmax(cur, size)
        cur_obj = cur_offmax / (cur_sum * cur_sum)
        
        # Precompute phase factors for FFT updates
        # F[k] = sum_j f[j] * exp(-2*pi*i*k*j/size)
        # Flipping f[i] from v to -v changes F[k] by -2*v*exp(-2*pi*i*k*i/size)
        k_vals = np.arange(size // 2 + 1)
        phase_factors = np.exp(-2j * np.pi * np.outer(k_vals, np.arange(n)) / size)
        
        improved = True
        while improved and (time.time() - start_ls < max_time) and (time.time() - start < DEADLINE):
            improved = False
            best_gain = 0.0
            best_idx = -1
            best_new_offmax = cur_offmax
            best_new_sum = cur_sum
            
            # Sample a subset of positions for speed
            if n > 100:
                indices = rng.choice(n, size=min(n, 50), replace=False)
            else:
                indices = range(n)
            
            for i in indices:
                if time.time() - start > DEADLINE:
                    return
                old_val = cur[i]
                new_sum = cur_sum - 2.0 * old_val
                if abs(new_sum) < 1e-9:
                    continue
                
                # Update FFT: F_new[k] = F_old[k] - 2*old_val*exp(-2*pi*i*k*i/size)
                F_old = np.fft.rfft(cur, size)
                delta = -2.0 * old_val * phase_factors[:, i]
                F_new = F_old + delta
                power_new = F_new * np.conj(F_new)
                ac_new = np.fft.irfft(power_new, size)[:2 * n - 1]
                off_new = np.concatenate([ac_new[:n - 1], ac_new[n:]])
                new_offmax = float(np.max(np.abs(off_new)))
                new_obj = new_offmax / (new_sum * new_sum)
                gain = cur_obj - new_obj
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
                    best_new_offmax = new_offmax
                    best_new_sum = new_sum
            
            if best_idx >= 0:
                cur[best_idx] = -cur[best_idx]
                cur_sum = best_new_sum
                cur_offmax = best_new_offmax
                cur_obj = cur_offmax / (cur_sum * cur_sum)
                if cur_obj < best_obj:
                    best_obj = cur_obj
                    best = cur.copy()
                improved = True
        
        return cur
    
    # Refine top candidates
    candidates_to_refine = []
    
    # Re-evaluate all starting points and keep top ones
    all_starts = [np.ones(n, dtype=np.float64)]
    rng2 = np.random.default_rng(42)
    for _ in range(50):
        all_starts.append(rng2.choice([-1.0, 1.0], size=n).astype(np.float64))
    for freq in [1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]:
        t = np.arange(n)
        for phase in [0, np.pi/2, np.pi, 3*np.pi/2]:
            cand = np.sign(np.cos(2*np.pi*freq*t/n + phase))
            cand[cand == 0] = 1.0
            all_starts.append(cand.astype(np.float64))
    for _ in range(30):
        F_rand = np.random.randn(size // 2 + 1) + 1j * np.random.randn(size // 2 + 1)
        cutoff = size // 8
        F_rand[cutoff:] = 0
        cand = np.real(np.fft.irfft(F_rand, size))[:n]
        cand = np.sign(cand)
        cand[cand == 0] = 1.0
        all_starts.append(cand.astype(np.float64))
    
    # Score all starts
    scored = []
    for cand in all_starts:
        s = cand.sum()
        if abs(s) < 1e-9:
            continue
        offmax = _autocorr_offmax(cand, size)
        obj = offmax / (s * s)
        scored.append((obj, cand, s))
    
    scored.sort(key=lambda x: x[0])
    
    # Refine top 10 candidates
    for obj_val, cand, s in scored[:10]:
        if time.time() - start > DEADLINE:
            break
        time_left = DEADLINE - (time.time() - start)
        ls_time = min(20.0, time_left * 0.4)
        if ls_time < 2.0:
            break
        local_search(cand, s, max_time=ls_time)
    
    # Phase 3: Soft-max guided search if time remains
    if time.time() - start < DEADLINE - 10:
        # Use soft objective to guide continuous optimization, then round
        for _ in range(5):
            if time.time() - start > DEADLINE - 5:
                break
            # Take best candidate, relax to continuous, do gradient steps
            if best is None:
                break
            f_cont = best.astype(np.float64).copy()
            beta = 50.0
            for step in range(50):
                if time.time() - start > DEADLINE - 5:
                    break
                grad = _gradient_soft(f_cont, beta)
                if grad is None:
                    break
                # Gradient descent with clipping
                lr = 0.1
                f_cont -= lr * grad
                # Project to [-1, 1]
                f_cont = np.clip(f_cont, -1.0, 1.0)
            
            # Round to signs
            f_rounded = np.sign(f_cont)
            f_rounded[f_rounded == 0] = 1.0
            try_candidate(f_rounded)
            
            # Local search on rounded
            if time.time() - start < DEADLINE - 5:
                s = f_rounded.sum()
                if abs(s) > 1e-9:
                    local_search(f_rounded, s, max_time=min(5.0, DEADLINE - (time.time() - start) - 3))
    
    return best if best is not None else np.ones(n, dtype=np.float64)
