"""Rewritten program with spectral initialization and accelerated local search."""
import numpy as np
import time
import random

N = 400
DEADLINE = 135.0
MARGIN = 12.0


def _autocorr_fft(x):
    n = 2 * N - 1
    nfft = 1 << (n - 1).bit_length()
    F = np.fft.rfft(x, nfft)
    P = F * np.conj(F)
    return np.fft.irfft(P, nfft)[:n]


def _c3_value(R):
    R0 = R[0]
    if abs(R0) < 1e-12:
        return 1e10
    pos = np.abs(R[1:N])
    neg = np.abs(R[N:2*N-1])
    return max(np.max(pos), np.max(neg)) / R0


def _c3_value_fast(x):
    R = _autocorr_fft(x)
    return _c3_value(R)


def _zadoff_chu(N, root=1):
    """Generate Zadoff-Chu sequence and binarize."""
    n = N
    m = root
    k = np.arange(n)
    zc = np.exp(-1j * np.pi * m * k * (k + 1) / n)
    return np.sign(np.real(zc)).astype(float)


def _chirp_binarized(N):
    """Linear chirp, binarized."""
    t = np.arange(N) / N
    phase = 2 * np.pi * (t**2)
    sig = np.cos(phase)
    return np.sign(sig).astype(float)


def _flat_spectral_init(N):
    """Create a signal with flat power spectrum, then binarize.
    Start in frequency domain with random phases, inverse FFT, binarize."""
    rng = np.random.RandomState(777)
    nfft = 1 << (N - 1).bit_length()
    F = np.zeros(nfft // 2 + 1, dtype=complex)
    # Uniform magnitude, random phase
    mags = np.ones(nfft // 2 + 1)
    phases = rng.uniform(0, 2*np.pi, size=nfft // 2 + 1)
    F = mags * np.exp(1j * phases)
    F[0] = 1.0  # DC
    x = np.fft.irfft(F, nfft)[:N]
    x = np.sign(x)
    x[x == 0] = 1.0
    return x.astype(float)


def _multi_freq_init(N, seed):
    """Multiple frequency-domain random initializations."""
    rng = np.random.RandomState(seed)
    nfft = 1 << (N - 1).bit_length()
    F = np.zeros(nfft // 2 + 1, dtype=complex)
    mags = rng.uniform(0.5, 1.5, size=nfft // 2 + 1)
    phases = rng.uniform(0, 2*np.pi, size=nfft // 2 + 1)
    F = mags * np.exp(1j * phases)
    x = np.fft.irfft(F, nfft)[:N]
    x = np.sign(x)
    x[x == 0] = 1.0
    return x.astype(float)


def _flip_delta(R, x, idx):
    """Compute delta in autocorrelation R when flipping x[idx].
    Returns only the positive-lag part delta[1..N-1]."""
    old = x[idx]
    # For lag k: R[k] = sum_i x[i]*x[i+k]
    # Flipping x[idx] affects:
    #   i=idx: term x[idx]*x[idx+k], k in 1..N-1-idx
    #   i=idx-k: term x[idx-k]*x[idx], k in 1..idx
    delta = np.zeros(N)  # index 0..N-1, we use 1..N-1
    # Positive k where idx+k < N
    kmax = N - 1 - idx
    if kmax >= 1:
        ks = np.arange(1, kmax + 1)
        delta[ks] -= 2.0 * old * x[idx + ks]
    # Positive k where idx-k >= 0
    if idx >= 1:
        ks = np.arange(1, idx + 1)
        delta[ks] -= 2.0 * x[idx - ks] * old
    return delta


def _local_search_v2(x0, budget):
    """Fast local search with vectorized flip evaluation."""
    t0 = time.time()
    x = x0.copy()
    R = _autocorr_fft(x)
    c3 = _c3_value(R)
    best_x = x.copy()
    best_c3 = c3
    no_improve = 0
    max_no_improve = 200
    n_improve = 0
    
    while no_improve < max_no_improve:
        if time.time() - t0 > budget or time.time() > DEADLINE - MARGIN:
            break
        # Pick random index
        idx = random.randint(0, N-1)
        old = x[idx]
        x[idx] = -old
        delta = _flip_delta(R, x, idx)  # x already flipped, so use new value
        # Wait, _flip_delta assumes x[idx] is the OLD value. Let me fix.
        x[idx] = old  # restore for delta computation
        delta = _flip_delta(R, x, idx)
        R_new = R + delta
        x[idx] = -old  # apply flip
        
        c3_new = _c3_value(R_new)
        if c3_new < c3 - 1e-14:
            R = R_new
            c3 = c3_new
            no_improve = 0
            n_improve += 1
            if c3 < best_c3:
                best_c3 = c3
                best_x = x.copy()
        else:
            x[idx] = old  # undo
            no_improve += 1
    
    return best_x, best_c3


def _local_search_batched(x0, budget):
    """Evaluate many candidate flips at once for speed."""
    t0 = time.time()
    x = x0.copy()
    R = _autocorr_fft(x)
    c3 = _c3_value(R)
    best_x = x.copy()
    best_c3 = c3
    no_improve = 0
    max_no_improve = 150
    batch_size = 64
    
    while no_improve < max_no_improve:
        if time.time() - t0 > budget or time.time() > DEADLINE - MARGIN:
            break
        
        # Sample batch of indices
        indices = random.sample(range(N), min(batch_size, N))
        best_delta_idx = None
        best_delta_c3 = c3
        best_new_R = None
        best_new_x = None
        
        for idx in indices:
            old = x[idx]
            delta = _flip_delta(R, x, idx)
            R_candidate = R + delta
            c3_candidate = _c3_value(R_candidate)
            if c3_candidate < best_delta_c3 - 1e-14:
                best_delta_c3 = c3_candidate
                best_delta_idx = idx
                best_new_R = R_candidate
                best_new_x = x.copy()
                best_new_x[idx] = -old
        
        if best_delta_idx is not None:
            x = best_new_x
            R = best_new_R
            c3 = best_delta_c3
            no_improve = 0
            if c3 < best_c3:
                best_c3 = c3
                best_x = x.copy()
        else:
            no_improve += 1
    
    return best_x, best_c3


def _spectral_refinement(x, budget):
    """Refine using frequency-domain perturbation."""
    t0 = time.time()
    x_best = x.copy()
    c3_best = _c3_value_fast(x_best)
    
    nfft = 1 << (N - 1).bit_length()
    rng = np.random.RandomState(999)
    
    for _ in range(5000):
        if time.time() - t0 > budget or time.time() > DEADLINE - MARGIN:
            break
        
        # Perturb in frequency domain
        F = np.fft.rfft(x_best, nfft)
        F_pert = F.copy()
        n_pert = random.randint(1, 5)
        freqs = random.sample(range(1, len(F_pert)), n_pert)
        for f in freqs:
            F_pert[f] += rng.normal(0, 0.1)
        
        x_pert = np.fft.irfft(F_pert, nfft)[:N]
        x_bin = np.sign(x_pert)
        x_bin[x_bin == 0] = 1.0
        x_bin = x_bin.astype(float)
        
        c3_pert = _c3_value_fast(x_bin)
        if c3_pert < c3_best - 1e-14:
            c3_best = c3_pert
            x_best = x_bin
    
    return x_best, c3_best


def _construct():
    t_start = time.time()
    random.seed(12345)
    np.random.seed(12345)
    
    # Generate diverse initializations
    inits = []
    
    # Zadoff-Chu sequences with different roots
    for root in [1, 3, 5, 7, 13, 17, 19]:
        inits.append(_zadoff_chu(N, root))
    
    # Chirp
    inits.append(_chirp_binarized(N))
    
    # Flat spectral initializations
    inits.append(_flat_spectral_init(N))
    for s in [1, 2, 3, 4, 5]:
        inits.append(_multi_freq_init(N, seed=s))
    
    # Thue-Morse
    tm = np.array([1.0 if bin(i).count('1') % 2 == 0 else -1.0 for i in range(N)])
    inits.append(tm)
    
    # Random
    rng = np.random.RandomState(42)
    inits.append(rng.choice([-1.0, 1.0], size=N))
    
    # Sort by initial C3
    scored = []
    for x in inits:
        c3 = _c3_value_fast(x)
        scored.append((c3, x))
    scored.sort(key=lambda t: t[0])
    
    best_x = scored[0][1].copy()
    best_c3 = scored[0][0]
    
    # Time budget for local search
    remaining = DEADLINE - MARGIN - time.time()
    n_search = min(8, len(scored))
    per_search = max(1.0, remaining * 0.6 / n_search)
    
    # Run local search on top candidates
    for i in range(n_search):
        if time.time() > DEADLINE - MARGIN:
            break
        c3_init, x_init = scored[i]
        x_trial, c3_trial = _local_search_batched(x_init, per_search)
        if c3_trial < best_c3:
            best_x = x_trial
            best_c3 = c3_trial
    
    # Spectral refinement
    if time.time() < DEADLINE - MARGIN:
        rem = DEADLINE - MARGIN - time.time()
        x_sp, c3_sp = _spectral_refinement(best_x, rem * 0.4)
        if c3_sp < best_c3:
            best_x = x_sp
            best_c3 = c3_sp
    
    # Final polish: single flip scan
    if time.time() < DEADLINE - MARGIN:
        rem = DEADLINE - MARGIN - time.time()
        x_pol, c3_pol = _local_search_v2(best_x, rem)
        if c3_pol < best_c3:
            best_x = x_pol
            best_c3 = c3_pol
    
    return best_x


def construct():
    x = _construct()
    return np.sign(x).astype(int)
