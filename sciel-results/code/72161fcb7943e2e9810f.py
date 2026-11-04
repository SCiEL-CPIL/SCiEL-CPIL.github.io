"""Rewritten for the third autocorrelation inequality task."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _autoconv_max_abs(f):
    """Compute max absolute value of autoconvolution (excluding self-lag 0)."""
    n = len(f)
    size = 2 * n
    F = np.fft.fft(f, n=size)
    conv = np.fft.ifft(F * F).real[:n]
    # Exclude lag 0 (self-convolution)
    return np.max(np.abs(conv[1:]))


def _thue_morse(n):
    """Generate Thue-Morse sequence of length n as ±1 values."""
    # Thue-Morse: t(n) = parity of number of 1s in binary representation of n
    # Convert to ±1: t(n) even -> +1, t(n) odd -> -1
    tm = np.zeros(n)
    for i in range(n):
        # Count set bits
        x = i
        count = 0
        while x:
            count += x & 1
            x >>= 1
        tm[i] = 1.0 if count % 2 == 0 else -1.0
    return tm


def _cyclic_shift(f, shift):
    """Cyclically shift the array by shift positions."""
    return np.roll(f, shift)


def _greedy_refine(f, deadline, rng=None):
    """Iterative greedy single-flip refinement until no improvement or deadline."""
    current_score = _autoconv_max_abs(f)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(N):
            if time.time() > deadline:
                return f, current_score
            old_val = f[i]
            f[i] = -old_val
            new_score = _autoconv_max_abs(f)
            if new_score < current_score - 1e-12:
                current_score = new_score
                improved = True
            else:
                f[i] = old_val
    return f, current_score


def _simulated_annealing(f, deadline, rng, T_init=2.0, cooling=0.9995, T_min=0.1):
    """Simulated annealing with single flips."""
    current_score = _autoconv_max_abs(f)
    T = T_init
    no_improve = 0
    max_no_improve = 200
    
    while time.time() < deadline:
        i = rng.randint(N)
        f[i] *= -1.0
        new_score = _autoconv_max_abs(f)
        
        if new_score < current_score:
            current_score = new_score
            no_improve = 0
        else:
            if T > T_min and rng.rand() < np.exp(-(new_score - current_score) / T):
                current_score = new_score
            else:
                f[i] *= -1.0  # revert
            no_improve += 1
        
        T *= cooling
        if T < T_min:
            T = T_min
        
        if no_improve > max_no_improve:
            # Restart with perturbation
            for _ in range(10):
                j = rng.randint(N)
                f[j] *= -1.0
            current_score = _autoconv_max_abs(f)
            no_improve = 0
            T = T_init * 0.5
    
    return f, current_score


def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s wall-clock limit
    
    best_f = None
    best_score = np.inf
    
    # Phase 1: Try Thue-Morse based initializations with various cyclic shifts
    # The Thue-Morse sequence has flat autocorrelation properties
    tm_base = _thue_morse(N)
    
    # Try different cyclic shifts of Thue-Morse
    for shift in range(0, N, 10):  # Sample shifts
        if time.time() > deadline - 20:
            break
        f = _cyclic_shift(tm_base, shift).copy()
        score = _autoconv_max_abs(f)
        if score < best_score:
            best_score = score
            best_f = f.copy()
    
    # Also try Thue-Morse with some random sign flips
    rng = np.random.RandomState(42)
    for trial in range(20):
        if time.time() > deadline - 20:
            break
        f = _cyclic_shift(tm_base, trial * 7 % N).copy()
        # Flip a few random positions
        for _ in range(5):
            f[rng.randint(N)] *= -1.0
        score = _autoconv_max_abs(f)
        if score < best_score:
            best_score = score
            best_f = f.copy()
    
    # Phase 2: Simulated annealing from best Thue-Morse init
    if best_f is not None and time.time() < deadline - 30:
        f = best_f.copy()
        rng_sa = np.random.RandomState(789)
        time_per_sa = min(30.0, deadline - time.time() - 30.0)
        f, score = _simulated_annealing(f, time.time() + time_per_sa, rng_sa)
        if score < best_score:
            best_score = score
            best_f = f.copy()
    
    # Phase 3: Greedy refinement on best solution
    if best_f is not None and time.time() < deadline - 10:
        f = best_f.copy()
        f, score = _greedy_refine(f, deadline - 5.0)
        if score < best_score:
            best_score = score
            best_f = f.copy()
    
    # Phase 4: If time remains, try a few random restarts with greedy
    remaining = deadline - 5.0 - time.time()
    if remaining > 15.0:
        for restart in range(3):
            if time.time() > deadline - 10.0:
                break
            rng_r = np.random.RandomState(10000 + restart * 999)
            f = rng_r.choice([-1.0, 1.0], size=N)
            # Quick greedy
            f, score = _greedy_refine(f, time.time() + min(10.0, deadline - 10.0 - time.time()))
            if score < best_score:
                best_score = score
                best_f = f.copy()
    
    # Phase 5: Final greedy polish
    if best_f is not None and time.time() < deadline - 2.0:
        f = best_f.copy()
        f, score = _greedy_refine(f, deadline - 1.0)
        if score < best_score:
            best_score = score
            best_f = f.copy()
    
    if best_f is None:
        return np.ones(N)
    
    return best_f
# EVOLVE-BLOCK-END
