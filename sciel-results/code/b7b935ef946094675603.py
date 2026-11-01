"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

DEADLINE = 120.0
TIME_MARGIN = 10.0  # safety margin in seconds


def _de_bruijn_sequence(n, k=2):
    """
    Generate a de Bruijn sequence B(k, n) as a list of symbols in {0, 1}.
    Uses the prefer-one algorithm (also known as the FKM algorithm).
    The sequence has length k^n and contains every length-n word over k symbols exactly once.
    """
    a = [0] * (k * (n + 1))
    sequence = []

    def db(t, p):
        if t > n:
            if n % p == 0:
                sequence.extend(a[1:p + 1])
        else:
            a[t] = a[t - p]
            db(t + 1, p)
            for j in range(a[t - p] + 1, k):
                a[t] = j
                db(t + 1, t)

    db(1, 1)
    return sequence


def _de_bruijn_to_signed(x_len, seed_offset=0):
    """
    Convert a de Bruijn sequence to a signed step function of length x_len.
    Maps 0 -> -1, 1 -> +1.
    If the de Bruijn sequence is shorter than x_len, we tile it.
    If longer, we take the first x_len elements.
    """
    # Choose n such that 2^n >= x_len for a good balance
    # For x_len=400, we need 2^n >= 400, so n=9 gives 512
    # But let's try different n values and pick the best one
    best_x = None
    best_score = float('inf')
    
    for n in range(6, 12):
        seq = _de_bruijn_sequence(n, k=2)
        # Convert to signed: 0 -> -1, 1 -> +1
        signed = np.array([2 * s - 1 for s in seq], dtype=float)
        
        # Tile or truncate to get x_len elements
        if len(signed) >= x_len:
            x_candidate = signed[:x_len]
        else:
            # Tile the sequence
            repeats = (x_len + len(signed) - 1) // len(signed)
            x_candidate = np.tile(signed, repeats)[:x_len]
        
        # Apply a phase shift based on seed_offset for diversity
        shift = seed_offset % len(x_candidate)
        x_candidate = np.roll(x_candidate, shift)
        
        # Compute C3 for this candidate
        c3 = _compute_c3(x_candidate)
        if c3 < best_score:
            best_score = c3
            best_x = x_candidate
    
    return best_x


def _compute_c3(x):
    """
    Compute the upper bound C3 for a signed step function x of length N.
    
    The autoconvolution upper bound C3 is related to the maximum absolute
    value of the autocorrelation sidelobes relative to the main lobe.
    
    For a signed step function (values in {-1, +1}), the autocorrelation is:
      R[k] = sum_{i=0}^{N-1-k} x[i] * x[i+k]  for k = 0, ..., N-1
    
    The "upper bound on C3" is typically defined as:
      C3 = max_{k != 0} |R[k]| / R[0]
    
    But since we're dealing with a step function on [-1/4, 1/4] with 400 equal
    intervals, and the task mentions "autoconvolution upper bound", let's define
    it more carefully.
    
    Actually, for a step function f(x) with heights x[i] on intervals, the
    autoconvolution (f * f)(t) = integral f(s) f(t-s) ds.
    
    For a step function with N equal intervals of width w = (1/4 - (-1/4))/N = 1/(2N),
    the autoconvolution at lag k*w is:
      (f*f)(k*w) = w * sum_{i} x[i] x[i+k]  (with appropriate boundary handling)
    
    The peak of the autoconvolution is at k=0: (f*f)(0) = w * sum x[i]^2 = w * N (for ±1)
    
    The "upper bound on C3" likely refers to the maximum sidelobe level relative
    to the main lobe, or some normalized quantity.
    
    Given the task says "score is the upper bound on C3" and "lower is better",
    I'll compute:
      C3 = max_{k=1,...,N-1} |R[k]| / R[0]
    
    where R[k] is the autocorrelation at lag k.
    """
    # Compute autocorrelation using FFT for efficiency
    n_fft = 1 << (2 * N - 1).bit_length()  # next power of 2 for linear correlation
    F = np.fft.rfft(x, n=n_fft)
    power = F * np.conj(F)
    R = np.fft.irfft(power, n=n_fft)[:2 * N - 1]
    # R[0] is the zero-lag correlation = sum x[i]^2
    # R[1:N] are the positive lags
    # R[N:2N-1] are the negative lags (symmetric)
    
    R0 = R[0]
    if R0 == 0:
        return 1.0  # degenerate case
    
    # Maximum absolute sidelobe
    sidelobes = np.concatenate([R[1:N], R[N:2*N-1]])
    max_sidelobe = np.max(np.abs(sidelobes))
    
    C3 = max_sidelobe / R0
    return C3


def _flip_random_positions(x, num_flips):
    """Flip a random subset of positions in x."""
    x = x.copy()
    if num_flips == 0:
        return x
    indices = random.sample(range(N), min(num_flips, N))
    for i in indices:
        x[i] = -x[i]
    return x


def _local_search(x, max_iter=5000, time_limit=None):
    """
    Local search: try flipping individual positions and accept if it improves C3.
    """
    start_time = time.time()
    current_c3 = _compute_c3(x)
    best_x = x.copy()
    best_c3 = current_c3
    no_improve_count = 0
    max_no_improve = 100
    
    for iteration in range(max_iter):
        # Check time
        if time_limit is not None and time.time() - start_time > time_limit:
            break
        # Check global deadline
        if time.time() > DEADLINE - TIME_MARGIN:
            break
        
        # Try a random flip
        idx = random.randint(0, N - 1)
        x_new = x.copy()
        x_new[idx] = -x_new[idx]
        new_c3 = _compute_c3(x_new)
        
        if new_c3 < current_c3 - 1e-12:
            x = x_new
            current_c3 = new_c3
            no_improve_count = 0
            if current_c3 < best_c3:
                best_x = x.copy()
                best_c3 = current_c3
        else:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                break
    
    return best_x, best_c3


def _simulated_annealing(x, max_iter=20000, T_start=1.0, T_end=0.01, time_limit=None):
    """
    Simulated annealing to escape local minima.
    """
    start_time = time.time()
    current_c3 = _compute_c3(x)
    best_x = x.copy()
    best_c3 = current_c3
    T = T_start
    
    for iteration in range(max_iter):
        # Check time
        if time_limit is not None and time.time() - start_time > time_limit:
            break
        if time.time() > DEADLINE - TIME_MARGIN:
            break
        
        # Annealing schedule
        T = T_start * (T_end / T_start) ** (iteration / max_iter)
        
        # Try a random flip
        idx = random.randint(0, N - 1)
        x_new = x.copy()
        x_new[idx] = -x_new[idx]
        new_c3 = _compute_c3(x_new)
        
        delta = new_c3 - current_c3
        if delta < 0 or random.random() < np.exp(-delta / max(T, 1e-10)):
            x = x_new
            current_c3 = new_c3
            if current_c3 < best_c3:
                best_x = x.copy()
                best_c3 = current_c3
    
    return best_x, best_c3


def _construct():
    """
    Main construction function that searches for a good signed step function.
    Uses de Bruijn sequences as deterministic starting points, then refines with local search.
    """
    start_time = time.time()
    random.seed(42)
    np.random.seed(42)
    
    best_x = None
    best_c3 = float('inf')
    
    # Strategy 1: Try multiple de Bruijn-based initializations with different phase shifts
    for phase in range(0, 8):
        if time.time() > DEADLINE - TIME_MARGIN:
            break
        x_db = _de_bruijn_to_signed(N, seed_offset=phase)
        if x_db is not None:
            c3_db = _compute_c3(x_db)
            if c3_db < best_c3:
                best_c3 = c3_db
                best_x = x_db.copy()
    
    # If de Bruijn didn't produce anything, fall back to all-ones
    if best_x is None:
        best_x = np.ones(N)
        best_c3 = _compute_c3(best_x)
    
    # Strategy 2: Local search from best de Bruijn solution
    if time.time() < DEADLINE - TIME_MARGIN:
        x1, c3_1 = _local_search(best_x.copy(), max_iter=5000, 
                                  time_limit=min(30, DEADLINE - TIME_MARGIN - (time.time() - start_time)))
        if c3_1 < best_c3:
            best_x = x1
            best_c3 = c3_1
    
    # Check time
    if time.time() > DEADLINE - TIME_MARGIN:
        return best_x
    
    # Strategy 3: Try a few random initializations + local search
    for trial in range(3):
        if time.time() > DEADLINE - TIME_MARGIN:
            break
        x_init = np.random.choice([-1, 1], size=N)
        x2, c3_2 = _local_search(x_init, max_iter=2000, 
                                  time_limit=min(15, DEADLINE - TIME_MARGIN - (time.time() - start_time)))
        if c3_2 < best_c3:
            best_x = x2
            best_c3 = c3_2
    
    # Check time
    if time.time() > DEADLINE - TIME_MARGIN:
        return best_x
    
    # Strategy 4: Simulated annealing from best solution
    if time.time() < DEADLINE - TIME_MARGIN:
        remaining = DEADLINE - TIME_MARGIN - (time.time() - start_time)
        if remaining > 5:
            x3, c3_3 = _simulated_annealing(best_x, max_iter=10000, T_start=0.5, T_end=0.01,
                                             time_limit=remaining)
            if c3_3 < best_c3:
                best_x = x3
                best_c3 = c3_3
    
    # Final local search polish
    if time.time() < DEADLINE - TIME_MARGIN:
        remaining = DEADLINE - TIME_MARGIN - (time.time() - start_time)
        if remaining > 2:
            x4, c3_4 = _local_search(best_x, max_iter=3000, time_limit=remaining)
            if c3_4 < best_c3:
                best_x = x4
                best_c3 = c3_4
    
    return best_x

def construct():
    """
    Entry point: return a sequence of signed step heights (±1) of length N=400.
    """
    x = _construct()
    # Ensure output is a numpy array of ±1 values
    return np.sign(x).astype(int)
# EVOLVE-BLOCK-END
