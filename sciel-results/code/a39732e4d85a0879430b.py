"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _psl(seq):
    """Compute peak sidelobe level: max |acf[1:]|."""
    n = len(seq)
    n_fft = 1
    while n_fft < 2 * n:
        n_fft *= 2
    f = np.fft.fft(seq, n=n_fft)
    acf = np.fft.ifft(f * np.conj(f)).real[:n]
    return float(np.max(np.abs(acf[1:])))

def _psl_sum_squared(seq):
    """Sum of squared off-peak autocorrelation values."""
    n = len(seq)
    n_fft = 1
    while n_fft < 2 * n:
        n_fft *= 2
    f = np.fft.fft(seq, n=n_fft)
    acf = np.fft.ifft(f * np.conj(f)).real[:n]
    return float(np.sum(acf[1:]**2))

def _flip_bit(seq, i):
    """Flip the sign at position i."""
    seq[i] = -seq[i]

def _evaluate_flip(seq, i, current_psl):
    """Try flipping position i and return new PSL."""
    old = seq[i]
    seq[i] = -old
    new_psl = _psl(seq)
    seq[i] = old
    return new_psl

def _hill_climb(seq, deadline):
    """Hill climbing: flip single bits to reduce PSL."""
    current_psl = _psl(seq)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        best_psl = current_psl
        best_idx = -1
        for i in range(N):
            if time.time() > deadline:
                break
            new_psl = _evaluate_flip(seq, i, current_psl)
            if new_psl < best_psl - 1e-12:
                best_psl = new_psl
                best_idx = i
        if best_idx >= 0:
            seq[best_idx] = -seq[best_idx]
            current_psl = best_psl
            improved = True
    return seq, current_psl

def _pair_flip_search(seq, deadline):
    """Try pairs of flips to escape local minima."""
    current_psl = _psl(seq)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        best_psl = current_psl
        best_pair = (-1, -1)
        # Sample pairs rather than all N^2/2
        for _ in range(500):
            if time.time() > deadline:
                break
            i = np.random.randint(0, N)
            j = np.random.randint(0, N)
            if i == j:
                continue
            seq[i] = -seq[i]
            seq[j] = -seq[j]
            new_psl = _psl(seq)
            seq[i] = -seq[i]
            seq[j] = -seq[j]
            if new_psl < best_psl - 1e-12:
                best_psl = new_psl
                best_pair = (i, j)
        if best_pair != (-1, -1):
            i, j = best_pair
            seq[i] = -seq[i]
            seq[j] = -seq[j]
            current_psl = best_psl
            improved = True
    return seq, current_psl

def _sine_initialization(freq, phase):
    """Generate a sign sequence from a sine wave."""
    t = np.linspace(-0.25, 0.25, N, endpoint=False)
    vals = np.sin(2 * np.pi * freq * t + phase)
    signs = np.sign(vals)
    signs[signs == 0] = 1
    return signs.astype(np.float64)

def _greedy_construct(deadline):
    """Greedy bit-by-bit construction."""
    seq = np.ones(1)
    for i in range(1, N):
        if time.time() > deadline:
            break
        # Try +1
        seq_plus = np.append(seq, 1.0)
        psl_plus = _psl(seq_plus)
        # Try -1
        seq_minus = np.append(seq, -1.0)
        psl_minus = _psl(seq_minus)
        if psl_plus <= psl_minus:
            seq = seq_plus
        else:
            seq = seq_minus
    if len(seq) < N:
        seq = np.concatenate([seq, np.ones(N - len(seq))])
    return seq

def construct():
    """Construct a signed step function to minimize the autoconvolution upper bound C3."""
    t_start = time.time()
    deadline = t_start + 140  # Safety margin before 150s limit
    
    rng = np.random.default_rng(42)
    best_seq = None
    best_psl = float('inf')
    
    # Strategy 1: Sine-based initializations with various frequencies/phases
    for freq in [1, 2, 3, 4, 5, 7, 10]:
        for phase in [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi]:
            if time.time() > deadline - 10:
                break
            seq = _sine_initialization(freq, phase)
            psl = _psl(seq)
            if psl < best_psl:
                best_psl = psl
                best_seq = seq.copy()
    
    # Strategy 2: Random initializations
    for _ in range(20):
        if time.time() > deadline - 10:
            break
        seq = rng.choice([-1.0, 1.0], size=N)
        psl = _psl(seq)
        if psl < best_psl:
            best_psl = psl
            best_seq = seq.copy()
    
    # Strategy 3: Greedy construction
    if time.time() < deadline - 20:
        seq = _greedy_construct(deadline - 10)
        psl = _psl(seq)
        if psl < best_psl:
            best_psl = psl
            best_seq = seq.copy()
    
    if best_seq is None:
        best_seq = np.ones(N)
        best_psl = _psl(best_seq)
    
    # Phase 2: Hill climbing from best starting point
    if time.time() < deadline - 5:
        best_seq, best_psl = _hill_climb(best_seq.copy(), deadline - 5)
    
    # Phase 3: Pair flip search to escape local minima
    if time.time() < deadline - 2:
        best_seq, best_psl = _pair_flip_search(best_seq.copy(), deadline - 2)
    
    # Phase 4: Another round of hill climbing
    if time.time() < deadline - 1:
        best_seq, best_psl = _hill_climb(best_seq.copy(), deadline - 1)
    
    return best_seq
# EVOLVE-BLOCK-END
