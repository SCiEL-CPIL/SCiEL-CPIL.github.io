"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _autocorr_peak_excl_lag0(s):
    """Peak of |linear autocorrelation| of s (±1, length N), excluding lag 0."""
    n = len(s)
    if n <= 1:
        return 0.0
    size = 2 * n
    S = np.fft.rfft(s, size)
    ac = np.fft.irfft(S * np.conj(S), size)
    # ac[0] is lag 0, we want max of |ac[1:n]|
    if n > 1:
        return float(np.max(np.abs(ac[1:n])))
    return 0.0


def _legendre_sequence(n, p):
    """
    Generate a Legendre sequence of length n using prime p.
    Legendre symbol (i/p) is 1 if i is a quadratic residue mod p, -1 otherwise.
    We map 0 (when i%p==0) to +1 for simplicity, or skip.
    For i=0..n-1, compute (i mod p / p).
    """
    seq = np.ones(n, dtype=np.float64)
    # Compute Legendre symbols for 1..p-1
    # (a/p) = a^((p-1)/2) mod p
    half_p = (p - 1) // 2
    for i in range(n):
        a = i % p
        if a == 0:
            seq[i] = 1.0  # Convention for 0
        else:
            ls = pow(a, half_p, p)
            if ls == 1:
                seq[i] = 1.0
            else:
                seq[i] = -1.0
    return seq


def _mian_sequence(n):
    """
    Generate a Mian-Chowla-like sequence or similar greedy low autocorrelation sequence.
    Actually, let's just use a simple greedy construction for ±1 sequences.
    Or better, stick to Legendre and Thue-Morse as primary deterministic seeds.
    """
    pass


def _thue_morse_sequence(n):
    """
    Generate the Thue-Morse sequence of length n.
    Map even parity to +1, odd to -1.
    """
    t = np.ones(n, dtype=np.float64)
    for i in range(n):
        count = bin(i).count('1')
        if count % 2 != 0:
            t[i] = -1.0
    return t


def _fast_greedy(h, time_limit=5.0):
    """
    Fast greedy local search.
    1. Flip signs of individual bits.
    Uses random ordering to escape local minima.
    """
    n = len(h)
    best_h = h.copy()
    best_score = _autocorr_peak_excl_lag0(best_h)
    start = time.time()
    
    while time.time() - start < time_limit:
        improved = False
        order = np.random.permutation(n)
        for i in order:
            trial = best_h.copy()
            trial[i] = -trial[i]
            s = _autocorr_peak_excl_lag0(trial)
            if s < best_score - 1e-12:
                best_h = trial
                best_score = s
                improved = True
                break
        
        if not improved:
            break
            
    return best_h, best_score


def _simulated_annealing(h, time_limit=10.0):
    """
    Simulated annealing to escape local minima.
    """
    n = len(h)
    current_h = h.copy()
    current_score = _autocorr_peak_excl_lag0(current_h)
    best_h = current_h.copy()
    best_score = current_score
    
    start = time.time()
    T = 5.0
    cooling = 0.995
    min_T = 0.1
    
    while T > min_T and time.time() - start < time_limit:
        # Pick a random move
        i = np.random.randint(n)
        trial = current_h.copy()
        trial[i] = -trial[i]
        
        new_score = _autocorr_peak_excl_lag0(trial)
        delta = new_score - current_score
        
        if delta < 0 or np.random.random() < np.exp(-delta / T):
            current_h = trial
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_h = current_h.copy()
        
        T *= cooling
        
    return best_h, best_score


def construct():
    """
    Construct a signed step function with N=400 intervals that minimizes
    the autoconvolution upper bound.
    
    Strategy:
    1. Initialize with Legendre sequence (deterministic, low additive energy).
    2. Also try Thue-Morse and random sequences.
    3. Use fast greedy local search to refine.
    4. Use simulated annealing to escape local minima.
    5. Return the best solution found within the time budget.
    """
    deadline = time.time() + 140  # Safety margin before 150s kill
    
    best_h = None
    best_score = float('inf')
    
    # Phase 1: Legendre sequence initialization
    # Find a prime close to 400, e.g., 401
    p = 401
    h_leg = _legendre_sequence(N, p)
    h_opt, score = _fast_greedy(h_leg, time_limit=5.0)
    if time.time() < deadline - 10:
        h_opt, score = _simulated_annealing(h_opt, time_limit=10.0)
    
    if score < best_score:
        best_score = score
        best_h = h_opt.copy()
    
    # Phase 2: Thue-Morse initialization
    h_tm = _thue_morse_sequence(N)
    h_opt, score = _fast_greedy(h_tm, time_limit=5.0)
    if time.time() < deadline - 10:
        h_opt, score = _simulated_annealing(h_opt, time_limit=10.0)
    
    if score < best_score:
        best_score = score
        best_h = h_opt.copy()
    
    # Phase 3: Random full sequences
    for trial in range(5):
        if time.time() > deadline - 10:
            break
        h_rand = np.random.choice([-1, 1], size=N).astype(np.float64)
        h_opt, score = _fast_greedy(h_rand, time_limit=3.0)
        if time.time() < deadline - 5:
            h_opt, score = _simulated_annealing(h_opt, time_limit=5.0)
        
        if score < best_score:
            best_score = score
            best_h = h_opt.copy()
    
    # Return as integer array to match expected interface if needed, 
    # but the task says "sequence of signed step heights", so float is fine.
    # The evaluator likely expects ±1 values.
    return best_h
# EVOLVE-BLOCK-END
