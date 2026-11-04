"""Structural redesign: genetic-annealing search over sign patterns, seeded by Thue-Morse."""
import numpy as np
import time

N = 400

def _thue_morse(n):
    """Generate Thue-Morse sequence of length n as ±1 array."""
    # Thue-Morse: t(k) = parity of popcount(k)
    # Map 0->1, 1->-1 (or vice versa; sign doesn't matter for autocorrelation magnitude)
    km = np.arange(n)
    # Compute popcount parity efficiently
    # Use bit operations: XOR-shift trick
    t = np.zeros(n, dtype=np.uint8)
    for i in range(n):
        # parity of bit count
        x = i
        c = 0
        while x:
            c += 1
            x &= x - 1
        t[i] = c & 1
    # Convert to ±1
    return 1.0 - 2.0 * t.astype(np.float64)

def _autocorr_peak(s):
    """Compute max |autocorrelation| over non-zero lags for a ±1 sequence.
    Uses FFT-based autocorrelation."""
    n = len(s)
    # Zero-pad to next power of 2 for FFT
    m = 1
    while m < 2 * n:
        m <<= 1
    f = np.fft.rfft(s, m)
    ac = np.fft.irfft(f * np.conj(f), m)[:n]
    # ac[0] is the DC term = sum of squares = n
    # We want the max over lags 1..n-1
    if n > 1:
        return np.max(np.abs(ac[1:]))
    return 0.0

def _score(s):
    """The C3 upper bound: we use the peak sidelobe autocorrelation as a proxy.
    Lower is better."""
    return _autocorr_peak(s)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill

    # Initialize with Thue-Morse
    s = _thue_morse(N)
    best = s.copy()
    best_score = _score(best)

    # Genetic algorithm with simulated annealing
    rng = np.random.default_rng(42)
    
    pop_size = 60
    population = [best.copy() for _ in range(pop_size)]
    # Diversify initial population
    for i in range(1, pop_size):
        idx = rng.choice(N, size=rng.integers(20, 80), replace=False)
        population[i][idx] *= -1  # flip some signs

    scores = [_score(p) for p in population]
    best_idx = int(np.argmin(scores))
    best = population[best_idx].copy()
    best_score = scores[best_idx]

    gen = 0
    while time.time() < deadline:
        gen += 1
        # Select parents (tournament)
        new_pop = [best.copy()]  # elitism
        for _ in range(pop_size - 1):
            # Tournament selection
            cands = rng.choice(pop_size, size=3, replace=False)
            c_scores = [scores[c] for c in cands]
            p1 = cands[int(np.argmin(c_scores))]
            cands2 = rng.choice(pop_size, size=3, replace=False)
            c_scores2 = [scores[c] for c in cands2]
            p2 = cands2[int(np.argmin(c_scores2))]
            
            # Crossover: uniform
            child = population[p1].copy()
            mask = rng.random(N) < 0.5
            child[mask] = population[p2][mask]
            
            # Mutation: flip a few random signs
            n_mut = rng.integers(2, 15)
            mut_idx = rng.choice(N, size=n_mut, replace=False)
            child[mut_idx] *= -1
            
            new_pop.append(child)
        
        population = new_pop
        scores = [_score(p) for p in population]
        
        # Update best
        cur_best_idx = int(np.argmin(scores))
        if scores[cur_best_idx] < best_score:
            best_score = scores[cur_best_idx]
            best = population[cur_best_idx].copy()
        
        # Occasionally re-inject Thue-Morse variants to escape local minima
        if gen % 10 == 0:
            tm = _thue_morse(N)
            idx = rng.choice(N, size=rng.integers(5, 30), replace=False)
            tm[idx] *= -1
            population[-1] = tm
            scores[-1] = _score(tm)
            if scores[-1] < best_score:
                best_score = scores[-1]
                best = tm.copy()

    return best

# Pre-compute to ensure construct() is fast on first call if needed
# Actually construct() does the work; keep it as is.
