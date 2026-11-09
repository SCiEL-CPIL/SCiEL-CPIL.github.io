"""Improved program: LFSR m-sequence design with local search polish on signed ±1 step functions."""
import numpy as np
import time

N = 400

def compute_autocorrelation_bound(f):
    """Compute an upper bound on C3 based on autocorrelation sidelobes."""
    # Linear autocorrelation via FFT
    nfft = 1 << (2 * N - 1).bit_length()
    F = np.fft.rfft(f, n=nfft)
    acf_full = np.fft.irfft(F * np.conj(F), n=nfft)
    # Linear autocorrelation: acf[k] = sum_{i=0}^{N-1-k} f[i]*f[i+k] for k >= 0
    acf = np.zeros(2 * N - 1)
    acf[:N] = acf_full[:N]
    acf[N:] = acf_full[nfft - N + 1:nfft][::-1]
    
    # The "third autocorrelation inequality" upper bound on C3
    # C3 relates to the L2 norm of the autocorrelation sidelobes
    # Use the normalized sidelobe energy as the metric
    peak = acf[0]  # This is sum(f^2) = N for ±1 signals
    if abs(peak) < 1e-12:
        return np.sqrt(np.sum(acf[1:] ** 2))
    sidelobe_energy = np.sum(acf[1:] ** 2)
    return np.sqrt(sidelobe_energy / (peak * peak))

def lfsr_sequence(poly, seed, length):
    """Generate an LFSR sequence of given length using the given polynomial and seed."""
    # poly is a list of tap positions (0-indexed from LSB)
    # seed is the initial state
    seq = np.zeros(length, dtype=np.float64)
    state = seed
    for i in range(length):
        seq[i] = 1.0 if (state & 1) else -1.0
        # Compute next state
        feedback = 0
        for tap in poly:
            feedback ^= (state >> tap) & 1
        state = (state >> 1) | (feedback << (len(poly) - 1))
        # Mask to keep only the relevant bits
        mask = (1 << len(poly)) - 1
        state &= mask
    return seq

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 150s kill
    
    rng = np.random.default_rng(42)
    
    # Try multiple LFSR polynomial candidates
    # For a 9-bit LFSR, we need maximal length sequences
    # Common primitive polynomials for degree 9:
    # x^9 + x^4 + 1, x^9 + x^5 + 1, x^9 + x^6 + 1, etc.
    # Taps for x^9 + x^4 + 1: [0, 4] (feedback from bit 0 and bit 4)
    # Taps for x^9 + x^5 + 1: [0, 5]
    # Taps for x^9 + x^6 + 1: [0, 6]
    
    candidates = []
    
    # Generate sequences from various LFSR configurations
    polys = [
        [0, 4],  # x^9 + x^4 + 1
        [0, 5],  # x^9 + x^5 + 1
        [0, 6],  # x^9 + x^6 + 1
        [0, 3, 5],  # x^9 + x^3 + x^5 + 1
        [0, 2, 6],  # x^9 + x^2 + x^6 + 1
        [0, 1, 7],  # x^9 + x + x^7 + 1
    ]
    
    for poly in polys:
        for seed in range(1, 1 << len(poly)):
            seq = lfsr_sequence(poly, seed, N)
            # Also try sign-flipped variant
            seq_neg = -seq
            candidates.append(seq)
            candidates.append(seq_neg)
    
    # Also add some random initializations
    for _ in range(20):
        candidates.append(rng.choice([-1.0, 1.0], size=N))
    
    # Find the best candidate
    best_f = None
    best_bound = np.inf
    
    for f in candidates:
        bound = compute_autocorrelation_bound(f)
        if bound < best_bound:
            best_bound = bound
            best_f = f.copy()
    
    # Local search: flip random bits to improve
    current_f = best_f.copy()
    current_bound = best_bound
    
    while time.time() < deadline:
        # Flip a random position
        idx = rng.integers(0, N)
        f_new = current_f.copy()
        f_new[idx] = -f_new[idx]
        
        new_bound = compute_autocorrelation_bound(f_new)
        
        if new_bound < current_bound:
            current_f = f_new
            current_bound = new_bound
            if new_bound < best_bound:
                best_bound = new_bound
                best_f = f_new.copy()
        else:
            # Occasionally accept worse solutions to escape local minima
            if rng.random() < 0.1:
                current_f = f_new
                current_bound = new_bound
    
    return best_f
