"""
Rewritten program: Third autocorrelation inequality — signed step function on 400 intervals.

We choose a signed step function h: [-1/4, 1/4] -> {+1, -1} with 400 equal intervals.
The autoconvolution upper bound C3 is related to the peak of |h * h| (autocorrelation)
away from the origin, or more precisely, we want to minimize the maximum sidelobe of
the autocorrelation.

The score is seed/value where value is the upper bound on C3. Lower raw value is better.
The seed program returns np.ones(N) which gives a trivial (poor) bound.

Strategy:
- Use a deterministic construction based on Welch sequences / Barker-like sequences
  or use a greedy / simulated-annealing approach to minimize the peak sidelobe.
- Given the 120s wall-clock limit, we use a combination of:
  1. A fast deterministic heuristic (e.g., based on Legendre sequences or Welch).
  2. A short simulated-annealing / hill-climbing loop with a deadline guard.

The autocorrelation of a step function h with N intervals is:
  R[k] = sum_{j=0}^{N-1-k} h[j] * h[j+k]  for k = 0, ..., N-1
  and by symmetry R[-k] = R[k].

The "upper bound on C3" is typically the maximum absolute value of the off-peak
autocorrelation, i.e., max_{k != 0} |R[k]|, normalized somehow.

Actually, let me reconsider. The task says "improve an autoconvolution upper bound"
and "the score is the upper bound on C3". The evaluator measures the raw value
(lower is better). So we want to minimize the peak sidelobe of the autocorrelation.

For a ±1 sequence of length N, the autocorrelation is:
  R[k] = sum_{j=0}^{N-1-|k|} h[j] * h[j+|k|]

The peak sidelobe is max_{k=1}^{N-1} |R[k]|.

We want to minimize this. The seed program uses all +1's, which gives R[k] = N - k,
so peak sidelobe = N - 1 = 399. That's the "seed value".

A good sequence (like a Barker sequence or similar) would have much lower peak sidelobe.

Let me implement:
1. Start with a Welch-like or Legendre-like sequence.
2. Use simulated annealing to flip bits to reduce the peak sidelobe.
3. Deadline guard to ensure we finish in time.
"""

import numpy as np
import time

N = 400
DEADLINE = time.time() + 110  # 110 seconds to be safe

def _peak_sidelobe(h):
    """Compute the peak sidelobe of the autocorrelation of h."""
    n = len(h)
    peak = 0
    for k in range(1, n):
        # R[k] = sum_{j=0}^{n-1-k} h[j] * h[j+k]
        r = np.dot(h[:n-k], h[k:])
        abs_r = abs(r)
        if abs_r > peak:
            peak = abs_r
    return peak

def _peak_sidelobe_fast(h):
    """Faster peak sidelobe using FFT."""
    n = len(h)
    # Use FFT to compute autocorrelation
    size = 1
    while size < 2 * n:
        size *= 2
    H = np.fft.fft(h, size)
    acf = np.fft.ifft(H * np.conj(H)).real[:n]
    # acf[0] is the peak (DC), acf[k] is R[k]
    # For real sequences, acf is symmetric
    peak = np.max(np.abs(acf[1:]))
    return peak

def _generate_welch_sequence(n):
    """Generate a Welch-like sequence. Welch sequences are optimal for certain lengths.
    For N=400, we can use a modified approach."""
    # Start with a Legendre sequence of length N
    # Legendre sequence: h[j] = 1 if j is a quadratic residue mod p, else -1
    # For prime p, this gives a sequence of length p-1
    # 400 is not prime, so let's use a different approach
    
    # Use a simple construction: alternate blocks
    # Or use a Barker-like pattern
    
    # Let's start with a random seed and use it as a base
    rng = np.random.default_rng(42)
    h = rng.choice([-1, 1], size=n)
    return h

def _hill_climb(h, max_iters=50000, deadline=None):
    """Simple hill climbing: flip one bit at a time if it reduces peak sidelobe."""
    current_peak = _peak_sidelobe_fast(h)
    best_h = h.copy()
    best_peak = current_peak
    
    for i in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Pick a random position to flip
        pos = np.random.randint(N)
        h[pos] *= -1
        
        new_peak = _peak_sidelobe_fast(h)
        
        if new_peak < best_peak:
            best_peak = new_peak
            best_h = h.copy()
        else:
            # Revert
            h[pos] *= -1
    
    return best_h, best_peak

def construct():
    """Construct a signed step function with 400 intervals that minimizes the
    autoconvolution upper bound (peak sidelobe of autocorrelation)."""
    
    # Try several initial constructions and pick the best, then refine
    candidates = []
    
    # Candidate 1: All +1 (seed)
    candidates.append(np.ones(N))
    
    # Candidate 2: Alternating
    candidates.append(np.tile([1, -1], N // 2))
    
    # Candidate 3: Random
    rng = np.random.default_rng(123)
    candidates.append(rng.choice([-1, 1], size=N))
    
    # Candidate 4: Block pattern (4 blocks of 100)
    candidates.append(np.repeat([1, -1, 1, -1], N // 4))
    
    # Candidate 5: Based on a quadratic phase
    idx = np.arange(N)
    candidates.append(np.sign(np.cos(2 * np.pi * idx / N)))
    
    # Evaluate all candidates
    best_h = None
    best_peak = float('inf')
    for h in candidates:
        peak = _peak_sidelobe_fast(h)
        if peak < best_peak:
            best_peak = peak
            best_h = h.copy()
    
    # Refine with hill climbing
    start_time = time.time()
    deadline = start_time + 105  # 105 seconds for hill climbing
    
    # Do multiple short hill climbing runs
    for restart in range(3):
        if time.time() > deadline:
            break
        
        h = best_h.copy()
        # Add some randomness
        if restart > 0:
            flip_positions = np.random.choice(N, size=N // 10, replace=False)
            h[flip_positions] *= -1
        
        best_h, best_peak = _hill_climb(h, max_iters=20000, deadline=deadline)
    
    return best_h

# Test
if __name__ == "__main__":
    h = construct()
    peak = _peak_sidelobe_fast(h)
    print(f"Best peak sidelobe: {peak}")
    print(f"Length: {len(h)}")
