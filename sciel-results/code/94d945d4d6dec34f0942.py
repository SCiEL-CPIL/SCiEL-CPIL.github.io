"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np


def construct():
    # Try several candidate sequences and pick the one with minimum
    # max|autoconv| / (sum f)^2
    
    candidates = []
    
    # Candidate 1: all ones (seed)
    candidates.append(np.ones(N))
    
    # Candidate 2: alternating +1/-1 with slight bias
    alt = np.ones(N)
    alt[1::2] = -1
    candidates.append(alt)
    
    # Candidate 3: random ±1
    rng = np.random.default_rng(42)
    rand = rng.choice([-1, 1], size=N)
    candidates.append(rand)
    
    # Candidate 4: low-frequency cosine
    t = np.arange(N)
    cos_seq = 1 + 0.5 * np.cos(2 * np.pi * t / N)
    candidates.append(cos_seq)
    
    # Candidate 5: square wave (block of +1 then -1)
    sq = np.ones(N)
    sq[N//2:] = -1
    candidates.append(sq)
    
    # Candidate 6: triangular wave
    tri = np.zeros(N)
    half = N // 2
    tri[:half] = np.linspace(1, -1, half)
    tri[half:] = np.linspace(-1, 1, N - half)
    candidates.append(tri)
    
    best = None
    best_score = np.inf
    
    for f in candidates:
        s = np.sum(f)
        if abs(s) < 1e-10:
            continue  # skip zero-sum to avoid div by zero
        autoconv = np.convolve(f, f, mode='full')
        max_abs = np.max(np.abs(autoconv))
        score = max_abs / (s * s)
        if score < best_score:
            best_score = score
            best = f
    
    if best is None:
        return np.ones(N)
    
    return best
# EVOLVE-BLOCK-END
