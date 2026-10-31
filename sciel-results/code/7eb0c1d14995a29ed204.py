"""
Combined program: m-sequence (Gold code) construction with local sign-flip refinement.
The m-sequence provides a structurally optimal starting point with flat autocorrelation
sidelobes, which we then polish with local sign flips to minimize max|autoconv|/(sum)^2.
"""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.signal import fftconvolve

_START_TIME = time.time()
_TIME_LIMIT = 115.0  # safety margin before 120s limit


def _autoconv_max_abs(x):
    """Compute maximum absolute value of autoconvolution of x."""
    ac = fftconvolve(x, x, mode='full')
    return np.max(np.abs(ac))


def _squared_integral(x):
    """Compute squared integral (sum of x) squared."""
    s = np.sum(x)
    return s * s


def _score(x):
    """Compute the C3' score: max_abs_autoconv / squared_integral. Lower is better."""
    sq_int = _squared_integral(x)
    if abs(sq_int) < 1e-12:
        return float('inf')
    return _autoconv_max_abs(x) / sq_int


def _m_sequence(n, seed_poly=None):
    """
    Generate an m-sequence (maximum length LFSR sequence) of length n.
    Uses a primitive polynomial over GF(2). Returns ±1 values.
    
    For a 9-bit LFSR with primitive polynomial x^9 + x^4 + 1:
    The period is 2^9 - 1 = 511, which is >= 400.
    """
    if seed_poly is None:
        # Primitive polynomial x^9 + x^4 + 1 (taps at positions 0, 4, 8)
        # This gives period 511
        poly_taps = [0, 4, 8]
        degree = 9
    else:
        poly_taps = seed_poly
        degree = max(poly_taps) + 1
    
    period = (1 << degree) - 1
    
    # Generate the full period using LFSR
    # State is a list of degree bits
    state = [0] * degree
    state[0] = 1  # Non-zero initial state
    
    seq = []
    for i in range(period):
        # Output the LSB
        seq.append(state[0])
        
        # Compute feedback
        feedback = 0
        for tap in poly_taps:
            feedback ^= state[tap]
        
        # Shift right (or left depending on convention)
        # Shift state right, put feedback in MSB
        for j in range(degree - 1, 0, -1):
            state[j] = state[j - 1]
        state[0] = feedback
    
    # Convert to ±1: 0 -> 1, 1 -> -1 (or vice versa)
    # We want the sequence to have a good sum for maximizing squared_integral
    # Map 0 -> 1, 1 -> -1
    seq = np.array([1.0 if b == 0 else -1.0 for b in seq])
    
    # Take first n elements
    if len(seq) >= n:
        return seq[:n]
    else:
        # If period is too short, tile it (shouldn't happen with our choice)
        repeats = (n // len(seq)) + 1
        seq = np.tile(seq, repeats)[:n]
        return seq


def _m_sequence_variants(n):
    """
    Generate several m-sequence variants using different primitive polynomials
    or phase shifts, to provide diverse starting points.
    """
    variants = []
    
    # Different primitive polynomials for degree 9
    # x^9 + x^4 + 1
    variants.append(_m_sequence(n, [0, 4, 8]))
    
    # x^9 + x^5 + 1
    variants.append(_m_sequence(n, [0, 5, 8]))
    
    # x^9 + x^6 + 1
    variants.append(_m_sequence(n, [0, 6, 8]))
    
    # Phase shifts of the first m-sequence
    base = _m_sequence(n, [0, 4, 8])
    for shift in [1, 5, 10]:
        shifted = np.roll(base, shift)
        variants.append(shifted)
    
    # Negated version (flips all signs)
    variants.append(-base)
    
    return variants


def _local_sign_flip_optimize(x, max_rounds=50, time_budget=None, rng=None):
    """
    Iteratively flip signs of individual elements to reduce the objective.
    Uses a greedy approach: try flipping each position, keep if it improves.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    best_h = x.copy().astype(float)
    best_score = _score(best_h)
    
    for rnd in range(max_rounds):
        if time_budget is not None:
            if time.time() - _START_TIME > time_budget:
                break
        
        improved = False
        
        # Try flipping each position in a random order
        indices = np.arange(len(best_h))
        rng.shuffle(indices)
        
        for i in indices:
            if time_budget is not None:
                if time.time() - _START_TIME > time_budget:
                    return best_h, best_score
            
            best_h[i] *= -1
            new_score = _score(best_h)
            if new_score < best_score - 1e-14:
                best_score = new_score
                improved = True
            else:
                best_h[i] *= -1  # revert
        
        if not improved:
            break
    
    return best_h, best_score


def _pairwise_flip_optimize(x, max_rounds=20, time_budget=None, rng=None):
    """
    Try flipping pairs of signs simultaneously to escape local minima.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    best_h = x.copy().astype(float)
    best_score = _score(best_h)
    
    for rnd in range(max_rounds):
        if time_budget is not None:
            if time.time() - _START_TIME > time_budget:
                break
        
        improved = False
        
        # Try random pairs
        n_pairs = min(200, len(best_h) * (len(best_h) - 1) // 10)
        for _ in range(n_pairs):
            if time_budget is not None:
                if time.time() - _START_TIME > time_budget:
                    return best_h, best_score
            
            i, j = rng.choice(len(best_h), size=2, replace=False)
            
            best_h[i] *= -1
            best_h[j] *= -1
            new_score = _score(best_h)
            
            if new_score < best_score - 1e-14:
                best_score = new_score
                improved = True
            else:
                best_h[i] *= -1
                best_h[j] *= -1
        
        if not improved:
            break
    
    return best_h, best_score


def construct():
    """
    Construct a signed step function sequence minimizing max_abs_autoconv / squared_integral.
    
    Strategy:
    1. Start with m-sequence (Gold code) variants which have structurally flat autocorrelation.
    2. Refine each variant with local sign-flip optimization.
    3. Use pairwise flips to escape local minima.
    4. Return the best result found within the time budget.
    """
    rng = np.random.default_rng(42)
    
    deadline = _TIME_LIMIT
    best_x = None
    best_score = float('inf')
    
    # Generate m-sequence variants
    variants = _m_sequence_variants(N)
    
    # Also add a few random sparse sequences as fallbacks
    for density in [0.3, 0.5]:
        x = np.zeros(N)
        mask = rng.random(N) < density
        signs = rng.choice([-1.0, 1.0], size=N)
        x[mask] = signs[mask]
        variants.append(x)
    
    # Optimize each variant
    for idx, x in enumerate(variants):
        if time.time() - _START_TIME > deadline:
            break
        
        # Check if sum is near zero (would make squared_integral tiny)
        s = np.sum(x)
        if abs(s) < 1e-6:
            # Flip all signs or skip
            x = -x
            s = np.sum(x)
            if abs(s) < 1e-6:
                continue
        
        # Local sign-flip optimization
        x_opt, sc = _local_sign_flip_optimize(
            x, 
            max_rounds=30, 
            time_budget=deadline - 5,
            rng=rng
        )
        
        if sc < best_score:
            best_score = sc
            best_x = x_opt.copy()
        
        # If we have time, try pairwise flips on the best so far
        if best_x is not None and time.time() - _START_TIME < deadline - 10:
            x_pw, sc_pw = _pairwise_flip_optimize(
                best_x,
                max_rounds=10,
                time_budget=deadline - 5,
                rng=rng
            )
            if sc_pw < best_score:
                best_score = sc_pw
                best_x = x_pw.copy()
    
    # Final refinement pass if time allows
    if best_x is not None and time.time() - _START_TIME < deadline - 8:
        best_x, best_score = _local_sign_flip_optimize(
            best_x,
            max_rounds=20,
            time_budget=deadline - 2,
            rng=rng
        )
    
    # Fallback
    if best_x is None:
        best_x = _m_sequence(N)
    
    return best_x
# EVOLVE-BLOCK-END
