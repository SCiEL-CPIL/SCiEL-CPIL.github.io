"""Rewritten program: LFSR m-sequence initialization + fast local search."""
import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START

def _autocorr_from_fft(h):
    """Compute full autocorrelation via FFT."""
    n_fft = 2 * N
    F = np.fft.rfft(h, n=n_fft)
    power = F * np.conj(F)
    return np.fft.irfft(power, n=n_fft)


def _c3_from_autocorr(r):
    """Compute C3 upper bound from autocorrelation array."""
    r0 = r[0]
    if r0 <= 0:
        return np.inf
    # Max absolute sidelobe at non-zero lags
    # Positive lags: indices 1..N-1
    # Negative lags: indices n_fft-N+1 .. n_fft-1 (symmetric)
    # For real signals, autocorrelation is symmetric, so max of positive lags suffices
    return np.max(np.abs(r[1:N])) / r0


def _flip_update(r, idx, h):
    """
    Update autocorrelation r after flipping h[idx] (h[idx] *= -1).
    Flipping h[idx] changes r[k] for all k by:
      r_new[k] = r[k] - 4 * h[idx] * h[(idx - k) % N]  (circular)
    But we need linear autocorrelation. Let's just recompute via FFT for simplicity
    but use a faster approach: track the autocorrelation incrementally.
    
    Actually, for linear autocorrelation of a length-N sequence:
    r[k] = sum_{i=0}^{N-1-k} h[i] * h[i+k]
    
    Flipping h[idx]: h[idx] -> -h[idx]
    r_new[k] = r[k] - 2*h[idx]*sum_{i: idx in [i, i+k]} h[i+k] + ... 
    This is complex. Let's use a different approach.
    
    For speed, I'll just do FFT each time but batch the search.
    """
    pass


def _generate_m_sequence(length, degree):
    """
    Generate an m-sequence of the given degree (2^degree - 1 period) and take
    the first `length` elements mapped to +/-1.
    """
    # Use a known primitive polynomial for small degrees
    # For degree 8: x^8 + x^4 + x^3 + x^2 + 1 (0x1F9 in some conventions)
    # Let's use a standard LFSR with primitive polynomial
    
    # Primitive polynomials for LFSR (x^n + ... + 1):
    # degree 8: x^8 + x^4 + x^3 + x^2 + 1 -> taps at [4,3,2,1,0] (1-indexed: 8,4,3,2,1)
    # We'll use tap positions [0, 2, 3, 4, 7] for degree 8 (0-indexed)
    
    # General approach: use numpy to simulate LFSR
    # State is degree bits
    # For simplicity, use a known m-sequence of degree 8 (period 255)
    # and concatenate or tile to get length 400
    
    # Actually, let's just generate using a simple LFSR
    # Primitive polynomial for degree 8: 0x1D7 (x^8+x^5+x^3+x^2+1) or 0x1F9
    # Let's use taps = [0, 2, 3, 4, 7] for degree 8
    
    taps = [0, 2, 3, 4, 7]  # for degree 8
    state = [1] * degree  # Initial state, all ones (non-zero)
    
    seq = []
    for _ in range(length):
        # Output is the last bit (or first bit depending on convention)
        seq.append(1 if state[-1] else -1)
        # Compute new bit
        new_bit = 0
        for t in taps:
            new_bit ^= state[t]
        # Shift state
        state = [new_bit] + state[:-1]
    
    return np.array(seq, dtype=np.float64)


def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s limit
    
    # Initialize with m-sequence of degree 8 (period 255), extended to 400
    rng = np.random.default_rng(42)
    
    # Generate m-sequence of length 400 from degree 8 LFSR
    h = _generate_m_sequence(N, 8)
    
    # Compute initial C3
    r = _autocorr_from_fft(h)
    best_c3 = _c3_from_autocorr(r)
    best_h = h.copy()
    
    # Local search: try single flips that improve C3
    # Use a faster evaluation: for each flip, recompute autocorrelation
    # To speed up, we can try multiple flips per iteration
    
    no_improve = 0
    max_no_improve = 3000
    
    while time.time() - start_time < deadline:
        if time.time() - start_time > deadline - 5:
            break
        
        # Try a random flip
        idx = rng.integers(0, N)
        h[idx] *= -1
        
        r = _autocorr_from_fft(h)
        new_c3 = _c3_from_autocorr(r)
        
        if new_c3 < best_c3:
            best_c3 = new_c3
            best_h = h.copy()
            no_improve = 0
        else:
            h[idx] *= -1  # revert
            no_improve += 1
        
        if no_improve > max_no_improve:
            # Try a batch of random restarts
            for _ in range(10):
                if time.time() - start_time > deadline - 10:
                    break
                h_new = _generate_m_sequence(N, 8)
                # Randomly flip some positions
                flip_count = rng.integers(0, 20)
                flip_idx = rng.choice(N, size=flip_count, replace=False)
                h_new[flip_idx] *= -1
                r_new = _autocorr_from_fft(h_new)
                c3_new = _c3_from_autocorr(r_new)
                if c3_new < best_c3:
                    best_c3 = c3_new
                    best_h = h_new.copy()
                    h = h_new.copy()
                    no_improve = 0
                    break
            else:
                no_improve = 0
    
    return best_h

# EVOLVE-BLOCK-END
