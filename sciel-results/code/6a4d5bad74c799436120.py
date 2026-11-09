# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_bound(h):
    """
    Compute the upper bound on C3 for a signed step function h of length N.
    The bound is related to the maximum absolute value of the off-center autocorrelation.
    """
    n = len(h)
    if n == 0:
        return 0.0
    size = 1 << (2 * n - 1).bit_length()
    F = np.fft.rfft(h, size)
    power = F * np.conj(F)
    corr = np.fft.irfft(power, size)[:n]
    if n > 1:
        return np.max(np.abs(corr[1:]))
    return 0.0

def _lfsr_sequence(n, poly, init_state):
    """
    Generate an m-sequence using a Linear Feedback Shift Register (LFSR).
    poly: integer representing the polynomial
    init_state: initial state (non-zero)
    Returns a sequence of +1 and -1 of length n.
    """
    degree = poly.bit_length() - 1
    mask = (1 << degree) - 1
    
    state = init_state & mask
    seq = []
    
    for _ in range(n):
        bit = state & 1
        seq.append(1.0 if bit else -1.0)
        
        fb = 0
        for i in range(degree):
            if (poly >> i) & 1:
                fb ^= (state >> i) & 1
        
        state = (state >> 1) | (fb << (degree - 1))
        state &= mask

    return np.array(seq)

def _try_flip(h, i, current_bound, n):
    """
    Try flipping position i and check if it improves the bound.
    Uses an incremental approach: only do full FFT if the flip might help.
    Returns (new_bound, improved) where improved is True if new_bound < current_bound.
    """
    old_val = h[i]
    h[i] = -old_val
    
    # Quick check: compute the new autocorrelation at the peak lag
    # and see if it might be better
    size = 1 << (2 * n - 1).bit_length()
    F = np.fft.rfft(h, size)
    power = F * np.conj(F)
    corr = np.fft.irfft(power, size)[:n]
    new_bound = np.max(np.abs(corr[1:])) if n > 1 else 0.0
    
    if new_bound < current_bound:
        return new_bound, True
    else:
        h[i] = old_val
        return current_bound, False

def construct():
    N = 400
    deadline = time.time() + 110
    
    # Generate multiple LFSR candidates with different polynomials and phases
    # Known primitive polynomials for degree 8
    polys = [
        0x119,  # x^8 + x^4 + x^3 + x^2 + 1
        0x12D,  # x^8 + x^5 + x^3 + x^2 + 1
        0x15D,  # x^8 + x^6 + x^5 + x^4 + 1
        0x169,  # x^8 + x^7 + x^5 + x^3 + 1
        0x199,  # x^8 + x^7 + x^6 + x^5 + 1
    ]
    
    best_h = None
    best_bound = float('inf')
    
    # Try multiple LFSR initializations
    for poly in polys:
        for init_state in range(1, 16):
            if time.time() > deadline:
                break
            h = _lfsr_sequence(N, poly, init_state)
            bound = _compute_bound(h)
            if bound < best_bound:
                best_bound = bound
                best_h = h.copy()
        
        if time.time() > deadline:
            break
    
    if best_h is None:
        # Fallback to random
        best_h = np.random.choice([1.0, -1.0], N)
        best_bound = _compute_bound(best_h)
    
    h = best_h.copy()
    
    # Local Search Refinement with focus on peak lags
    improved = True
    passes = 0
    while improved and time.time() < deadline:
        improved = False
        passes += 1
        
        # Find the lag with maximum absolute value to focus on
        size = 1 << (2 * N - 1).bit_length()
        F = np.fft.rfft(h, size)
        power = F * np.conj(F)
        corr = np.fft.irfft(power, size)[:N]
        abs_corr = np.abs(corr[1:]) if N > 1 else np.array([0.0])
        
        if len(abs_corr) == 0:
            break
            
        peak_lag = np.argmax(abs_corr) + 1  # +1 because we skipped lag 0
        
        # Strategy: flip positions that contribute most to the peak lag
        # For lag k, positions i and i+k contribute h[i]*h[i+k]
        # We want to reduce |corr[peak_lag]|
        
        # Try flipping positions near the peak lag contributions
        candidates = list(range(N))
        # Prioritize positions that are part of the peak lag pair
        for i in range(N - peak_lag):
            candidates.insert(0, i)
            candidates.insert(0, i + peak_lag)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_candidates = []
        for c in candidates:
            if c not in seen and 0 <= c < N:
                seen.add(c)
                unique_candidates.append(c)
        
        for i in unique_candidates:
            if time.time() > deadline:
                break
            
            new_bound, did_improve = _try_flip(h, i, best_bound, N)
            
            if did_improve:
                best_bound = new_bound
                improved = True
            else:
                pass  # h was already restored in _try_flip
        
        # If no improvement in this pass, try random flips to escape local minimum
        if not improved:
            rng = np.random.RandomState(int(time.time() * 1000) % 100000)
            for _ in range(20):
                if time.time() > deadline:
                    break
                pos = rng.randint(N)
                new_bound, did_improve = _try_flip(h, pos, best_bound, N)
                if did_improve:
                    best_bound = new_bound
                    improved = True
            if not improved:
                # Try a few more random restarts if stuck
                for _ in range(5):
                    if time.time() > deadline:
                        break
                    h = np.random.choice([1.0, -1.0], N)
                    bound = _compute_bound(h)
                    if bound < best_bound:
                        best_bound = bound
                        improved = True
                if not improved:
                    break
    
    return h
# EVOLVE-BLOCK-END
