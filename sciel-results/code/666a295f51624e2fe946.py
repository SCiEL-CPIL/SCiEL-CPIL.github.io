import numpy as np
import time

N = 400
M = 2 * N  # FFT size for full autocorrelation

def _calc_score_fft(h):
    """
    Calculate the score: max_{k != 0} |C[k]| / ||h||^2
    """
    h_padded = np.zeros(M)
    h_padded[:N] = h
    
    H = np.fft.fft(h_padded)
    C = np.fft.ifft(H * np.conj(H)).real
    
    # Extract lags 1 to N-1
    # C[0] is lag 0
    # C[1:N] are lags 1 to N-1
    # C[M-N+1:M] are lags -(N-1) to -1
    pos_lags = C[1:N]
    neg_lags = C[M-N+1:M]
    
    all_lags = np.concatenate([pos_lags, neg_lags])
    peak = np.max(np.abs(all_lags))
    
    norm = np.sum(h**2)
    if norm == 0:
        return 0.0
    return peak / norm

def _get_m_sequence(length):
    """
    Generate a maximal length sequence (m-sequence) of given length (must be 2^m - 1).
    Returns a numpy array of -1 and 1.
    """
    # Find m such that 2^m - 1 == length
    m = int(np.log2(length + 1))
    if (2**m - 1) != length:
        return None
        
    # Primitive polynomial for m=9 (length 511)
    # x^9 + x^4 + 1 is a primitive polynomial
    # Feedback taps: 9 and 4 (0-indexed: 8 and 3)
    # State is m bits.
    state = 1  # Non-zero initial state
    seq = []
    for _ in range(length):
        seq.append(1 if state & 1 else -1)
        # Shift right
        new_bit = ((state >> (m-1)) ^ (state >> (m-5))) & 1 # Taps at 9 (index 8) and 4 (index 3)? 
        # Let's use a standard primitive poly for 511.
        # x^9 + x^4 + 1. Taps at 9 and 4.
        # In register [b0...b8], b8 is MSB.
        # Feedback = b8 ^ b3.
        # New bit enters b0.
        # Shift: b0<-b1, ..., b7<-b8, b8<-feedback.
        # My state is integer. bit 0 is b0.
        # b8 is bit 8. b3 is bit 3.
        # feedback = (state >> 8) ^ (state >> 3)
        fb = ((state >> 8) ^ (state >> 3)) & 1
        state = ((state >> 1) | (fb << (m-1))) & (2**m - 1)
        
    return np.array(seq)

def _initialization_strategy(seed):
    """
    Generate initial sequences.
    Strategy 0-4: Truncated m-sequences with different shifts.
    Strategy 5+: Random.
    """
    rng = np.random.RandomState(seed)
    
    m_seq = _get_m_sequence(511)
    if m_seq is not None:
        strategy = seed % 6
        if strategy < 5:
            # Use different shifts of the m-sequence
            shift = (seed * 7) % 511
            # Create a circular buffer
            h = np.zeros(N, dtype=int)
            for i in range(N):
                h[i] = m_seq[(shift + i) % 511]
            return h
        else:
            # Random
            return rng.choice([-1, 1], size=N)
    else:
        return rng.choice([-1, 1], size=N)

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    best_h = None
    best_score = float('inf')
    
    # Phase 1: Try multiple initializations (M-sequence based and random)
    num_initial = 10
    for i in range(num_initial):
        if time.time() > deadline - 80:
            break
        
        h = _initialization_strategy(i)
        score = _calc_score_fft(h)
        
        if score < best_score:
            best_score = score
            best_h = h.copy()
            
    if best_h is None:
        best_h = _initialization_strategy(0)
        best_score = _calc_score_fft(best_h)

    # Phase 2: Simulated Annealing with Block Flips
    h = best_h.copy()
    current_peak = best_score * N  # Denormalize to work with raw peak for SA
    # Actually, let's just work with the score directly or peak.
    # Peak is an integer for +/-1 signals. Score = Peak / N.
    # Minimizing Peak is equivalent to minimizing Score.
    
    # SA Parameters
    T = 50.0  # Initial temperature (in units of peak sidelobe)
    T_min = 0.1
    cooling = 0.999
    no_improve = 0
    max_no_improve = 20000
    
    best_peak_sa = current_peak
    best_h_sa = h.copy()
    
    rng = np.random.RandomState(42)
    
    while time.time() < deadline - 5 and T > T_min:
        # Generate a move: Flip a random block
        # Block length L between 1 and 20
        L = rng.randint(1, 21)
        start_idx = rng.randint(0, N - L + 1)
        
        # Apply move
        h[start_idx:start_idx+L] *= -1
        
        # Calculate new score
        new_score = _calc_score_fft(h)
        new_peak = new_score * N
        
        delta = new_peak - current_peak
        
        # Acceptance criterion
        if delta < 0 or rng.rand() < np.exp(-delta / T):
            current_peak = new_peak
            if new_peak < best_peak_sa:
                best_peak_sa = new_peak
                best_h_sa = h.copy()
                no_improve = 0
                # Update global best
                if new_score < best_score:
                    best_score = new_score
                    best_h = h.copy()
            else:
                no_improve += 1
        else:
            # Reject move
            h[start_idx:start_idx+L] *= -1
            no_improve += 1
        
        T *= cooling
        
        # Restart if stuck
        if no_improve > max_no_improve:
            # Restart from best known or random
            if rng.rand() < 0.5:
                h = best_h_sa.copy()
                current_peak = best_peak_sa
            else:
                h = rng.choice([-1, 1], size=N).astype(int)
                current_peak = _calc_score_fft(h) * N
            
            T = 50.0
            no_improve = 0
            max_no_improve = int(max_no_improve * 1.5) # Allow more patience in later stages? Or reset?
            max_no_improve = 20000

    # Final Hill Climbing to polish
    h = best_h_sa.copy()
    current_peak = best_peak_sa
    
    improved = True
    while improved and time.time() < deadline - 1:
        improved = False
        for j in range(N):
            if time.time() > deadline - 1:
                break
            
            old_val = h[j]
            h[j] = -old_val
            new_score = _calc_score_fft(h)
            new_peak = new_score * N
            
            if new_peak < current_peak:
                current_peak = new_peak
                improved = True
                if new_score < best_score:
                    best_score = new_score
                    best_h = h.copy()
            else:
                h[j] = old_val
                
    if best_h is None:
        best_h = _initialization_strategy(0)
        
    return best_h
