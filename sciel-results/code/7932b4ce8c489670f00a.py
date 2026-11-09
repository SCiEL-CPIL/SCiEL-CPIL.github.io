import numpy as np
import time

N = 400

def compute_peak(h):
    """Compute the peak of the non-zero lags of the autocorrelation."""
    # Use FFT for initial evaluation
    n_fft = 1 << (2 * N).bit_length()
    F = np.fft.rfft(h, n=n_fft)
    power = F * np.conj(F)
    r = np.fft.irfft(power, n=n_fft)
    # r[0] is N. We care about r[1:N] and r[N+1:2N] which are symmetric.
    # Actually, irfft returns the full linear correlation if n_fft is large enough.
    # The correlation lags are at indices 0 to N-1 and 2N-N to 2N-1?
    # No, standard linear correlation via FFT:
    # r[k] = sum h[i] h[i+k]
    # The FFT method gives the circular correlation. If we zero-pad to 2N, 
    # the first N values are the positive lags, and the rest are negative lags.
    # r[0] is lag 0.
    # r[1] to r[N-1] are lags 1 to N-1.
    # r[N] to r[2N-1] correspond to negative lags.
    # Due to symmetry r[k] = r[-k], we only need to check 1 to N-1.
    return np.max(np.abs(r[1:N]))

def lfsr_sequence(poly, init, length):
    """Generate a binary sequence from an LFSR."""
    # poly is a list of tap positions (1-indexed from right, or 0-indexed)
    # Let's use a standard representation: state is a list of bits.
    # x^m + x^k + ... + 1
    # For m=9, x^9 + x^4 + 1 is primitive.
    # Taps: 9, 4, 0 (conceptually)
    # Feedback bit = XOR of bits at positions specified by poly.
    
    # Simpler: Use numpy to implement the recurrence.
    # Let's hardcode a few good primitive polynomials for m=9.
    # P1: x^9 + x^4 + 1
    # P2: x^9 + x^5 + 1
    # P3: x^9 + x^3 + 1
    
    m = 9
    # State is an integer representing the register.
    # Let's just use a list for clarity, or integer bit manipulation.
    # Integer bit manipulation is faster.
    
    # Define taps for x^9 + x^4 + 1
    # The next bit is XOR of bits at positions 4 and 0 (if LSB is pos 0)?
    # Standard LFSR: new_bit = h0 ^ h4.
    # Shift left, insert new_bit at LSB? Or shift right?
    # Let's stick to a simple Python loop for generation, it's only 511 bits.
    
    reg = init
    seq = []
    for _ in range(length):
        # Extract LSB
        bit = reg & 1
        seq.append(1 if bit == 1 else -1)
        
        # Compute feedback
        # For x^9 + x^4 + 1:
        # Feedback = (reg >> 3) & 1  # This is bit 4 (index 3 if LSB is 0? No, x^4 is 4th power)
        # Let's be precise.
        # Polynomial x^9 + x^4 + 1.
        # Recurrence: a_{n+9} = a_{n+4} + a_n
        # If we store a_n at bit 0, a_{n+1} at bit 1, ..., a_{n+8} at bit 8.
        # Then a_{n+9} depends on a_{n+4} (bit 4) and a_n (bit 0).
        fb = (reg & 1) ^ ((reg >> 4) & 1)
        
        # Shift register left by 1, clear top bit (bit 8), insert fb at bit 8?
        # No, usually we shift in the new bit.
        # If we shift left: (reg << 1) & 0x1FF. Then set bit 8 to fb?
        # Or shift right?
        # Let's shift right and insert at top.
        # reg = (reg >> 1) | (fb << 8)
        reg = (reg >> 1) | (fb << 8)
        
    return np.array(seq, dtype=np.int8)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    candidates = []
    
    # 1. Legendre Sequence (P=401)
    P = 401
    h_leg = np.zeros(N, dtype=np.int8)
    for i in range(1, N):
        res = pow(i, (P-1)//2, P)
        h_leg[i] = 1 if res == 1 else -1
    h_leg[0] = 1
    candidates.append(h_leg)
    
    # 2. LFSR Sequences
    # Primitive polynomials for m=9 (period 511)
    # We need to map the LFSR polynomial to the feedback logic.
    # Let's try a few standard ones.
    # x^9 + x^4 + 1
    # x^9 + x^5 + 1
    # x^9 + x^3 + 1
    
    # We will generate sequences for different initial states and polynomials.
    # To keep it simple, we'll use the x^9+x^4+1 logic derived above.
    # And try a few initial states.
    
    for init in [1, 2, 3, 5, 7, 15, 31, 63, 127, 255]:
        seq = lfsr_sequence(1, init, N) # The '1' placeholder for poly, logic is hardcoded
        candidates.append(seq)
        
    # Also try reversed or shifted versions?
    # Shifts of an m-sequence are equivalent in periodic autocorrelation, 
    # but for aperiodic, the edges matter.
    # Let's add a few random sequences too.
    rng = np.random.default_rng(42)
    for _ in range(10):
        candidates.append(rng.choice([-1, 1], N).astype(np.int8))
        
    # Evaluate all candidates
    best_h = None
    best_peak = float('inf')
    
    for h in candidates:
        peak = compute_peak(h)
        if peak < best_peak:
            best_peak = peak
            best_h = h.copy()
            
    # Local Search on the best candidate
    h = best_h.copy()
    current_peak = best_peak
    
    # Precompute FFT size
    n_fft = 1 << (2 * N).bit_length()
    
    # We will use a simple hill climbing with random restarts if stuck.
    # To speed up, we can use the O(N) update rule, but FFT is robust.
    # Let's try to do as many iterations as possible.
    
    stuck_count = 0
    max_stuck = 1000
    
    while time.time() < deadline:
        # Pick a random index to flip
        i = np.random.randint(N)
        
        # Flip
        h[i] *= -1
        
        # Recompute peak
        F = np.fft.rfft(h, n=n_fft)
        power = F * np.conj(F)
        r = np.fft.irfft(power, n=n_fft)
        new_peak = np.max(np.abs(r[1:N]))
        
        if new_peak < current_peak:
            current_peak = new_peak
            stuck_count = 0
            if current_peak < best_peak:
                best_peak = current_peak
                best_h = h.copy()
        else:
            # Revert
            h[i] *= -1
            stuck_count += 1
            
        if stuck_count > max_stuck:
            # Restart with a random sequence or best_h
            # Let's try a random restart
            h = rng.choice([-1, 1], N).astype(np.int8)
            F = np.fft.rfft(h, n=n_fft)
            power = F * np.conj(F)
            r = np.fft.irfft(power, n=n_fft)
            current_peak = np.max(np.abs(r[1:N]))
            stuck_count = 0
            
    return best_h
