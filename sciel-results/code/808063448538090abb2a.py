"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
import math

def _autocorr_periodic(h):
    """
    Compute the periodic autocorrelation of a sequence h.
    h is a numpy array of length N with values +/-1.
    Returns an array R of length N where R[k] = sum_{n=0}^{N-1} h[n] * h[(n+k) % N].
    """
    # Using FFT for periodic autocorrelation is O(N log N)
    # But for N=400, O(N^2) is also fine, and FFT might have precision issues or overhead for small N.
    # However, FFT is standard.
    # Let's use direct computation for clarity and exactness (integers).
    # R[k] = sum h[n] h[n+k]
    # This is circular cross-correlation.
    
    # FFT method:
    # H = fft(h)
    # R = ifft(H * conj(H))
    # Note: ifft(H*conj(H)) gives the circular autocorrelation.
    
    h_float = h.astype(np.float64)
    H = np.fft.fft(h_float)
    power = H * np.conj(H)
    R = np.fft.ifft(power).real
    
    # Due to floating point errors, round to nearest integer
    # Since h is +/-1, R[k] must be an integer with same parity as N.
    # N=400, so R[k] is even.
    R = np.round(R)
    return R

def _autocorr_peak_periodic(R):
    """
    Compute the max absolute autocorrelation over non-zero lags.
    R is the periodic autocorrelation array.
    """
    if N <= 1:
        return 0.0
    # R[0] is the sum of squares = N. We ignore lag 0.
    return np.max(np.abs(R[1:]))

def _generate_lfsr_sequence(poly, init, length):
    """
    Generate an LFSR sequence.
    poly: list of tap positions (0-indexed from LSB, or similar convention).
    Let's use a standard Galois or Fibonacci LFSR.
    For m-sequences, we need primitive polynomials.
    For N=400, we can use a 9-bit LFSR (length 511) and take the first 400 bits.
    Common primitive polynomial for 9 bits: x^9 + x^4 + 1 (taps at 9, 4, 0? depends on implementation).
    Let's implement a simple Fibonacci LFSR.
    State is a list of bits.
    """
    # We'll just generate a long sequence and slice it.
    # Using a known good primitive polynomial for 9 bits: 0x119 (x^9 + x^4 + 1)
    # Taps: 9, 4.
    # Implementation:
    # state = [b0, b1, ..., b8]
    # next_bit = b0 ^ b4
    # shift left, insert next_bit at end?
    
    # Let's use a generic approach for 9-bit LFSR
    # Polynomial x^9 + x^4 + 1
    # Tap positions: 9 and 4.
    # In 0-indexed 9-bit register: bits 0..8.
    # Feedback = bit 0 XOR bit 4.
    # Shift register right, insert feedback at MSB?
    # Standard Fibonacci:
    # s(t) = s(t-9) XOR s(t-5) ? No, x^9 + x^4 + 1 means s(t) = s(t-9) + s(t-4) + s(t)?
    # Characteristic poly: P(x) = x^9 + x^4 + 1.
    # Recurrence: s_t = s_{t-9} + s_{t-5} (if taps are 9 and 5? No, degree 9 and 4).
    # If P(x) = x^n + ... + 1, recurrence is s_t = sum c_i s_{t-i}.
    # x^9 + x^4 + 1 -> s_t = s_{t-9} + s_{t-5}.
    # Let's verify.
    
    def lfsr_9bit(seed):
        # 9-bit state
        state = seed
        seq = []
        for _ in range(length):
            # Extract LSB
            out = state & 1
            seq.append(out)
            # Feedback
            fb = (state >> 0) ^ (state >> 4)
            # Shift right, insert fb at bit 8
            state = (state >> 1) | (fb << 8)
        return seq

    # Try multiple seeds to find good prefixes
    best_seq = None
    best_peak = float('inf')
    
    # We need a sequence of length N.
    # Generate 511 bits, take first 400.
    # Map 0->-1, 1->1.
    
    for seed in range(1, 512):
        if seed == 0: continue
        bits = lfsr_9bit(seed)
        if len(bits) < N:
            bits += bits[:N-len(bits)] # Just in case, though 511 > 400
        h = np.array([1 if b==1 else -1 for b in bits[:N]], dtype=np.float64)
        R = _autocorr_periodic(h)
        peak = _autocorr_peak_periodic(R)
        if peak < best_peak:
            best_peak = peak
            best_seq = h
            
    return best_seq

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s limit
    
    # 1. Generate LFSR-based initializations
    # We'll try a few different LFSR configurations or just the 9-bit one with different seeds
    # The function above finds the best seed for the 9-bit LFSR.
    # Let's also try other primitive polynomials if time permits, but 9-bit is good enough for N=400.
    
    candidates = []
    
    # LFSR candidates
    lfsr_h = _generate_lfsr_sequence(0, 0, N)
    if lfsr_h is not None:
        candidates.append(lfsr_h)
        
    # Random candidates
    rng = np.random.RandomState(42)
    for _ in range(20):
        h_rand = rng.choice([-1.0, 1.0], size=N)
        candidates.append(h_rand)
        
    # 2. Local Search
    # We will run local search on each candidate.
    # We can merge the search: keep a global best, and run searches on all candidates.
    
    best_h = None
    best_score = float('inf')
    
    for h_init in candidates:
        if time.time() > deadline:
            break
            
        h = h_init.copy()
        R = _autocorr_periodic(h)
        current_score = _autocorr_peak_periodic(R)
        
        if current_score < best_score:
            best_score = current_score
            best_h = h.copy()
            
        # Local Search: Single bit flips
        improved = True
        # Limit passes to avoid infinite loops or wasting time
        max_passes = 50 
        passes = 0
        
        while improved and passes < max_passes and time.time() < deadline:
            improved = False
            passes += 1
            
            # Shuffle order to avoid bias? Or just iterate 0..N-1
            # Iterating 0..N-1 is fine.
            for i in range(N):
                if time.time() > deadline:
                    break
                
                # Calculate delta R for flipping bit i
                # R[k] = sum h[n] h[n+k]
                # Flip h[i] -> -h[i]
                # Terms involving i:
                # 1. n=i: h[i]h[i+k] -> -h[i]h[i+k]. Change: -2 h[i] h[i+k]
                # 2. n+k=i => n=i-k: h[i-k]h[i] -> -h[i-k]h[i]. Change: -2 h[i-k] h[i]
                # Total change for R[k]: -2 h[i] (h[i+k] + h[i-k])
                
                # Vectorized calculation of delta_R
                # indices = (i + np.arange(N)) % N
                # idx1 = (i + np.arange(N)) % N
                # idx2 = (i - np.arange(N)) % N
                # delta_R = -2 * h[i] * (h[idx1] + h[idx2])
                
                idx1 = (i + np.arange(N)) % N
                idx2 = (i - np.arange(N)) % N
                delta_R = -2.0 * h[i] * (h[idx1] + h[idx2])
                
                new_R = R + delta_R
                # R[0] should remain N. 
                # Let's check: k=0. delta_R[0] = -2 h[i] (h[i] + h[i]) = -4 h[i]^2 = -4.
                # R[0] = N. New R[0] = N-4? 
                # Wait. R[0] = sum h[n]^2 = N.
                # If we flip one bit, sum h[n]^2 is still N.
                # Why did my formula give -4?
                # R[0] = sum_{n} h[n] h[n].
                # Term n=i: h[i]h[i] = 1.
                # Term n+k=0 => n=0? No, k=0.
                # The formula R[k] = sum h[n] h[n+k] includes n such that n+k wraps.
                # For k=0, R[0] = sum h[n]h[n].
                # The term where n=i is h[i]h[i]=1.
                # The term where n+k=i is n=i, so h[i]h[i]=1.
                # Are they the same term?
                # In the sum sum_{n=0}^{N-1} h[n]h[n+k], for k=0, it is sum h[n]^2.
                # The index n=i appears once.
                # My decomposition "Term 1: n=i" and "Term 2: n+k=i" double counts when k=0?
                # No, for k=0, n=i and n+k=i are the same condition n=i.
                # So the change is just -2 h[i] h[i] = -2?
                # Old: h[i]h[i] = 1. New: (-h[i])(-h[i]) = 1. Change 0.
                # Let's re-evaluate delta R[k].
                # R[k] = sum_{n} h[n] h[(n+k)%N]
                # If we flip h[i]:
                # Only terms where n=i OR (n+k)%N = i change.
                # Case 1: n=i. Term is h[i] h[(i+k)%N]. Becomes -h[i] h[(i+k)%N]. Change: -2 h[i] h[(i+k)%N].
                # Case 2: (n+k)%N = i => n = (i-k)%N. Term is h[(i-k)%N] h[i]. Becomes h[(i-k)%N] (-h[i]). Change: -2 h[(i-k)%N] h[i].
                # If k=0, Case 1 and Case 2 are the same term (n=i).
                # So for k=0, Change = -4 h[i]^2 = -4?
                # But h[i]^2=1. Old term 1, New term 1. Change 0.
                # Ah, the "New" term in Case 1 is (-h[i]) * h[i+k].
                # The "New" term in Case 2 is h[i-k] * (-h[i]).
                # If k=0, New Term 1 = -h[i]^2 = -1.
                # New Term 2 = -h[i]^2 = -1.
                # Sum of new terms = -2.
                # Old sum = 1 + 1 = 2? No, it's the SAME term.
                # In the sum, the term is just one.
                # So for k=0, there is only ONE term involving h[i].
                # My formula adds the changes from both "perspectives".
                # So for k=0, we double counted the change.
                # We must handle k=0 separately or subtract the double count.
                # Actually, R[0] is always N. It doesn't change.
                # So we can just set new_R[0] = N.
                
                new_R[0] = N
                
                new_score = np.max(np.abs(new_R[1:]))
                
                if new_score < current_score:
                    # Accept flip
                    h[i] = -h[i]
                    R = new_R
                    current_score = new_score
                    improved = True
                    
                    if current_score < best_score:
                        best_score = current_score
                        best_h = h.copy()
                    # If we found a very good score, we might want to break early?
                    # No, keep searching.
                
        # After local search, try some random restarts from the best found so far?
        # Or just proceed to next candidate.
        
    return best_h

# EVOLVE-BLOCK-END
