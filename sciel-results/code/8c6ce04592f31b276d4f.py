import numpy as np
import time

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    def max_autocorr(seq):
        """Compute max absolute autocorrelation for final scoring."""
        n = len(seq)
        size = 1
        while size < 2*n:
            size *= 2
        f = np.zeros(size, dtype=np.complex128)
        f[:n] = seq.astype(np.float64)
        F = np.fft.fft(f)
        ac = np.fft.ifft(F * np.conj(F)).real[:n]
        return np.max(np.abs(ac[1:]))
    
    def compute_ac_fast(seq):
        """Compute autocorrelation using FFT, returns ac[1:] only."""
        n = len(seq)
        size = 1
        while size < 2*n:
            size *= 2
        f = np.zeros(size, dtype=np.complex128)
        f[:n] = seq.astype(np.float64)
        F = np.fft.fft(f)
        ac = np.fft.ifft(F * np.conj(F)).real[:n]
        return ac[1:]
    
    def delta_max_ac(seq, idx):
        """
        Compute the new max |ac[1:]| after flipping seq[idx], in O(N).
        Returns (new_max, ac_before) where ac_before is the old ac[1:].
        """
        n = len(seq)
        # Current autocorrelation lags 1..n-1
        ac = np.zeros(n-1)
        for k in range(1, n):
            # ac[k] = sum_{i=0}^{n-1-k} seq[i]*seq[i+k]
            ac[k-1] = np.dot(seq[:n-k], seq[k:])
        
        old_max = np.max(np.abs(ac))
        
        # Flip seq[idx]
        seq[idx] *= -1
        
        # Recompute ac efficiently: only lags k where the flipped element participates
        # For lag k, ac[k] changes by -2*seq[idx]*seq[idx+k] if idx < n-k, 
        # and -2*seq[idx]*seq[idx-k] if idx >= k (but we need to be careful)
        # Actually, let's just recompute the affected lags
        # When we flip seq[idx], for lag k:
        #   if idx < n-k: the term seq[idx]*seq[idx+k] changes sign -> delta = -2*seq[idx]*seq[idx+k] (but seq[idx] already flipped)
        #   Wait, let me think more carefully.
        # 
        # Original: ac[k] = sum_{i=0}^{n-1-k} s[i]*s[i+k]
        # After flipping s[idx]:
        #   If idx < n-k: term at i=idx changes from s[idx]*s[idx+k] to -s[idx]*s[idx+k]
        #     delta = -2*s[idx]*s[idx+k]  (but s[idx] is now -s[idx]_original)
        #     Actually: new_term = (-s[idx])*s[idx+k] = -s[idx]*s[idx+k]
        #     old_term = s[idx]*s[idx+k]
        #     delta = -2*s[idx]*s[idx+k] where s[idx] is the ORIGINAL value
        #   If idx >= k and idx+k < n: term at i=idx-k changes from s[idx-k]*s[idx] to s[idx-k]*(-s[idx])
        #     delta = -2*s[idx-k]*s[idx] (original)
        #
        # Let me just recompute all lags that involve idx. idx participates in:
        #   Lag k where 0 <= idx <= n-1-k, i.e., k <= n-1-idx
        #   Lag k where idx-k >= 0, i.e., k <= idx
        #   So k ranges from 1 to min(idx, n-1-idx) for the "middle" terms,
        #   but actually idx appears as s[i] when i=idx (so k <= n-1-idx) 
        #   and as s[i+k] when i+k=idx, i.e., i=idx-k (so k <= idx)
        #
        # For simplicity and correctness, let me just recompute the full ac.
        # But that's O(N^2). Let me be smarter.
        
        # Actually, for a single flip at idx, the change in ac[k] is:
        #   delta_ac[k] = -2 * s_orig[idx] * (s[idx+k] if idx+k < n else 0) 
        #                   - 2 * s_orig[idx] * (s[idx-k] if idx-k >= 0 else 0)
        #   Wait no. Let's denote the original sequence as s.
        #   ac_orig[k] = sum_{i=0}^{n-1-k} s[i]*s[i+k]
        #   After flipping s[idx] to -s[idx]:
        #   ac_new[k] = ac_orig[k] - 2*s[idx]*s[idx+k] (if idx+k < n and idx < n-k)
        #                - 2*s[idx]*s[idx-k] (if idx-k >= 0 and k <= idx)
        #   But wait, if both conditions hold, we're double counting? No, because
        #   s[idx] appears in at most one term for each k: either as s[i] with i=idx,
        #   or as s[i+k] with i+k=idx. These are different terms unless idx = idx+k which is impossible.
        #   Actually for a given k, s[idx] can appear as:
        #     - s[i] where i=idx, contributing s[idx]*s[idx+k] (valid if idx+k < n)
        #     - s[i+k] where i=idx-k, contributing s[idx-k]*s[idx] (valid if idx-k >= 0)
        #   These are two different terms in the sum for ac[k], so both contribute.
        
        # Let me just recompute ac from scratch but it's O(N^2). For N=400, that's 160k ops per flip.
        # With 400 flips per sweep, that's 64M ops per sweep. Too slow.
        
        # Better: precompute the contribution. Let me just do a full recompute of ac using FFT.
        # FFT is O(N log N) ~ 400*9 = 3600 ops. For 400 flips, that's 1.4M ops per sweep.
        # That's fast enough.
        
        # Recompute ac after flip
        size = 1
        while size < 2*n:
            size *= 2
        f = np.zeros(size, dtype=np.complex128)
        f[:n] = seq.astype(np.float64)
        F = np.fft.fft(f)
        ac_new = np.fft.ifft(F * np.conj(F)).real[:n][1:]
        
        new_max = np.max(np.abs(ac_new))
        
        # Undo flip
        seq[idx] *= -1
        
        return new_max
    
    # Generate m-sequence using LFSR
    # For N=400, we need an m-sequence of length 2^m - 1 >= 400, so m >= 9 (2^9-1=511)
    # Use a primitive polynomial of degree 9
    
    def generate_msequence(m, poly):
        """Generate m-sequence of length 2^m - 1 using LFSR."""
        L = (1 << m) - 1
        seq = np.zeros(L, dtype=np.int8)
        # Initialize LFSR state
        state = 1  # Non-zero initial state
        for i in range(L):
            seq[i] = 1 if (state & 1) else -1
            # Feedback: XOR of bits specified by poly
            feedback = 0
            for bit in poly:
                feedback ^= (state >> bit) & 1
            state = (state >> 1) | (feedback << (m - 1))
        return seq
    
    # Primitive polynomials of degree 9 (x^9 + x^5 + 1, etc.)
    # x^9 + x^5 + 1: bits at positions 0, 4, 8 (0-indexed from LSB)
    # Actually, let me use a known primitive polynomial.
    # For degree 9: x^9 + x^5 + 1 is primitive.
    # The LFSR taps are at positions corresponding to the polynomial terms.
    # If poly = x^9 + x^5 + 1, then feedback = bit[4] XOR bit[0] (for a 9-bit register)
    # Let me use a standard form.
    
    # Actually, let me just try several different LFSR configurations and random sequences,
    # then use simulated annealing to improve.
    
    rng = np.random.default_rng(42)
    
    best_a = None
    best_score = np.inf
    
    # Try m-sequences of various lengths, truncated to 400
    for m in [9, 10]:
        if time.time() > deadline:
            break
        
        L = (1 << m) - 1
        # Generate m-sequence
        # Use a simple LFSR with a primitive polynomial
        # For m=9, use taps [0, 4] (x^9 + x^5 + 1)
        # For m=10, use taps [0, 3] (x^10 + x^7 + 1)
        
        if m == 9:
            taps = [0, 4]
        elif m == 10:
            taps = [0, 3]
        else:
            continue
        
        # Generate the m-sequence
        state = 1
        mse = np.zeros(L, dtype=np.int8)
        for i in range(L):
            mse[i] = 1 if (state & 1) else -1
            feedback = 0
            for bit in taps:
                feedback ^= (state >> bit) & 1
            state = (state >> 1) | (feedback << (m - 1))
        
        # Truncate to N=400
        a = mse[:N].copy()
        
        # Evaluate
        score = max_autocorr(a)
        if score < best_score:
            best_score = score
            best_a = a.copy()
        
        # Simulated annealing refinement
        T = 10.0
        cooling = 0.999
        a = best_a.copy()
        current_score = best_score
        
        for iteration in range(5000):
            if time.time() > deadline:
                break
            
            # Random flip
            idx = rng.integers(0, N)
            a[idx] *= -1
            
            new_score = max_autocorr(a)
            
            # Accept if better or with probability exp(-(new-current)/T)
            if new_score < current_score:
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_a = a.copy()
            else:
                # Reject: undo flip
                a[idx] *= -1
                # With small probability, keep the worse solution to escape local minima
                if rng.random() < np.exp(-(new_score - current_score) / T):
                    a[idx] *= -1
                    current_score = new_score
                else:
                    a[idx] *= -1
            
            T *= cooling
        
        # Greedy local search after SA
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for idx in range(N):
                if time.time() > deadline:
                    break
                best_a[idx] *= -1
                new_score = max_autocorr(best_a)
                if new_score < best_score:
                    best_score = new_score
                    improved = True
                else:
                    best_a[idx] *= -1
    
    # Also try random initialization with SA
    if time.time() < deadline:
        a = rng.choice([-1, 1], size=N).astype(np.int8)
        score = max_autocorr(a)
        if score < best_score:
            best_score = score
            best_a = a.copy()
        
        T = 5.0
        cooling = 0.999
        current_score = best_score
        a = best_a.copy()
        
        for iteration in range(3000):
            if time.time() > deadline:
                break
            
            idx = rng.integers(0, N)
            a[idx] *= -1
            
            new_score = max_autocorr(a)
            
            if new_score < current_score:
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_a = a.copy()
            else:
                a[idx] *= -1
                if rng.random() < np.exp(-(new_score - current_score) / T):
                    a[idx] *= -1
                    current_score = new_score
                else:
                    a[idx] *= -1
            
            T *= cooling
        
        # Final greedy
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for idx in range(N):
                if time.time() > deadline:
                    break
                best_a[idx] *= -1
                new_score = max_autocorr(best_a)
                if new_score < best_score:
                    best_score = new_score
                    improved = True
                else:
                    best_a[idx] *= -1
    
    return best_a
