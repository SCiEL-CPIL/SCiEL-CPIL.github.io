import numpy as np

N = 400

def generate_m_sequence(L, taps):
    """
    Generate a binary m-sequence of length L using an LFSR.
    taps: list of tap positions (0-indexed from LSB, i.e., x^0 is pos 0)
    For x^9 + x^4 + 1, taps are at 9 and 4.
    Note: Standard LFSR implementation varies. 
    Let's use a standard Fibonacci LFSR.
    """
    # Initialize register with 1s
    reg = [1] * L
    seq = []
    for _ in range(L):
        # Calculate feedback
        # For x^L + ... + 1, feedback is XOR of tapped bits.
        # If polynomial is x^9 + x^4 + 1, taps are 9 and 4.
        # In a register of length 9 (indices 0..8), we tap 8 and 3? 
        # Let's stick to a known working implementation for 511.
        # Polynomial: 0x109 (x^9 + x^4 + 1)
        # Taps: 9, 4.
        # Let's use a simpler approach: pre-compute or use a known sequence.
        # Actually, for N=400, we can just use a random sequence and optimize, 
        # but m-sequence is a much better starting point.
        
        # Let's generate the 511 sequence properly.
        # Primitive polynomial for m=9: x^9 + x^4 + 1
        # Taps at 9 and 4.
        # LFSR state: 9 bits.
        pass

def construct():
    # 1. Generate a good initial sequence
    # Use a truncated m-sequence of length 511
    # Primitive polynomial: x^9 + x^4 + 1
    # We will generate the sequence using a simple LFSR simulation.
    
    m = 9
    # Taps for x^9 + x^4 + 1 are at positions 9 and 4.
    # In a 9-bit register (indices 0 to 8), if we shift right, 
    # the feedback bit is XOR of bit 8 (x^8 term in reg?) and bit 3 (x^3 term?).
    # Let's use the standard convention:
    # reg[0] is the newest bit.
    # Feedback = reg[8] ^ reg[3] (for x^9 + x^4 + 1, taps are 9 and 4, so 9-1=8 and 4-1=3 in 0-indexed 9-bit reg?)
    # Actually, for x^m + x^k + 1, taps are m and k.
    # Let's just use a known m-sequence of length 511.
    
    # Generating via LFSR:
    # poly = 0x109 (1 0000 1001) -> x^9 + x^4 + 1
    # Taps: 9, 4.
    
    reg = [1] * m
    seq = []
    for _ in range(511):
        # Output the first bit (or last, doesn't matter for autocorrelation properties)
        seq.append(reg[0])
        # Calculate feedback
        # Tap positions: 9 and 4. 
        # In a 9-bit register, indices 0..8.
        # If we treat the polynomial as x^9 + x^4 + 1, the feedback is XOR of bits at positions corresponding to x^8 and x^3?
        # Let's use a reliable online generator logic or just hardcode a segment.
        # Actually, I can generate it in the code.
        
        # Let's try: feedback = reg[8] ^ reg[3]
        fb = reg[8] ^ reg[3]
        # Shift right
        reg = [fb] + reg[:-1]
        
    # seq is now 0/1 of length 511
    # Convert to +/- 1
    seq_signed = np.array([2*x - 1 for x in seq])
    
    # Truncate to N=400
    x = seq_signed[:N]
    
    # 2. Local Optimization (Hill Climbing)
    # Objective: Minimize sum of squares of non-zero autocorrelations
    # R[k] = sum x[i] x[i+k]
    
    def objective(x):
        # Compute autocorrelation
        # Using FFT for speed
        n = len(x)
        # Pad to next power of 2 for FFT
        N_fft = 1 << (2 * n - 1).bit_length()
        X = np.fft.fft(x, N_fft)
        R = np.fft.ifft(X * np.conj(X)).real[:n]
        # R[0] is the DC term (energy). We want to minimize sum of R[k]^2 for k != 0
        # Due to FFT, R is circular. We need linear autocorrelation.
        # The ifft of |X|^2 gives circular autocorrelation.
        # For linear, we can use the first N elements if we padded enough?
        # Actually, np.correlate is easier for N=400 (fast enough).
        R_lin = np.correlate(x, x, mode='full')
        # R_lin has length 2N-1. Center is at N-1.
        # Non-zero lags are all except the center.
        # We want sum of squares of all non-central lags.
        # R_lin[N-1] is R[0].
        # The rest are R[k] for k != 0.
        # Note: R_lin is symmetric.
        # Sum of squares of non-central parts:
        # left = R_lin[:N-1]
        # right = R_lin[N:]
        # total_energy = np.sum(left**2) + np.sum(right**2)
        # This is 2 * sum_{k=1}^{N-1} R[k]^2
        # Minimizing this is equivalent to minimizing sum_{k=1}^{N-1} R[k]^2
        
        non_central = np.concatenate([R_lin[:N-1], R_lin[N:]])
        return np.sum(non_central**2)
    
    current_score = objective(x)
    
    # Simple Hill Climbing with restarts or just a fixed number of iterations
    # Since N=400, flipping one bit changes R[k] by 2*x[i]x[i+k] for all k.
    # We can compute the delta efficiently, but for 400 iterations * 400 flips, 
    # direct recomputation is 400*400*400 = 64M ops, which is fine in Python/NumPy? 
    # 120s limit. 64M float ops is < 1s in NumPy.
    # Let's do 500 iterations of best-improvement hill climbing.
    
    for _ in range(500):
        best_idx = -1
        best_delta = 0
        best_new_score = current_score
        
        # Sample a subset of indices to flip to save time, or try all?
        # Trying all 400 indices per iteration:
        # Cost per flip check: O(N) to compute new autocorrelation? No, O(N^2) to compute full autocorr.
        # 400 flips * 400^2 = 64,000,000 ops per iteration.
        # 500 iterations * 64M = 32 Billion ops. Too slow in Python loop.
        
        # We need a faster delta calculation.
        # Let x be the current sequence.
        # Let x' be x with x[i] flipped.
        # x'[j] = x[j] + 2*x[i]*delta_{ij} ? No.
        # x'[j] = -x[j] if j==i else x[j].
        # R'[k] = sum_j x'[j] x'[j+k]
        # R'[k] = R[k] - 2*x[i]*(x[i+k] + x[i-k]) + 4*x[i]*x[i+k]*x[i-k] (if i+k == i-k? no)
        # This is complex.
        
        # Alternative: Use a random subset of flips.
        # Or, use the FFT property.
        # Let's just do a few random flips and keep if better.
        
        # Simpler: Random Search with Memory
        # Try 100 random flips per iteration, keep the best one.
        candidates = np.random.choice(N, 100, replace=False)
        for i in candidates:
            x_trial = x.copy()
            x_trial[i] *= -1
            score_trial = objective(x_trial)
            if score_trial < best_new_score:
                best_new_score = score_trial
                best_idx = i
        
        if best_idx != -1:
            x[best_idx] *= -1
            current_score = best_new_score
        else:
            # Stagnation, break or random restart
            # Let's just break for now, m-sequence is already good.
            break
            
    return x

# EVOLVE-BLOCK-END
