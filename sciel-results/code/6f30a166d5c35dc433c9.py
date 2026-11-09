import numpy as np
import time

N = 400

def max_sidelobe(h):
    """
    Compute the maximum normalized autocorrelation sidelobe.
    h: np.array of shape (N,) with values +/- 1.
    """
    n_fft = 1 << (2 * N - 1).bit_length()
    f = np.zeros(n_fft, dtype=np.float64)
    f[:N] = h
    F = np.fft.fft(f)
    R = np.real(np.fft.ifft(F * np.conj(F)))
    R = R[:N]
    R[0] = 0
    # Normalized by N (peak power)
    return np.max(np.abs(R)) / N

def simulate_annealing(h, deadline, rng):
    """
    Optimize h using Simulated Annealing.
    """
    current_score = max_sidelobe(h)
    best_h = h.copy()
    best_score = current_score
    
    # Initial temperature
    T = 0.5
    # Cooling rate
    alpha = 0.999
    # Minimum temperature
    T_min = 1e-5
    
    # We will perform steps until deadline is reached
    # Each step involves a random flip and an FFT.
    # FFT size is 1024. Cost is approx 1024*log2(1024) ~ 10k ops.
    # Python overhead is significant. Let's estimate ~1000 steps/sec.
    # 115 seconds -> ~115,000 steps.
    
    while time.time() < deadline and T > T_min:
        # Randomly select an index to flip
        i = rng.integers(0, N)
        h[i] *= -1
        
        new_score = max_sidelobe(h)
        
        # Acceptance criterion
        delta = new_score - current_score
        if delta < 0 or rng.random() < np.exp(-delta / T):
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_h = h.copy()
        else:
            # Reject: revert
            h[i] *= -1
            
        # Cool down
        T *= alpha
        
    return best_h, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    rng = np.random.default_rng(42)
    
    candidates = []
    
    # 1. Random Start
    h_rand = rng.choice([-1.0, 1.0], N)
    candidates.append(h_rand)
    
    # 2. Thue-Morse Sequence
    # Thue-Morse sequence t_n has low autocorrelation properties in some contexts.
    # t_0 = 0, t_{2n} = t_n, t_{2n+1} = 1 - t_n
    # Map 0 -> 1, 1 -> -1
    def generate_thue_morse(length):
        seq = [0]
        for i in range(length - 1):
            # Extend sequence
            # Efficiently generate next bits
            # Actually, easier to generate up to power of 2 >= length
            pass
        # Simpler generation
        tm = np.zeros(length, dtype=int)
        for i in range(1, length):
            # t_i = popcount(i) % 2
            # Using numpy bit counting is not direct, but we can do it in a loop or use lookup
            # For N=400, loop is fine
            # popcount logic:
            # i & (i-1) removes lowest set bit
            # We can precompute or just use a simple loop in python, or use np.unpackbits?
            # Let's just use a simple property: t(i) = t(i//2) + (i%2) mod 2? No.
            # t(2n)=t(n), t(2n+1)=1-t(n)
            # This recursive definition is hard to vectorize simply without recursion depth issues in Python.
            # Let's generate iteratively:
            # Start with [0]
            # Next: [0, 1]
            # Next: [0, 1, 1, 0]
            # Next: [0, 1, 1, 0, 1, 0, 0, 1]
            # It doubles in size.
            # We need 400. 2^8=256, 2^9=512.
            # Generate 512 and truncate.
            seq = np.array([0])
            while len(seq) < length:
                # Invert the current sequence
                inv = 1 - seq
                seq = np.concatenate([seq, inv])
            return (2 * seq[:length] - 1).astype(np.float64) # 0->-1, 1->1? 
            # Wait, standard mapping:
            # If t=0, val=1. If t=1, val=-1.
            # So val = 1 - 2*t.
            # My previous line: 2*seq - 1 gives 0->-1, 1->1. 
            # Let's stick to 1 - 2*seq.
            return (1 - 2 * seq[:length]).astype(np.float64)

    h_tm = generate_thue_morse(N)
    candidates.append(h_tm)

    # 3. M-Sequence Start (Length 511, truncated to 400)
    def generate_mseq_length(m, taps, length):
        state = 1
        seq = []
        for i in range(length):
            seq.append(1.0 if (state & 1) else -1.0)
            fb = 0
            for t in taps:
                fb ^= (state >> t) & 1
            state = (state >> 1) | (fb << (m - 1))
        return np.array(seq)
        
    # Taps for x^9+x^4+1: bits 0 and 4
    h_mseq = generate_mseq_length(9, [0, 4], 511)
    h_mseq = h_mseq[:N]
    candidates.append(h_mseq)
    
    # 4. All Ones (Baseline, though bad, good for contrast)
    h_ones = np.ones(N)
    candidates.append(h_ones)
    
    best_h = None
    best_score = float('inf')
    
    # Allocate time to each candidate
    # We have ~115s. 4 candidates. ~28s each.
    # Or, we can run SA on the best candidate from a quick pre-screen?
    # Let's run SA on each candidate with a time slice.
    
    time_per_candidate = (deadline - time.time()) / len(candidates)
    
    for idx, h in enumerate(candidates):
        if time.time() > deadline:
            break
            
        h_work = h.copy()
        # Reduce deadline for this specific run to ensure we finish all
        # Leave 1s buffer for the last one?
        # Simpler: just use the global deadline, but if we are on the last candidate, use the remaining time.
        # To be safe, let's give each candidate (deadline - time.time()) / (len(candidates) - idx)
        local_deadline = time.time() + (deadline - time.time()) / (len(candidates) - idx)
        
        h_opt, score = simulate_annealing(h_work, local_deadline, rng)
        
        if score < best_score:
            best_score = score
            best_h = h_opt
            
    if best_h is None:
        return np.ones(N)
        
    return best_h
