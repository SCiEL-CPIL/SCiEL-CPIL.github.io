import numpy as np
import time

N = 400

def _compute_psl(h):
    """
    Computes the Peak Sidelobe Level (PSL) of the aperiodic autocorrelation of h.
    PSL = max_{tau != 0} |R(tau)|
    """
    n_fft = 1 << (2 * N - 1).bit_length()
    H = np.fft.fft(h, n=n_fft)
    R = np.real(np.fft.ifft(H * np.conj(H)))
    # Extract lags 1 to N-1
    sidelobes = R[1:N]
    return np.max(np.abs(sidelobes))

def _thue_morse(n):
    """Generate Thue-Morse sequence of length n (0s and 1s)."""
    seq = np.zeros(n, dtype=np.int8)
    for i in range(1, n):
        seq[i] = seq[i >> 1] ^ (i & 1)
    return seq

def _m_sequence(n):
    """
    Generate an m-sequence (maximum length sequence) of length n.
    For n=400, we use a 9-bit LFSR which gives period 511, then truncate to 400.
    Polynomial: x^9 + x^4 + 1 (primitive)
    """
    # 9-bit LFSR with taps at positions 9 and 4 (0-indexed from LSB: bits 8 and 3)
    # Actually, let's use a standard primitive polynomial.
    # x^9 + x^4 + 1 is primitive.
    # Taps: 9 and 4. In a 9-bit register, feedback comes from bit 8 and bit 3.
    # Let's implement a simple LFSR.
    # State: 9 bits.
    # Feedback = bit[8] ^ bit[3]
    # Shift right, new MSB is feedback.
    
    state = 1  # Non-zero initial state
    seq = np.zeros(n, dtype=np.int8)
    for i in range(n):
        seq[i] = state & 1
        # Compute feedback
        fb = (state >> 8) ^ (state >> 3)
        state = (state >> 1) | (fb << 8)
    return seq

def construct():
    """
    Constructs a signed step function (sequence of +/- 1) of length N
    that minimizes the peak sidelobe level of its autocorrelation.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_psl = np.inf
    best_h = None
    
    # Generate diverse initial seeds
    seeds = []
    
    # 1. Thue-Morse based
    tm = _thue_morse(N)
    h_tm = 1 - 2 * tm  # Convert 0/1 to +1/-1
    seeds.append(h_tm)
    
    # 2. m-sequence based
    mseq = _m_sequence(N)
    h_mseq = 1 - 2 * mseq
    seeds.append(h_mseq)
    
    # 3. Random seeds
    np.random.seed(42)  # Reproducibility
    for _ in range(10):
        h_rand = np.random.choice([-1, 1], size=N)
        seeds.append(h_rand)
    
    # 4. Block-based seeds: alternate blocks of +1 and -1 with varying lengths
    # This exploits the "sparse combinatorial pattern" idea
    for block_len in [4, 8, 16, 32]:
        h_block = np.zeros(N, dtype=np.int8)
        i = 0
        sign = 1
        while i < N:
            end = min(i + block_len, N)
            h_block[i:end] = sign
            sign *= -1
            i = end
        seeds.append(h_block)
    
    # Evaluate all seeds and keep the best
    for h in seeds:
        psl = _compute_psl(h)
        if psl < best_psl:
            best_psl = psl
            best_h = h.copy()
    
    # Simulated Annealing with block moves
    # Start from the best seed
    h = best_h.copy()
    current_psl = best_psl
    
    # SA parameters
    T0 = 50.0  # Initial temperature
    T_final = 0.1
    alpha = 0.999  # Cooling rate
    T = T0
    
    # Block move sizes to try
    block_sizes = [1, 2, 4, 8, 16, 32]
    
    max_iters = 500000
    iter_count = 0
    
    while time.time() < deadline and iter_count < max_iters:
        # Choose a random block size
        bs = np.random.choice(block_sizes)
        
        # Choose a random start position for the block
        start = np.random.randint(0, N - bs + 1)
        
        # Create a new candidate by flipping a block
        h_new = h.copy()
        h_new[start:start+bs] *= -1
        
        # Compute new PSL
        new_psl = _compute_psl(h_new)
        
        # Accept if better or with probability exp(-delta/T)
        delta = new_psl - current_psl
        if delta < 0 or np.random.rand() < np.exp(-delta / T):
            h = h_new
            current_psl = new_psl
            
            # Update global best
            if current_psl < best_psl:
                best_psl = current_psl
                best_h = h.copy()
        
        # Cool down
        T *= alpha
        if T < T_final:
            T = T_final
        
        iter_count += 1
    
    return best_h
