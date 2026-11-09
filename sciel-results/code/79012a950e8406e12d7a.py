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

def construct():
    """
    Constructs a signed step function (sequence of +/- 1) of length N
    that minimizes the peak sidelobe level of its autocorrelation.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    np.random.seed(42)
    
    best_h = None
    best_psl = np.inf
    
    # 1. Generate diverse initial seeds
    seeds = []
    
    # Random seeds
    for _ in range(10):
        h_rand = np.random.choice([-1, 1], size=N)
        seeds.append(h_rand)
        
    # Block-based seeds
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
        
    # Thue-Morse
    tm = np.zeros(N, dtype=np.int8)
    for i in range(1, N):
        tm[i] = tm[i >> 1] ^ (i & 1)
    seeds.append(1 - 2 * tm)
    
    # Evaluate seeds
    for h in seeds:
        psl = _compute_psl(h)
        if psl < best_psl:
            best_psl = psl
            best_h = h.copy()
            
    # 2. Optimization: Simulated Annealing with a structural change
    # Instead of random single-bit flips, we use a "Greedy Local Search" approach 
    # combined with larger perturbations to escape local minima.
    # We will alternate between:
    # a) Single bit flips (fine tuning)
    # b) Block flips (coarse tuning)
    # c) Random restarts if stuck
    
    h = best_h.copy()
    current_psl = best_psl
    
    # SA parameters
    T = 50.0
    T_final = 0.1
    alpha = 0.9995
    
    max_iters = 200000
    iter_count = 0
    stuck_count = 0
    max_stuck = 500
    
    while time.time() < deadline and iter_count < max_iters:
        # Decide move type: 80% single bit, 20% block flip
        if np.random.rand() < 0.8:
            idx = np.random.randint(N)
            h_new = h.copy()
            h_new[idx] *= -1
        else:
            # Block flip
            bs = np.random.choice([2, 4, 8, 16])
            if N - bs < 0: continue
            start = np.random.randint(0, N - bs + 1)
            h_new = h.copy()
            h_new[start:start+bs] *= -1
            
        new_psl = _compute_psl(h_new)
        
        # Accept if better or with probability exp(-delta/T)
        delta = new_psl - current_psl
        if delta < 0 or (T > 0 and np.random.rand() < np.exp(-delta / T)):
            h = h_new
            current_psl = new_psl
            stuck_count = 0
            
            if current_psl < best_psl:
                best_psl = current_psl
                best_h = h.copy()
        else:
            stuck_count += 1
            
        # Cool down
        T *= alpha
        if T < T_final:
            T = T_final
            
        # Restart if stuck too long
        if stuck_count > max_stuck:
            # Random restart from a new random sequence or perturbation of best
            if np.random.rand() < 0.5:
                h = np.random.choice([-1, 1], size=N)
            else:
                h = best_h.copy()
                # Perturb 10% of bits
                num_flips = N // 10
                flip_indices = np.random.choice(N, num_flips, replace=False)
                h[flip_indices] *= -1
            
            current_psl = _compute_psl(h)
            T = 50.0  # Reset temperature
            stuck_count = 0
            
        iter_count += 1
        
    return best_h
