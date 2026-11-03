"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _autoconv_max_abs(h):
    """Compute max absolute value of linear autoconvolution using FFT for speed."""
    n = len(h)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)
    return np.max(np.abs(ac))

def _score(h):
    """Compute the objective value: max|autoconv| / (sum h)^2."""
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    return _autoconv_max_abs(h) / (s * s)

def _block_flip_search(h, deadline, block_sizes=None):
    """
    Local optimization using block flips: flip contiguous segments of various sizes.
    This can escape local minima that single-bit flips cannot.
    """
    h = h.copy()
    best_val = _score(h)
    n = len(h)
    
    if block_sizes is None:
        block_sizes = [1, 2, 3, 4, 5, 8, 16]
    
    # Shuffle order to avoid systematic bias
    rng = np.random.default_rng(123)
    
    while time.time() < deadline:
        improved = False
        
        # Try different block sizes
        for bs in block_sizes:
            if time.time() > deadline:
                break
                
            # Try flipping blocks of size bs at random positions
            num_attempts = min(n // bs, 50)  # Limit attempts per block size
            for _ in range(num_attempts):
                if time.time() > deadline:
                    break
                start = rng.integers(0, max(1, n - bs + 1))
                end = min(start + bs, n)
                
                # Flip the block
                h[start:end] = -h[start:end]
                new_val = _score(h)
                
                if new_val < best_val:
                    best_val = new_val
                    improved = True
                else:
                    h[start:end] = -h[start:end]  # revert
        
        # Also try single-bit flips for fine-tuning
        for _ in range(10):
            if time.time() > deadline:
                break
            i = rng.integers(0, n)
            h[i] = -h[i]
            new_val = _score(h)
            if new_val < best_val:
                best_val = new_val
                improved = True
            else:
                h[i] = -h[i]  # revert
        
        if not improved:
            break
            
    return h, best_val

def _simulated_annealing(h, deadline, t_start=1.0, t_end=0.01, cooling_rate=0.995):
    """
    Simulated annealing for block flips to escape local minima.
    Accepts worse solutions with probability based on temperature.
    """
    h = h.copy()
    best_h = h.copy()
    best_val = _score(h)
    current_val = best_val
    n = len(h)
    
    rng = np.random.default_rng(456)
    t = t_start
    
    # Precompute some block sizes to try
    block_sizes = [1, 2, 3, 4, 5, 8, 16, 32]
    
    while time.time() < deadline and t > t_end:
        # Choose a random move: flip a block
        bs = block_sizes[rng.integers(0, len(block_sizes))]
        if bs >= n:
            bs = max(1, n // 2)
        start = rng.integers(0, max(1, n - bs + 1))
        end = min(start + bs, n)
        
        # Apply move
        h[start:end] = -h[start:end]
        new_val = _score(h)
        
        # Decide whether to accept
        delta = new_val - current_val
        if delta < 0 or rng.random() < np.exp(-delta / max(t, 1e-10)):
            current_val = new_val
            if new_val < best_val:
                best_val = new_val
                best_h = h.copy()
        else:
            # Revert
            h[start:end] = -h[start:end]
        
        # Cool down
        t *= cooling_rate
        
    return best_h, best_val

def _spectral_init(rng, n):
    """
    Generate initial sequence by trying to flatten the spectrum.
    Start with random, then adjust signs to reduce spectral peaks.
    """
    h = rng.choice([-1.0, 1.0], size=n)
    
    # Try to reduce spectral variance by flipping signs
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    
    for _ in range(20):
        H = np.fft.rfft(h, fsize)
        mag = np.abs(H)
        
        # Find the peak bin
        peak_idx = np.argmax(mag)
        
        # Flip a random element and see if it reduces the peak
        i = rng.integers(0, n)
        h[i] = -h[i]
        
        H_new = np.fft.rfft(h, fsize)
        if np.max(np.abs(H_new)) < np.max(mag):
            pass  # Keep the flip
        else:
            h[i] = -h[i]  # Revert
    
    return h

def _generate_initial_candidates():
    """Generate diverse initial binary sequences."""
    rng = np.random.default_rng(42)
    candidates = []
    
    # 1. Random binary sequences
    for _ in range(5):
        h = rng.choice([-1.0, 1.0], size=N)
        candidates.append(h)
        
    # 2. Spectral-initialized sequences
    for _ in range(3):
        h = _spectral_init(rng, N)
        candidates.append(h)
    
    # 3. Structured sequences
    x = np.arange(N)
    h_chirp = np.sign(np.cos(np.pi * x**2 / N))
    candidates.append(h_chirp)
    
    # 4. Alternating
    h_alt = np.array([1.0 if i % 2 == 0 else -1.0 for i in range(N)])
    candidates.append(h_alt)
    
    # 5. Block patterns
    for block_size in [4, 8, 16]:
        h_block = np.zeros(N)
        for i in range(N):
            h_block[i] = 1.0 if (i // block_size) % 2 == 0 else -1.0
        candidates.append(h_block)
        
    return candidates

def construct():
    t_start = time.time()
    deadline = t_start + 140.0  # Safety margin before 150s limit

    best_score = np.inf
    best_h = np.ones(N)  # Baseline: constant sequence (score will be high, but valid)

    # Evaluate baseline
    base_score = _score(best_h)
    if base_score < best_score:
        best_score = base_score
        best_h = best_h.copy()

    # Generate initial candidates
    candidates = _generate_initial_candidates()
    
    # Optimize each candidate with block flip search first, then SA for the best ones
    for idx, h_init in enumerate(candidates):
        if time.time() > deadline - 10:
            break
            
        # Ensure sum is not zero
        s = np.sum(h_init)
        if abs(s) < 1e-9:
            h_init[0] = -h_init[0]
            
        # Phase 1: Block flip local search
        time_left = deadline - 3
        h_opt, val_opt = _block_flip_search(h_init, time_left)
        
        if val_opt < best_score:
            best_score = val_opt
            best_h = h_opt.copy()
            
        # Phase 2: For the best candidates, try simulated annealing
        if idx < 3 and time.time() < deadline - 5:
            h_sa, val_sa = _simulated_annealing(h_opt, deadline - 2)
            if val_sa < best_score:
                best_score = val_sa
                best_h = h_sa.copy()

    # Final polish: one more pass of block flips on the best solution
    if time.time() < deadline - 1:
        h_final, val_final = _block_flip_search(best_h, deadline - 0.5, block_sizes=[1, 2, 3, 4, 5])
        if val_final < best_score:
            best_score = val_final
            best_h = h_final.copy()

    return best_h

# EVOLVE-BLOCK-END
