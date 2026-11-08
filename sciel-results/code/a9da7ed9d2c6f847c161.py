"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

N = 400

# EVOLVE-BLOCK-START

def _autoconv(a):
    """Linear autoconvolution via FFT."""
    n = len(a)
    size = 1 << (2 * n - 1).bit_length()
    fa = np.fft.rfft(a, size)
    fa2 = fa * fa
    ac = np.fft.irfft(fa2, size)[:2 * n - 1]
    return ac

def _score(a):
    """Compute max|autoconv| / (sum a)^2. Lower is better."""
    s = np.sum(a)
    if s == 0:
        return float('inf')
    ac = _autoconv(a)
    return np.max(np.abs(ac)) / (s * s)

def construct():
    t_start = time.time()
    deadline = t_start + 140.0  # safety margin before 150s kill
    
    # Start with a structured initialization
    # Use a "block" structure: alternating +1 and -1 blocks
    # This naturally reduces autoconvolution peaks
    
    # Try multiple initial structures
    candidates = []
    
    # 1. Simple alternating
    alt = np.ones(N)
    alt[::2] = -1
    candidates.append(alt)
    
    # 2. Block alternating with varying block sizes
    for block_size in [2, 4, 8, 16, 32]:
        a = np.ones(N)
        sign = 1.0
        for i in range(0, N, block_size):
            a[i:i+block_size] = sign
            sign *= -1
        candidates.append(a)
    
    # 3. Random sign assignments with some structure
    np.random.seed(42)
    for _ in range(5):
        a = np.random.choice([-1.0, 1.0], size=N)
        candidates.append(a)
    
    # 4. Walsh-Hadamard-like patterns
    # Use first few rows of Hadamard matrix
    from scipy.linalg import hadamard
    try:
        H = hadamard(512)  # 512 >= 400
        for row_idx in [1, 2, 3, 4, 8, 16, 32]:
            if row_idx < 512:
                h_row = H[row_idx, :N]
                a = np.sign(h_row)
                a[a == 0] = 1.0
                candidates.append(a)
    except:
        pass
    
    # Evaluate all candidates
    best_score = float('inf')
    best_a = None
    for a in candidates:
        s = _score(a)
        if s < best_score:
            best_score = s
            best_a = a.copy()
    
    # Local search: flip individual elements to improve score
    if best_a is not None:
        current_score = best_score
        current_a = best_a.copy()
        
        # Greedy local search with time budget
        max_iterations = 50000
        it = 0
        while it < max_iterations:
            if time.time() > deadline:
                break
            
            # Try flipping a random element
            idx = np.random.randint(N)
            old_val = current_a[idx]
            current_a[idx] = -old_val
            
            new_score = _score(current_a)
            
            if new_score < current_score:
                current_score = new_score
                it += 1
            else:
                current_a[idx] = old_val
                it += 1
            
            # Occasionally try a larger neighborhood
            if it % 100 == 0:
                # Try flipping a random block
                block_len = np.random.randint(1, 10)
                start = np.random.randint(0, max(1, N - block_len + 1))
                end = min(start + block_len, N)
                old_block = current_a[start:end].copy()
                current_a[start:end] *= -1
                
                new_score = _score(current_a)
                if new_score < current_score:
                    current_score = new_score
                else:
                    current_a[start:end] = old_block
        
        best_a = current_a
        best_score = current_score
    
    # Second phase: more sophisticated optimization using gradient-like approach
    # For signed step functions, we can try to minimize the max absolute autoconvolution
    # by adjusting the "phase" of different frequency components
    
    if time.time() < deadline - 20:
        # Try a spectral approach: design the signal in frequency domain
        # to minimize autoconvolution peaks
        
        # Use a "flat spectrum" approach with random phases
        np.random.seed(123)
        for trial in range(10):
            if time.time() > deadline:
                break
            
            # Create a signal with flat amplitude spectrum and random phases
            n_fft = 1 << (N).bit_length()
            freqs = np.fft.rfftfreq(n_fft)
            amplitudes = np.ones(len(freqs))
            phases = 2 * np.pi * np.random.rand(len(freqs))
            spectrum = amplitudes * np.exp(1j * phases)
            spectrum[0] = 0  # Remove DC component
            
            a = np.fft.irfft(spectrum, n_fft)[:N]
            # Round to nearest sign
            a = np.sign(a)
            a[a == 0] = 1.0
            
            s = _score(a)
            if s < best_score:
                best_score = s
                best_a = a.copy()
        
        # Refine the best spectral candidate
        if best_a is not None:
            current_a = best_a.copy()
            current_score = best_score
            
            # Local refinement
            for _ in range(20000):
                if time.time() > deadline:
                    break
                idx = np.random.randint(N)
                old_val = current_a[idx]
                current_a[idx] = -old_val
                new_score = _score(current_a)
                if new_score < current_score:
                    current_score = new_score
                else:
                    current_a[idx] = old_val
    
    # Final check
    if best_a is None:
        best_a = np.ones(N)
    
    return best_a

# EVOLVE-BLOCK-END
