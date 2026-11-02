import numpy as np
import time

N = 400
DEADLINE = 115.0  # Safety margin below 150s evaluator limit

def _compute_r3(h):
    """
    Compute R3[k] = sum_i h[i] h[i+k] h[i+2k] for k = 0 .. N//2
    Using FFT for initialization speed.
    """
    h_float = h.astype(np.float64)
    size = 2 * N + 1
    h_fft = np.fft.fft(h_float, n=size)
    r3_fft = h_fft * h_fft * h_fft
    r3_full = np.fft.ifft(r3_fft).real
    return r3_full[:N+1]

def _score(h):
    r3 = _compute_r3(h)
    return np.max(np.abs(r3))

def _make_sine_pattern(freq, phase=0.0):
    """Create a signed step function based on a sine wave."""
    x = np.linspace(-0.25, 0.25, N, endpoint=False)
    values = np.sin(2 * np.pi * freq * x + phase)
    return np.sign(values)

def _evaluate_block_flip(h, r3, block_start, block_len, k_max):
    """
    Evaluate the effect of flipping a block of bits from block_start to block_start+block_len-1.
    Returns (new_r3, new_score) if improvement, else (None, None).
    """
    h_new = h.copy()
    h_new[block_start:block_start+block_len] *= -1
    
    # Recompute r3 from scratch for simplicity (N=400 is small)
    new_r3 = _compute_r3(h_new)
    new_score = np.max(np.abs(new_r3))
    
    return h_new, new_r3, new_score

def _simulated_annealing(h, r3, current_score, start_temp, end_temp, n_iterations, elapsed_ok):
    """
    Simulated annealing to escape local minima.
    """
    h_best = h.copy()
    r3_best = r3.copy()
    score_best = current_score
    
    for i in range(n_iterations):
        if not elapsed_ok():
            break
        
        # Temperature schedule
        temp = start_temp * (end_temp / start_temp) ** (i / max(n_iterations - 1, 1))
        
        # Random flip: choose a random position and flip it
        i_flip = np.random.randint(N)
        h_new = h.copy()
        h_new[i_flip] *= -1
        
        new_r3 = _compute_r3(h_new)
        new_score = np.max(np.abs(new_r3))
        
        delta = new_score - current_score
        
        # Accept if better, or with probability exp(-delta/temp) if worse
        if delta < 0 or np.random.random() < np.exp(-delta / max(temp, 1e-10)):
            h = h_new
            r3 = new_r3
            current_score = new_score
            
            if current_score < score_best:
                score_best = current_score
                h_best = h.copy()
                r3_best = r3.copy()
    
    return h_best, r3_best, score_best

def construct():
    start_time = time.time()
    
    def elapsed_ok():
        return (time.time() - start_time) < DEADLINE
    
    # Try multiple structured initializations
    initializations = []
    
    # Sine patterns with different frequencies
    for freq in [1, 2, 3, 4, 5, 7, 10, 13, 17, 20]:
        for phase in [0.0, np.pi/2, np.pi, 3*np.pi/2]:
            initializations.append(_make_sine_pattern(freq, phase))
    
    # Random patterns
    for _ in range(20):
        initializations.append(np.random.choice([-1.0, 1.0], size=N))
    
    # Periodic patterns
    for period in [7, 11, 13, 17, 19, 23, 29, 31]:
        pattern = np.tile(np.random.choice([-1.0, 1.0], size=period), N // period + 1)[:N]
        initializations.append(pattern)
    
    best_h = None
    best_score = np.inf
    best_r3 = None
    
    # Evaluate all initializations
    for h_init in initializations:
        if not elapsed_ok():
            break
        score = _score(h_init)
        if score < best_score:
            best_score = score
            best_h = h_init.copy()
            best_r3 = _compute_r3(best_h)
    
    # Coordinate descent with vectorized updates
    h = best_h.copy()
    r3 = best_r3.copy()
    current_score = best_score
    
    max_passes = 500
    
    for p in range(max_passes):
        if not elapsed_ok():
            break
        
        improved = False
        order = np.random.permutation(N)
        
        for i in order:
            if not elapsed_ok():
                return best_h
            
            # Calculate the change in r3 for all k if we flip h[i]
            # Vectorized computation
            h_i = h[i]
            
            # For each k, compute the sum of terms involving h[i]
            # Term 1: h[i+k] * h[i+2k] for i+2k < N
            # Term 2: h[i-k] * h[i+k] for 0 <= i-k and i+k < N
            # Term 3: h[i-2k] * h[i-k] for i-2k >= 0
            
            k_vals = np.arange(N // 2 + 1)
            
            # Term 1: valid when i + 2k < N => k < (N - i) / 2
            mask1 = (i + 2 * k_vals) < N
            term1 = np.zeros_like(k_vals, dtype=np.float64)
            if np.any(mask1):
                idx1 = k_vals[mask1]
                term1[mask1] = h[i + idx1] * h[i + 2 * idx1]
            
            # Term 2: valid when i - k >= 0 and i + k < N => k <= i and k < N - i
            mask2 = (i - k_vals >= 0) & (i + k_vals < N)
            term2 = np.zeros_like(k_vals, dtype=np.float64)
            if np.any(mask2):
                idx2 = k_vals[mask2]
                term2[mask2] = h[i - idx2] * h[i + idx2]
            
            # Term 3: valid when i - 2k >= 0 => k <= i / 2
            mask3 = (i - 2 * k_vals) >= 0
            term3 = np.zeros_like(k_vals, dtype=np.float64)
            if np.any(mask3):
                idx3 = k_vals[mask3]
                term3[mask3] = h[i - 2 * idx3] * h[i - idx3]
            
            term_sum = term1 + term2 + term3
            delta_r3 = -2.0 * h_i * term_sum
            
            if np.any(delta_r3 != 0):
                new_r3 = r3 + delta_r3
                new_score = np.max(np.abs(new_r3))
                
                if new_score < current_score - 1e-12:
                    # Accept flip
                    h[i] = -h_i
                    r3 = new_r3
                    current_score = new_score
                    improved = True
                    
                    if current_score < best_score:
                        best_score = current_score
                        best_h = h.copy()
                        best_r3 = r3.copy()
        
        if not improved:
            # Try simulated annealing to escape local minimum
            if elapsed_ok():
                h_sa, r3_sa, score_sa = _simulated_annealing(
                    h, r3, current_score, 
                    start_temp=0.5, end_temp=0.01, 
                    n_iterations=2000, elapsed_ok=elapsed_ok
                )
                if score_sa < best_score:
                    best_score = score_sa
                    best_h = h_sa.copy()
                    best_r3 = r3_sa.copy()
                    h = h_sa.copy()
                    r3 = r3_sa.copy()
                    current_score = score_sa
                else:
                    # Try random restart
                    if elapsed_ok():
                        h_new = np.random.choice([-1.0, 1.0], size=N)
                        score_new = _score(h_new)
                        if score_new < best_score:
                            best_score = score_new
                            best_h = h_new.copy()
                            best_r3 = _compute_r3(best_h)
                            h = best_h.copy()
                            r3 = best_r3.copy()
                            current_score = best_score
    
    # Final check
    final_score = _score(best_h)
    if final_score < best_score:
        best_score = final_score
        
    return best_h

if __name__ == "__main__":
    h = construct()
    print("Constructed:", h[:20])
    print("Score:", _score(h))
