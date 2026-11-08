import numpy as np
import time

N = 400
FFT_SIZE = 1024  # Next power of 2 >= 2*N - 1

def _calc_score(h):
    """
    Calculate the score: max_abs_sid / integral^2
    """
    # Compute autoconvolution using FFT
    # h is real, so f * f is power spectrum
    f = np.fft.fft(h, FFT_SIZE)
    c = np.fft.ifft(f * f).real
    
    # Round to nearest integer to remove floating point noise
    # This is safe because the true values are integers
    c = np.round(c)
    
    # Extract sidelobes (lag != 0)
    # Positive lags 1 to N-1
    pos_lags = c[1:N]
    # Negative lags -(N-1) to -1
    # In FFT output, these are at the end
    neg_lags = c[FFT_SIZE - (N - 1):]
    
    max_abs_sid = max(np.max(np.abs(pos_lags)), np.max(np.abs(neg_lags)))
    
    integral = np.sum(h)
    int_sq = integral * integral
    
    if int_sq == 0:
        return float('inf')
        
    return max_abs_sid / int_sq

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin for safety
    
    rng = np.random.default_rng(42)
    
    best_score = float('inf')
    best_h = np.ones(N, dtype=np.int8)
    
    # Generate diverse initial candidates
    candidates = []
    
    # 1. All 1s
    candidates.append(np.ones(N, dtype=np.int8))
    
    # 2. Biased random sequences (high sum)
    for bias in [0.95, 0.9, 0.85, 0.8, 0.7]:
        h_rand = rng.choice([1, -1], size=N, p=[bias, 1-bias]).astype(np.int8)
        candidates.append(h_rand)
        
    # 3. Block sequence (half 1s, half -1s) - might have low sum but low sidelobes?
    h_block = np.ones(N, dtype=np.int8)
    h_block[N//2:] = -1
    candidates.append(h_block)
    
    # 4. Rudin-Shapiro like (recursive)
    # Simple recursive construction for low autocorrelation
    def rudin_shapiro(n):
        if n == 1:
            return np.array([1], dtype=np.int8)
        prev = rudin_shapiro(n // 2)
        return np.concatenate([prev, prev * np.array([1, -1], dtype=np.int8).repeat(n // 4) if n % 4 == 0 else np.ones(n//2)])
    # Rudin-Shapiro is tricky to implement exactly for arbitrary N.
    # Let's stick to biased randoms which are more flexible for the sum constraint.
    
    # Evaluate candidates and pick the best starting point
    current_h = None
    current_score = float('inf')
    for h_cand in candidates:
        s = _calc_score(h_cand)
        if s < current_score:
            current_score = s
            current_h = h_cand.copy()
            
    if current_score < best_score:
        best_score = current_score
        best_h = current_h.copy()
        
    # Simulated Annealing
    h = current_h.copy()
    score = current_score
    
    # Temperature tuning
    # Scores are typically ~0.001 to 0.005
    # Deltas are small
    T = 5e-4
    T_min = 1e-7
    cooling_rate = 0.9995
    
    # Pre-allocate for speed? Not necessary with numpy.
    
    while time.time() < deadline and T > T_min:
        # Choose move: 80% single flip, 20% double flip
        move_type = rng.random()
        indices_to_flip = []
        
        if move_type < 0.8:
            idx = rng.integers(N)
            indices_to_flip.append(idx)
        else:
            idxs = rng.integers(0, N, size=2)
            if idxs[0] != idxs[1]:
                indices_to_flip.extend(idxs)
            else:
                indices_to_flip.append(idxs[0])
                
        # Apply move
        for idx in indices_to_flip:
            h[idx] *= -1
            
        new_score = _calc_score(h)
        delta = new_score - score
        
        # Acceptance criterion
        if delta < 0:
            score = new_score
            if score < best_score:
                best_score = score
                best_h = h.copy()
        else:
            # Accept worse solution with probability exp(-delta / T)
            # Avoid overflow in exp
            if delta / T < 50: 
                prob = np.exp(-delta / T)
            else:
                prob = 0.0
                
            if rng.random() < prob:
                score = new_score
            else:
                # Revert move
                for idx in indices_to_flip:
                    h[idx] *= -1
        
        T *= cooling_rate
        
    # Final Polish: Strict Hill Climbing
    # This helps refine the result if SA ended in a local optimum
    h = best_h.copy()
    score = _calc_score(h)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        best_flip_idx = -1
        best_flip_score = score
        
        # Check single flips
        for i in range(N):
            if time.time() > deadline:
                break
            h[i] *= -1
            s = _calc_score(h)
            if s < best_flip_score:
                best_flip_score = s
                best_flip_idx = i
                improved = True
            else:
                h[i] *= -1
                
        if improved:
            h[best_flip_idx] *= -1
            score = best_flip_score
            if score < best_score:
                best_score = score
                best_h = h.copy()
        else:
            break
            
    return best_h
