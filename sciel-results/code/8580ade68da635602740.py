import numpy as np
import time

def construct():
    # Based on the inspiration, N=400 is a viable length.
    # The theoretical minimum for binary sequences is 1/N.
    # We aim to approach this limit using local search.
    N = 400
    TIME_LIMIT = 150.0
    SAFETY_MARGIN = 10.0
    DEADLINE = time.time() + TIME_LIMIT - SAFETY_MARGIN

    # Precompute FFT size for linear convolution of two length N sequences
    # Result length is 2N - 1
    FFT_SIZE = 1
    while FFT_SIZE < 2 * N - 1:
        FFT_SIZE *= 2

    def _autoconv_fft(f):
        """Compute linear autoconvolution using FFT."""
        F = np.fft.rfft(f, n=FFT_SIZE)
        C = np.fft.irfft(F * F, n=FFT_SIZE)
        return C[:2 * N - 1]

    def _objective_vectorized(f):
        """
        Compute max(|autoconv(f)|) / (sum(f))^2.
        """
        S = np.sum(f)
        if abs(S) < 1e-12:
            return 1e10
        C = _autoconv_fft(f)
        return np.max(np.abs(C)) / (S * S)

    def _local_search_binary_fast(f_init, time_budget=10.0):
        """
        Local search on binary {-1, 1} sequences using incremental convolution updates.
        """
        f = np.sign(f_init).astype(np.float64)
        f[f == 0] = 1.0
        
        S = np.sum(f)
        if abs(S) < 1e-6:
            return f, 1e10
            
        C = _autoconv_fft(f)
        score = np.max(np.abs(C)) / (S * S)
        
        start_time = time.time()
        no_improve_count = 0
        
        while time.time() - start_time < time_budget:
            if time.time() > DEADLINE:
                break
                
            best_improvement = 0
            best_flip_idx = -1
            best_new_C = None
            best_new_S = S
            
            # Try flipping each position
            for i in range(N):
                v = f[i]
                # delta = -2 * v
                # C_new[k] = C[k] - 4 * v * f[k-i] + 4 * delta_{k, 2i}
                
                C_new = C.copy()
                
                # Apply linear change
                k_start = i
                k_end = i + N
                
                if k_start < len(C_new):
                    end_idx = min(k_end, len(C_new))
                    f_segment = f[:end_idx - k_start]
                    C_new[k_start:end_idx] -= 4.0 * v * f_segment
                    
                # Apply quadratic change
                if 2 * i < len(C_new):
                    C_new[2 * i] += 4.0
                    
                new_S = S - 2.0 * v
                if abs(new_S) < 1e-6:
                    continue
                    
                new_score = np.max(np.abs(C_new)) / (new_S * new_S)
                
                if new_score < score - best_improvement:
                    best_improvement = score - new_score
                    best_flip_idx = i
                    best_new_C = C_new
                    best_new_S = new_S
            
            if best_flip_idx >= 0:
                v = f[best_flip_idx]
                f[best_flip_idx] = -v
                S = best_new_S
                C = best_new_C
                score = np.max(np.abs(C)) / (S * S)
                no_improve_count = 0
            else:
                no_improve_count += 1
                if no_improve_count > 5:
                    break
                
        return f, score

    best_f = None
    best_score = 1e10
    
    # Multiple restarts to escape local minima
    # We have ~130 seconds of budget.
    # Each local search takes some time.
    # We will run as many restarts as possible.
    
    while time.time() < DEADLINE:
        # Initialize with a random binary sequence
        # Bias towards more +1s to keep S large?
        # Or just random.
        f_init = np.random.choice([-1, 1], size=N)
        
        # Run local search with a small time budget per restart
        # If it converges fast, we get more restarts.
        f, score = _local_search_binary_fast(f_init, time_budget=5.0)
        
        if score < best_score:
            best_score = score
            best_f = f.copy()
            
        # Check if we should break
        if time.time() > DEADLINE - 5:
            break

    if best_f is None:
        # Fallback to constant sequence
        best_f = np.ones(N)
        
    return best_f

