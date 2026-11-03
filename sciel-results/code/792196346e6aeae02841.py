import numpy as np
import time

N = 400

def _calc_acf(s):
    """Compute full autocorrelation array A[0..N-1]"""
    n_fft = 1
    while n_fft < 2 * N:
        n_fft *= 2
    s_f = s.astype(np.float64)
    F = np.fft.rfft(s_f, n=n_fft)
    P = F * np.conj(F)
    acf = np.fft.irfft(P, n=n_fft)
    return np.real(acf[:N])

def _max_abs_acf_from_array(A):
    return float(np.max(np.abs(A[1:])))

def _best_improvement_polish(s, current_val, deadline):
    """
    Perform a steepest descent polish by finding the best single-bit flip
    in each iteration.
    """
    improved = True
    max_passes = 1000 # Safety cap, usually converges much faster
    passes = 0
    
    while improved and passes < max_passes:
        if time.time() > deadline:
            break
        improved = False
        passes += 1
        
        best_idx = -1
        best_val = current_val
        
        # Evaluate flipping each bit
        # Optimization: We can update the ACF incrementally, but for N=400, 
        # 400 FFTs is cheap enough. 
        # Let's try to be smarter: 
        # A_new = A + 2 * s[i] * (s * s_shifted) ... this is complex.
        # Just do FFTs.
        
        for i in range(N):
            if time.time() > deadline:
                break
            
            old_si = s[i]
            s[i] = -old_si
            A = _calc_acf(s)
            new_val = _max_abs_acf_from_array(A)
            
            if new_val < best_val:
                best_val = new_val
                best_idx = i
            s[i] = old_si
        
        if best_idx != -1:
            s[best_idx] *= -1
            current_val = best_val
            improved = True
            
    return s, current_val

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin, evaluator kills at 150s
    
    best_val = float('inf')
    best_seq = np.ones(N)
    
    # Parameters for Simulated Annealing
    max_restarts = 50
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
            
        # Initialize with random ±1
        s = np.random.randint(0, 2, size=N) * 2 - 1
        A = _calc_acf(s)
        current_val = _max_abs_acf_from_array(A)
        
        # Simulated Annealing
        T = 10.0
        T_min = 0.01
        cooling_rate = 0.99
        
        # Run SA for a shorter duration, then rely on Polish
        # SA is for escaping local minima, Polish is for converging.
        steps_per_restart = 2000
        
        for step in range(steps_per_restart):
            if time.time() > deadline:
                break
                
            # Generate block flip
            # Use smaller blocks for finer tuning in later stages? 
            # Or random size 1-8.
            L = np.random.randint(1, 9)
            i = np.random.randint(0, N - L + 1)
            
            # Flip block
            s[i:i+L] *= -1
            
            # Recalculate A and objective
            A = _calc_acf(s)
            new_val = _max_abs_acf_from_array(A)
            
            delta_E = new_val - current_val
            
            # Acceptance criterion
            if delta_E < 0 or np.random.random() < np.exp(-delta_E / T):
                current_val = new_val
                if current_val < best_val:
                    best_val = current_val
                    best_seq = s.copy()
            else:
                # Reject and flip back
                s[i:i+L] *= -1
            
            T *= cooling_rate
            if T < T_min:
                break
        
        # High-precision polish: Steepest Descent
        # This is the key improvement. It ensures we are at a local minimum
        # with respect to single-bit flips before considering the result.
        s, current_val = _best_improvement_polish(s, current_val, deadline)
        
        if current_val < best_val:
            best_val = current_val
            best_seq = s.copy()
            
        # If we found a very good solution, stop early
        if best_val < 3.0:
            break

    # Final polish on the best sequence found
    if time.time() < deadline:
        s_final = best_seq.copy()
        val_final = _max_abs_acf_from_array(_calc_acf(s_final))
        s_final, val_final = _best_improvement_polish(s_final, val_final, deadline)
        if val_final < best_val:
            best_val = val_final
            best_seq = s_final

    return best_seq
