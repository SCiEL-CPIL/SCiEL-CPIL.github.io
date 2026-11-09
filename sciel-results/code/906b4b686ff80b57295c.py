import numpy as np
import time

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Objective: Minimize max(|R(k)|) for k != 0
    def get_peak_sidelobe(x):
        # Use FFT
        # Pad to 2N to avoid circular convolution artifacts for lags < N
        # Actually, for aperiodic autocorrelation, we need length 2N-1.
        # np.fft.rfft with n=2*N is safe.
        n_fft = 2 * N
        F = np.fft.rfft(x, n=n_fft)
        R = np.fft.irfft(F * np.conj(F), n=n_fft)[:N]
        # R[0] is N. We want max |R[1:]|
        return np.max(np.abs(R[1:]))
    
    best_x = None
    best_score = float('inf')
    
    # Try a few random restarts with SA
    rng = np.random.default_rng(42)
    
    while time.time() < deadline - 10.0: # Leave 10s for final polish
        x = rng.choice([-1.0, 1.0], size=N)
        
        # Simulated Annealing
        T = 10.0
        T_min = 0.01
        cooling_rate = 0.995
        current_score = get_peak_sidelobe(x)
        current_x = x.copy()
        
        # Track best in this run
        run_best_x = current_x.copy()
        run_best_score = current_score
        
        while T > T_min and time.time() < deadline - 10.0:
            # Pick random index
            i = rng.integers(0, N)
            
            # Flip
            old_val = current_x[i]
            current_x[i] = -old_val
            
            new_score = get_peak_sidelobe(current_x)
            
            delta = new_score - current_score
            
            # Accept if better or with probability exp(-delta/T)
            if delta < 0 or (T > 0 and rng.random() < np.exp(-delta / T)):
                current_score = new_score
                if new_score < run_best_score:
                    run_best_score = new_score
                    run_best_x = current_x.copy()
            else:
                # Revert
                current_x[i] = old_val
            
            T *= cooling_rate
            
        # End of SA run
        if run_best_score < best_score:
            best_score = run_best_score
            best_x = run_best_x.copy()
            
        # If we found a very good score, we might break early?
        # Theoretical lower bound for peak sidelobe is ~sqrt(N)?
        # For N=400, sqrt(400)=20.
        # If best_score < 25, it's quite good.
        if best_score < 25:
             break

    # Final Polish with Local Search on best_x
    if best_x is not None and time.time() < deadline - 1.0:
        x = best_x.copy()
        current_score = best_score
        
        improved = True
        while improved and time.time() < deadline - 1.0:
            improved = False
            for i in range(N):
                if time.time() > deadline - 1.0:
                    break
                old_val = x[i]
                x[i] = -old_val
                new_score = get_peak_sidelobe(x)
                
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                    if current_score < best_score:
                        best_score = current_score
                        best_x = x.copy()
                else:
                    x[i] = old_val
                    
    return best_x
