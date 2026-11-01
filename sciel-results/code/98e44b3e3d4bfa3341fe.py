import numpy as np
import time

N = 400

def _compute_autocorr(h):
    """
    Compute the linear aperiodic autocorrelation using FFT.
    Returns an array of length 2*N where indices 0..N-1 are lags 0..N-1
    and indices N..2N-1 are lags -N..-1.
    Note: For real input, C[k] = C[-k].
    """
    n_fft = 2 * N
    F = np.fft.rfft(h, n=n_fft)
    power = F * np.conj(F)
    ac = np.fft.irfft(power, n=n_fft)
    return ac

def _max_sidelobe(ac):
    """
    Compute max absolute value of sidelobes from autocorrelation array.
    ac: numpy array of length 2*N.
    """
    # Due to symmetry of autocorrelation for real signals, C[k] = C[-k].
    # Thus, max |C[k]| for k=1..N-1 is sufficient.
    pos_lags = ac[1:N]
    return np.max(np.abs(pos_lags))

def _get_worst_lag(ac):
    """
    Get the lag index (1 to N-1) with the maximum absolute sidelobe value.
    """
    pos_lags = ac[1:N]
    abs_vals = np.abs(pos_lags)
    worst_idx = np.argmax(abs_vals)
    return worst_idx + 1  # Return actual lag (1-indexed)

def construct():
    start_time = time.time()
    deadline = start_time + 147.0  # Safety margin before 150s kill

    best_h = None
    best_val = float('inf')
    
    # Simulated Annealing parameters
    T_initial = 50.0
    T_final = 0.1
    cooling_rate = 0.9995
    
    max_restarts = 1000
    
    for restart in range(max_restarts):
        if time.time() >= deadline:
            break
            
        # Initialize sequence
        if best_h is not None and restart > 0:
            # Start from best with perturbation
            h = best_h.copy()
            # Random perturbation: flip 5-15 random elements
            num_flips = np.random.randint(5, 16)
            flip_indices = np.random.choice(N, size=num_flips, replace=False)
            h[flip_indices] *= -1
        else:
            h = np.random.choice([-1.0, 1.0], size=N)
        
        ac = _compute_autocorr(h)
        current_val = _max_sidelobe(ac)
        
        if current_val < best_val:
            best_val = current_val
            best_h = h.copy()
            
        # Simulated Annealing loop
        T = T_initial
        steps_per_restart = 500
        
        for step in range(steps_per_restart):
            if time.time() >= deadline:
                break
            
            # Generate move: flip a random block
            # Block size varies: larger when T is high, smaller when T is low
            if T > 10.0:
                L = np.random.randint(3, 12)
            elif T > 1.0:
                L = np.random.randint(2, 8)
            else:
                L = np.random.randint(1, 4)
            
            s = np.random.randint(0, N - L + 1)
            
            # Save original block
            orig_block = h[s:s+L].copy()
            
            # Flip block
            h[s:s+L] *= -1
            
            # Evaluate new state
            ac_new = _compute_autocorr(h)
            new_val = _max_sidelobe(ac_new)
            
            delta = new_val - current_val
            
            # Acceptance criterion: always accept if better,
            # accept worse with probability exp(-delta/T)
            if delta < 0 or np.random.random() < np.exp(-delta / T):
                # Accept
                current_val = new_val
                ac = ac_new
                if current_val < best_val:
                    best_val = current_val
                    best_h = h.copy()
            else:
                # Revert
                h[s:s+L] = orig_block
            
            # Cool down
            T *= cooling_rate
        
        # Targeted refinement: focus on the worst lag
        if time.time() < deadline - 3.0:
            # Find worst lag and try to reduce it by flipping elements
            # that contribute to that lag
            ac_final = _compute_autocorr(h)
            worst_lag = _get_worst_lag(ac_final)
            
            # Try flipping individual elements and check if worst lag improves
            for polish_step in range(100):
                if time.time() >= deadline:
                    break
                
                idx = np.random.randint(0, N)
                orig_bit = h[idx]
                h[idx] *= -1
                
                ac_new = _compute_autocorr(h)
                new_val = _max_sidelobe(ac_new)
                
                if new_val < current_val:
                    current_val = new_val
                    ac = ac_new
                    if current_val < best_val:
                        best_val = current_val
                        best_h = h.copy()
                else:
                    h[idx] = orig_bit

    if best_h is None:
        return np.ones(N)
        
    return best_h
