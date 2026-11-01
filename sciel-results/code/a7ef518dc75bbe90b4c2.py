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

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin

    best_h = None
    best_val = float('inf')
    
    # Structural Change: Greedy Block Flip Hill Climbing
    # Instead of single-bit flips with SA, we use block flips with greedy acceptance.
    # This allows for larger structural changes to the sequence.
    
    max_restarts = 5000 # We will run as many as possible before deadline
    
    for restart in range(max_restarts):
        if time.time() >= deadline:
            break
            
        # Initialize sequence
        if best_h is not None and restart > 0:
            # Perturb the best solution found so far
            if np.random.random() < 0.7:
                h = best_h.copy()
                # Flip a small number of random blocks to escape local minima
                num_blocks = np.random.randint(1, 4)
                for _ in range(num_blocks):
                    L = np.random.randint(2, 8)
                    s = np.random.randint(0, N - L + 1)
                    h[s:s+L] *= -1
            else:
                h = np.random.choice([-1.0, 1.0], size=N)
        else:
            h = np.random.choice([-1.0, 1.0], size=N)
        
        ac = _compute_autocorr(h)
        current_val = _max_sidelobe(ac)
        
        if current_val < best_val:
            best_val = current_val
            best_h = h.copy()
            
        # Hill Climbing Loop
        # Limit steps per restart to ensure we get many restarts
        steps_per_restart = 200
        
        for step in range(steps_per_restart):
            if time.time() >= deadline:
                break
                
            # Pick a block to flip
            L = np.random.randint(2, 8) # Block length 2 to 7
            s = np.random.randint(0, N - L + 1)
            
            # Save original block
            orig_block = h[s:s+L].copy()
            
            # Flip block
            h[s:s+L] *= -1
            
            # Evaluate new state
            ac_new = _compute_autocorr(h)
            new_val = _max_sidelobe(ac_new)
            
            if new_val < current_val:
                # Accept
                current_val = new_val
                ac = ac_new
                if current_val < best_val:
                    best_val = current_val
                    best_h = h.copy()
            else:
                # Revert
                h[s:s+L] = orig_block
                # Keep old ac

    if best_h is None:
        return np.ones(N)
        
    return best_h
