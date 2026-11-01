import numpy as np
import time

N = 400

def _compute_psl(h):
    """
    Compute the Peak Sidelobe Level (max abs autocorrelation for lags 1..N-1).
    h: array of length N with values in {-1, 1}
    """
    n_fft = 2 * N
    F = np.fft.rfft(h, n=n_fft)
    power = F * np.conj(F)
    ac = np.fft.irfft(power, n=n_fft)
    # ac[0] is the peak (N). ac[1:N] are the sidelobes.
    # We take the max absolute value of the sidelobes.
    return np.max(np.abs(ac[1:N]))

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    best_h = None
    best_psl = float('inf')
    
    t_start = 10.0
    t_end = 0.01
    
    # We will run multiple SA instances.
    # Each instance runs for a fixed time slice, then we restart.
    # This helps escape local minima.
    
    while time.time() < deadline - 5.0:
        # Initialize random sequence
        h = np.random.choice([-1.0, 1.0], size=N)
        
        # Run SA for this instance
        t_instance_start = time.time()
        t_instance_limit = 5.0
        
        current_psl = _compute_psl(h)
        best_local_h = h.copy()
        best_local_psl = current_psl
        
        max_iter = 20000
        for i in range(max_iter):
            if time.time() - t_instance_start > t_instance_limit:
                break
                
            # Temperature: Exponential cooling
            # Rate adjusted so T drops significantly over the iteration count
            T = t_start * np.exp(-i / 3000.0)
            
            # Move selection
            r = np.random.random()
            move_type = 0
            move_data = None
            
            if r < 0.5:
                # Flip single bit
                idx = np.random.randint(N)
                h[idx] *= -1
                move_type = 1
                move_data = (idx,)
            elif r < 0.8:
                # Swap two bits
                idx1, idx2 = np.random.randint(N, size=2)
                if idx1 != idx2:
                    h[idx1], h[idx2] = h[idx2], h[idx1]
                    move_type = 2
                    move_data = (idx1, idx2)
                else:
                    # Invalid move, skip
                    continue
            else:
                # Flip a block
                length = np.random.randint(2, 6)
                start = np.random.randint(0, N - length + 1)
                h[start:start+length] *= -1
                move_type = 3
                move_data = (start, length)
            
            new_psl = _compute_psl(h)
            
            if new_psl < current_psl:
                current_psl = new_psl
                if new_psl < best_local_psl:
                    best_local_psl = new_psl
                    best_local_h = h.copy()
            else:
                delta = new_psl - current_psl
                # Accept with probability exp(-delta/T)
                # If T is very small, exp(-large) is 0
                if np.random.random() < np.exp(-delta / T):
                    current_psl = new_psl
                else:
                    # Revert the move
                    if move_type == 1:
                        h[move_data[0]] *= -1
                    elif move_type == 2:
                        h[move_data[0]], h[move_data[1]] = h[move_data[1]], h[move_data[0]]
                    elif move_type == 3:
                        h[move_data[0]:move_data[0]+move_data[1]] *= -1
                        
        if best_local_psl < best_psl:
            best_psl = best_local_psl
            best_h = best_local_h
            
    # Final local search (greedy hill climbing) on the best result
    if best_h is not None:
        h_final = best_h.copy()
        current_psl = best_psl
        improved = True
        # Run a few passes of greedy improvement
        while improved and (time.time() < deadline - 1.0):
            improved = False
            for i in range(N):
                if time.time() >= deadline - 1.0:
                    break
                h_final[i] *= -1
                new_psl = _compute_psl(h_final)
                if new_psl < current_psl:
                    current_psl = new_psl
                    improved = True
                else:
                    h_final[i] *= -1 # Revert
                    
        if current_psl < best_psl:
            best_psl = current_psl
            best_h = h_final

    if best_h is None:
        return np.ones(N)
        
    return best_h
