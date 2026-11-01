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
    # Sidelobes are at lags 1..N-1 and -(N-1)..-1
    # ac[1:N] are positive lags 1..N-1
    # ac[N+1:2*N] are negative lags -(N-1)..-1 (indices N+1 to 2N-1)
    # ac[N] is lag -N which is 0
    pos_lags = ac[1:N]
    neg_lags = ac[N+1:2*N]
    return np.max(np.concatenate([np.abs(pos_lags), np.abs(neg_lags)]))

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin

    best_h = None
    best_val = float('inf')
    
    # Simulated Annealing Parameters
    # T0: Initial temperature. Should be large enough to accept bad moves.
    # T_end: Final temperature.
    # alpha: Cooling rate.
    # We will run multiple independent SA runs.
    
    num_restarts = 50 # We will adjust this dynamically based on time
    
    for restart in range(num_restarts):
        if time.time() >= deadline:
            break
            
        # Initialize random sequence
        h = np.random.choice([-1.0, 1.0], size=N)
        ac = _compute_autocorr(h)
        current_val = _max_sidelobe(ac)
        
        if current_val < best_val:
            best_val = current_val
            best_h = h.copy()
            
        # SA Parameters for this run
        # Estimate time per step. 
        # O(N) update + O(N) max. ~800 ops. Very fast.
        # Let's aim for ~10,000 steps per restart initially, or adapt.
        
        T = 10.0 # Initial temp
        T_min = 0.01
        alpha = 0.999 # Cooling factor
        
        # We will run a fixed number of steps or until time is up
        # Let's say 5000 steps per restart.
        max_steps = 5000
        
        for step in range(max_steps):
            if time.time() >= deadline:
                break
                
            # Pick random index
            i = np.random.randint(N)
            
            # Calculate delta in autocorrelation
            # Flip h[i] -> -h[i]
            # Delta C[k] = -2 * h[i] * (h[i+k] + h[i-k]) for valid indices
            
            # We need to update ac array
            # It is safer to create a copy or update in place and revert if needed.
            # Since we need to check if it's better, we can compute the new max.
            
            # To avoid copying the whole ac array (size 800), we can
            # just compute the change in the max? No, max is global.
            # We must update ac.
            
            # Optimization: 
            # We only need to update the parts of ac that change.
            # But ac is used to find max.
            # Let's just update ac in place.
            
            old_h_i = h[i]
            h[i] = -old_h_i
            
            # Update ac
            # For each k in 1..N-1:
            # ac[k] += -2 * old_h_i * h[i+k]  (if i+k < N)
            # ac[N+k] += -2 * old_h_i * h[i-k] (if i-k >= 0) -- Wait, ac[N+k] corresponds to lag -k?
            # Let's check indices.
            # ac[0] is lag 0.
            # ac[k] is lag k.
            # ac[N+k] is lag -k?
            # In FFT of size 2N:
            # Output index m corresponds to lag m if m < N, and m-N if m >= N.
            # So ac[N+k] is lag k-N? No.
            # Standard linear convolution/autocorrelation via FFT:
            # If we zero-pad to 2N, the result is circular autocorrelation of length 2N.
            # For a sequence of length N, the linear autocorrelation is embedded.
            # Lag 0 is at 0.
            # Lag k (1..N-1) is at k.
            # Lag -k (-(N-1)..-1) is at 2N-k.
            
            # So:
            # Positive lag k (1..N-1): index k.
            # Negative lag -k: index 2N-k.
            
            # Change in C[k] (lag k):
            # Term j=i: h[i]h[i+k] -> -h[i]h[i+k]. Delta: -2 h[i] h[i+k].
            # Term j=i-k: h[i-k]h[i] -> h[i-k](-h[i]). Delta: -2 h[i] h[i-k].
            # So Delta C[k] = -2 * old_h_i * (h[i+k] + h[i-k])
            
            # We need to update ac[k] and ac[2N-k].
            
            # Loop over k
            # This is O(N) python loop, which is slow.
            # Vectorized update?
            # The indices i+k and i-k wrap around or go out of bounds.
            
            # Let's try to vectorize.
            # k ranges from 1 to N-1.
            # idx_pos = i + k. Valid if < N.
            # idx_neg = i - k. Valid if >= 0.
            
            # This is tricky to vectorize cleanly without masks.
            # Given N=400, a python loop of 400 iterations is ~0.1-0.2 ms.
            # 5000 steps * 0.2ms = 1s. This is fine.
            
            # Let's do the python loop for correctness and simplicity first.
            # If too slow, we optimize.
            
            # Store old values to revert if needed?
            # We can just compute new_val, and if we don't accept, we revert h[i] and ac.
            
            # To revert ac, we need to subtract the same delta.
            
            deltas = np.zeros(2*N) # This is allocation overhead.
            
            # Better: apply changes, compute new max, decide.
            
            # Apply changes
            for k in range(1, N):
                val = 0.0
                if i + k < N:
                    val += h[i+k]
                if i - k >= 0:
                    val += h[i-k]
                
                delta = -2.0 * old_h_i * val
                ac[k] += delta
                ac[2*N - k] += delta # Symmetry
                
            new_val = _max_sidelobe(ac)
            
            # Acceptance criterion
            if new_val < current_val:
                current_val = new_val
                if current_val < best_val:
                    best_val = current_val
                    best_h = h.copy()
            else:
                # Probabilistic acceptance
                delta_E = new_val - current_val
                # If delta_E is 0, accept (or random). Let's accept if < T.
                if T > 0:
                    # Avoid overflow in exp
                    if delta_E / T < 50: # exp(-50) is tiny
                        prob = np.exp(-delta_E / T)
                        if np.random.random() < prob:
                            current_val = new_val
                            # Don't update best_h here unless it's better, 
                            # but current_val is worse than best_val usually in SA.
                            # We only update best_h if we find a global improvement.
                        else:
                            # Revert
                            h[i] = old_h_i
                            for k in range(1, N):
                                val = 0.0
                                if i + k < N:
                                    val += h[i+k]
                                if i - k >= 0:
                                    val += h[i-k]
                                delta = -2.0 * old_h_i * val
                                ac[k] -= delta
                                ac[2*N - k] -= delta
                    else:
                        # Revert
                        h[i] = old_h_i
                        for k in range(1, N):
                            val = 0.0
                            if i + k < N:
                                val += h[i+k]
                            if i - k >= 0:
                                val += h[i-k]
                            delta = -2.0 * old_h_i * val
                            ac[k] -= delta
                            ac[2*N - k] -= delta
                else:
                    # Revert
                    h[i] = old_h_i
                    for k in range(1, N):
                        val = 0.0
                        if i + k < N:
                            val += h[i+k]
                        if i - k >= 0:
                            val += h[i-k]
                        delta = -2.0 * old_h_i * val
                        ac[k] -= delta
                        ac[2*N - k] -= delta
            
            # Cool down
            T *= alpha
            if T < T_min:
                T = T_min

    if best_h is None:
        return np.ones(N)
        
    return best_h
