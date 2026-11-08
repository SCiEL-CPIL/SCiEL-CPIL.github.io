import numpy as np
import time

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Initialize
    np.random.seed(42)
    h = np.random.choice([-1, 1], size=N)
    best_h = h.copy()
    
    # Compute initial autocorrelation
    def get_autocorr(h):
        H = np.fft.fft(h)
        R = np.fft.ifft(np.abs(H)**2).real
        return R
        
    R = get_autocorr(h)
    # Peak is max of |R[1:]|
    current_peak = np.max(np.abs(R[1:]))
    best_peak = current_peak
    
    # Local Search
    # We try to flip bits to reduce the peak
    stuck_count = 0
    while time.time() < deadline:
        # Pick a random index
        i = np.random.randint(N)
        old_h_i = h[i]
        
        # Flip
        h[i] = -old_h_i
        
        # Update R
        # Delta R_k = -2 * old_h_i * (h_{i+k} + h_{i-k})
        # Note: h is already flipped, so h_{i+k} is the new value.
        # We need the NEW h for the delta calculation?
        # Let's re-derive.
        # R_new = R_old + Delta.
        # Delta = R_new - R_old.
        # R_old had terms old_h_i * h_old_{i+k} and h_old_{i-k} * old_h_i.
        # R_new has terms new_h_i * h_new_{i+k} and h_new_{i-k} * new_h_i.
        # new_h_i = -old_h_i.
        # h_new_{i+k} = h_old_{i+k} (since k!=0, i+k != i).
        # So term 1 change: (-old_h_i * h_old_{i+k}) - (old_h_i * h_old_{i+k}) = -2 old_h_i h_old_{i+k}.
        # Term 2 change: (h_old_{i-k} * -old_h_i) - (h_old_{i-k} * old_h_i) = -2 h_old_{i-k} old_h_i.
        # So Delta = -2 old_h_i (h_old_{i+k} + h_old_{i-k}).
        # But h is currently NEW.
        # h_new_{i+k} = h_old_{i+k}.
        # h_new_{i-k} = h_old_{i-k}.
        # So we can use the current h (which is new) for the neighbors.
        
        delta = -2 * old_h_i * (np.roll(h, -i) + np.roll(h, i))
        R = R + delta
        
        # Check peak
        # We only care about k != 0
        new_peak = np.max(np.abs(R[1:]))
        
        if new_peak < best_peak:
            best_peak = new_peak
            best_h = h.copy()
            stuck_count = 0
        else:
            # Revert
            h[i] = old_h_i
            R = R - delta
            stuck_count += 1
            
            # If stuck for a long time, restart
            if stuck_count > 1000:
                h = np.random.choice([-1, 1], size=N)
                R = get_autocorr(h)
                current_peak = np.max(np.abs(R[1:]))
                if current_peak < best_peak:
                    best_peak = current_peak
                    best_h = h.copy()
                stuck_count = 0
                
    return best_h
