"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

TIME_LIMIT = 145  # safety margin before 150s kill


def _compute_c3_bound(heights):
    """
    Compute the upper bound on C3 from the autoconvolution of the signed step function.
    """
    h = np.asarray(heights, dtype=np.float64)
    n = len(h)
    dx = 0.5 / n
    
    # Linear autoconvolution via FFT
    h_pad = np.zeros(2 * n - 1)
    h_pad[:n] = h
    f_fft = np.fft.fft(h_pad, n=2 * (2 * n - 1))
    conv = np.fft.ifft(f_fft * np.conj(f_fft)).real
    
    max_abs = np.max(np.abs(conv))
    c3_bound = dx * max_abs
    
    return c3_bound

def _local_search_efficient(heights, time_budget=10):
    """
    Efficient local search using incremental autoconvolution updates.
    """
    h = np.array(heights, dtype=np.float64)
    n = len(h)
    dx = 0.5 / n
    
    # Compute initial autoconvolution
    h_pad = np.zeros(2 * n - 1)
    h_pad[:n] = h
    f_fft = np.fft.fft(h_pad, n=2 * (2 * n - 1))
    conv = np.fft.ifft(f_fft * np.conj(f_fft)).real
    current_max_abs = np.max(np.abs(conv))
    current_bound = dx * current_max_abs
    
    start_time = time.time()
    improved = True
    
    while improved:
        if time.time() - start_time > time_budget:
            break
        
        improved = False
        best_gain = 0
        best_idx = -1
        
        for i in range(n):
            # Calculate change in autoconvolution if we flip h[i]
            # delta C[k] = -2 h[i] h[i+k] - 2 h[i-k] h[i]
            # We need to find the new max |C[k]|
            
            # To do this efficiently, we can create a delta vector
            # But allocating a new array for each i is expensive.
            # Instead, we can just compute the new max by updating a copy of conv?
            # Or better: just recompute the max for the modified conv.
            
            # Let's try a simpler approach:
            # Since N=400, we can just update conv in place for the specific i,
            # find the new max, and revert.
            
            # Indices affected: k such that i+k is in [0, n-1] or i-k is in [0, n-1]
            # In the conv array (length 2n-1), index j corresponds to lag j-(n-1).
            # Let's work with lags directly to avoid confusion, or just indices in conv.
            
            # conv index p corresponds to sum over j h[j] h[j + (p - (n-1))]
            # Let lag = p - (n-1).
            # Term 1: j=i, lag = (i+k) - i = k. Index p1 = i+k - (n-1) + (n-1) = i+k? No.
            # Let's stick to the lag definition.
            # C_lag = sum_j h[j] h[j+lag].
            # If we flip h[i], C_lag changes by:
            # -2 h[i] h[i+lag] (if 0 <= i+lag < n)
            # -2 h[i-lag] h[i] (if 0 <= i-lag < n)
            
            # We can update conv array.
            # conv array index p is C_{p-(n-1)}.
            
            # Let's just make a copy of conv, update it, and find max.
            # This is O(N) copy + O(N) update + O(N) max = O(N).
            # Total per pass O(N^2).
            
            conv_temp = conv.copy()
            hi = h[i]
            
            # Update lags where i+lag is valid
            # lag ranges from -(n-1) to n-1
            # i+lag in [0, n-1] => lag in [-i, n-1-i]
            lags1 = np.arange(-i, n - i)
            vals1 = h[lags1 + i] # h[i+lag]
            # conv index for lag L is L + (n-1)
            idx1 = lags1 + (n - 1)
            conv_temp[idx1] -= 2 * hi * vals1
            
            # Update lags where i-lag is valid
            # i-lag in [0, n-1] => lag in [i-(n-1), i]
            lags2 = np.arange(i - (n - 1), i + 1)
            vals2 = h[i - lags2] # h[i-lag]
            idx2 = lags2 + (n - 1)
            conv_temp[idx2] -= 2 * hi * vals2
            
            # Note: if lag=0, we added -2*hi*hi twice.
            # But for lag=0, the change should be 0.
            # Let's check:
            # Term 1 for lag 0: -2 hi h[i] = -2 hi^2.
            # Term 2 for lag 0: -2 h[i] hi = -2 hi^2.
            # Total -4 hi^2.
            # But C[0] = sum h[j]^2. Flipping sign doesn't change sum of squares.
            # So we must add 4 hi^2 back to conv_temp[0 + n - 1]?
            # Or just handle lag 0 separately.
            
            # Let's check the formula again.
            # C_new[0] = C_old[0].
            # Our update gave -4 hi^2.
            # So we need to add 4 hi^2 to conv_temp[n-1].
            
            conv_temp[n - 1] += 4 * hi * hi
            
            new_max_abs = np.max(np.abs(conv_temp))
            new_bound = dx * new_max_abs
            
            gain = current_bound - new_bound
            if gain > best_gain:
                best_gain = gain
                best_idx = i
        
        if best_idx >= 0:
            # Apply the flip
            h[best_idx] = -h[best_idx]
            current_bound -= best_gain
            
            # Update conv array permanently
            hi = -h[best_idx] # old value was h[best_idx], new is -h[best_idx].
            # Wait, we need to update conv from old state to new state.
            # The delta calculated was for flipping h[best_idx] (which was val) to -val.
            # So we apply the same delta to the main conv array.
            
            # Re-apply the delta logic to conv
            val = h[best_idx] # This is the NEW value (-old).
            # We want to go from OLD state to NEW state.
            # The delta computed above was: New - Old.
            # So we add that delta to conv.
            
            # Let's re-calculate delta for the flip of index best_idx
            # Old value was -val (since we just flipped it? No, h[best_idx] is now -old_val).
            # Let's just use the fact that we found the best flip.
            # We can just recompute conv from scratch? No, that defeats purpose.
            # Let's apply the delta we calculated.
            # The calculation in the loop used `hi = h[i]` (the OLD value).
            # So the delta was computed correctly for the flip.
            # We just need to apply that specific delta to `conv`.
            
            # But we didn't store the delta for best_idx.
            # We can just re-compute the delta for best_idx and apply it.
            
            hi_old = -h[best_idx] # The value before flip
            # Delta for flipping hi_old to -hi_old
            lags1 = np.arange(-best_idx, n - best_idx)
            vals1 = h[lags1 + best_idx]
            # Note: h has been updated. So h[lags1+best_idx] is the NEW h.
            # The formula used OLD h for the neighbors?
            # "conv_temp[idx1] -= 2 * hi * vals1"
            # vals1 was taken from `h` (which was old state in the loop).
            # Now `h` is new state.
            # So we must be careful.
            
            # Easiest safe path: Since N=400, just recompute conv from scratch if we improve?
            # O(N log N) is cheap.
            h_pad = np.zeros(2 * n - 1)
            h_pad[:n] = h
            f_fft = np.fft.fft(h_pad, n=2 * (2 * n - 1))
            conv = np.fft.ifft(f_fft * np.conj(f_fft)).real
            current_max_abs = np.max(np.abs(conv))
            current_bound = dx * current_max_abs
            
            improved = True
    
    return h, current_bound


def _thue_morse(n):
    """Generate Thue-Morse sequence of length n as +/-1 values."""
    seq = [1]
    while len(seq) < n:
        seq += [-x for x in seq]
    return np.array(seq[:n], dtype=np.float64)


def construct():
    start_time = time.time()
    
    # Initialize with multiple starting points
    best_h = np.ones(N)
    best_bound = _compute_c3_bound(best_h)
    
    # Try Thue-Morse sequence
    tm = _thue_morse(N)
    tm_bound = _compute_c3_bound(tm)
    if tm_bound < best_bound:
        best_bound = tm_bound
        best_h = tm.copy()
    
    # Try random initializations
    np.random.seed(42)
    for trial in range(20):
        if time.time() - start_time > TIME_LIMIT - 30:
            break
        
        h = np.random.choice([-1.0, 1.0], size=N)
        bound = _compute_c3_bound(h)
        if bound < best_bound:
            best_bound = bound
            best_h = h.copy()
    
    # Local search from best initialization
    remaining = TIME_LIMIT - (time.time() - start_time)
    if remaining > 15:
        local_time = min(remaining - 10, 100)
        best_h, best_bound = _local_search_efficient(best_h, time_budget=local_time)
    
    # Additional restarts if time permits
    remaining = TIME_LIMIT - (time.time() - start_time)
    if remaining > 30:
        for trial in range(10):
            if time.time() - start_time > TIME_LIMIT - 10:
                break
            
            h = np.random.choice([-1.0, 1.0], size=N)
            bound = _compute_c3_bound(h)
            
            if bound < best_bound:
                local_time = min(TIME_LIMIT - (time.time() - start_time) - 5, 20)
                h, bound = _local_search_efficient(h, time_budget=local_time)
                if bound < best_bound:
                    best_bound = bound
                    best_h = h.copy()
    
    # Final verification
    final_bound = _compute_c3_bound(best_h)
    
    return best_h


# Test
if __name__ == "__main__":
    h = construct()
    print(f"Length: {len(h)}")
    print(f"First 20: {h[:20]}")
    print(f"C3 bound: {_compute_c3_bound(h)}")
    print(f"Score: {1.0 / _compute_c3_bound(h)}")
# EVOLVE-BLOCK-END
