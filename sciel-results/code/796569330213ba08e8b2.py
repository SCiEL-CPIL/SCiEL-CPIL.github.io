import numpy as np
import time

N = 400
TIME_LIMIT = 115  # Safety margin before 120s kill

# Precompute FFT size for efficiency
# We need linear autocorrelation of length N.
# Convolution length is 2N - 1.
# Next power of 2 >= 2N - 1 = 799 is 1024.
FFT_SIZE = 1024

def _compute_acf(h):
    """
    Compute the aperiodic autocorrelation of h.
    Returns array of length N where acf[k] = sum_{i=0}^{N-1-k} h[i] h[i+k] for k=0..N-1.
    Note: This is the 'positive lag' part. The full autocorrelation is symmetric.
    """
    # Pad to FFT_SIZE
    h_pad = np.zeros(FFT_SIZE)
    h_pad[:N] = h
    
    # FFT
    H = np.fft.fft(h_pad)
    # Power spectrum
    P = H * np.conj(H)
    # Inverse FFT gives circular autocorrelation of the padded sequence
    # Since we padded with zeros, the first N-1 lags correspond to aperiodic autocorrelation
    acf = np.fft.ifft(P).real
    
    # Due to numerical errors, take real part. 
    # The result acf[k] for k=0..N-1 is the sum h[i]h[i+k].
    # The result acf[FFT_SIZE-k] is the sum h[i]h[i-k] which is the same by symmetry.
    # We only care about lags 1 to N-1 for the max absolute value (excluding lag 0 which is N).
    return acf[:N]

def _score(h):
    """
    Compute the upper bound on C3.
    This is defined as the maximum absolute value of the autocorrelation for lags 1 to N-1.
    Note: The prompt mentions "upper bound on C3". 
    Based on inspiration 1, the bound is related to the max autocorrelation.
    Specifically, for the third autocorrelation inequality, the quantity of interest is often 
    the max off-diagonal autocorrelation.
    Let's assume the score is simply max_{1<=k<N} |A_k|.
    """
    acf = _compute_acf(h)
    # Exclude lag 0
    if N > 1:
        return np.max(np.abs(acf[1:N]))
    return 0.0

def _fast_flip_score(h, idx):
    """
    Estimate the new score after flipping h[idx] without full FFT.
    This is complex to do exactly without recomputing the full ACF.
    However, we can update the ACF incrementally.
    A_k = sum_{i=0}^{N-1-k} h[i] h[i+k]
    If we flip h[j], then:
    A_k changes by:
    - h[j] h[j+k] if j+k < N  (term where i=j)
    - h[j] h[j-k] if j-k >= 0 (term where i+k=j => i=j-k)
    - Also, if j is the second index, i.e., i+k=j => i=j-k, covered above.
    - If j is the first index, i=j, covered above.
    
    Let's derive the delta for each lag k.
    Old A_k = sum h[i] h[i+k]
    New h'[j] = -h[j], h'[i] = h[i] for i != j.
    
    Delta A_k = New A_k - Old A_k
    
    Terms involving h[j]:
    1. i = j: term is h[j] h[j+k] (if j+k < N). New term is -h[j] h[j+k]. Delta: -2 h[j] h[j+k].
    2. i+k = j => i = j-k: term is h[j-k] h[j] (if j-k >= 0). New term is h[j-k] (-h[j]). Delta: -2 h[j-k] h[j].
    
    So Delta A_k = -2 h[j] ( h[j+k] * (j+k<N) + h[j-k] * (j-k>=0) )
    
    We can precompute the current ACF.
    Then for a flip at j, we can update the ACF array in O(N).
    Then we can find the max abs value.
    This is O(N) per flip evaluation, O(N^2) per full pass.
    With N=400, N^2 = 160,000 operations. This is very fast.
    """
    # We pass the current ACF to avoid recomputation
    # But for simplicity and correctness first, let's just recompute FFT in the main loop.
    # O(N log N) ~ 400 * 9 ~ 3600 ops.
    # O(N) update ~ 400 ops.
    # The update is faster. Let's implement the update.
    pass

def _local_search(h, max_passes=1000, time_limit=100.0):
    """
    Perform local search by flipping bits.
    Uses incremental ACF update for speed.
    """
    start_time = time.time()
    current_h = h.copy()
    current_acf = _compute_acf(current_h)
    current_score = np.max(np.abs(current_acf[1:N]))
    
    best_h = current_h.copy()
    best_score = current_score
    
    improved = True
    passes = 0
    
    while improved and passes < max_passes and (time.time() - start_time) < time_limit:
        improved = False
        passes += 1
        
        # Try flipping each bit
        best_gain = 1e-9 # Small epsilon to avoid floating point noise
        best_idx = -1
        
        for j in range(N):
            # Calculate delta for each lag k
            # Delta A_k = -2 * h[j] * (h[j+k] * (j+k<N) + h[j-k] * (j-k>=0))
            
            # We need to update the ACF array to see the new max.
            # Instead of updating the whole array, we can just compute the new max.
            # But the max depends on all k.
            # Let's update a copy of the ACF? No, that's O(N) memory copy.
            # Let's compute the new ACF values for all k in a vectorized way.
            
            # h[j] is constant for this flip
            hj = current_h[j]
            
            # Vector k from 0 to N-1
            # Term 1: h[j+k] for j+k < N
            # Term 2: h[j-k] for j-k >= 0
            
            # Let's construct the delta array
            delta_acf = np.zeros(N)
            
            # Mask for j+k < N => k < N-j
            mask1 = np.arange(N) < (N - j)
            # Indices for h[j+k] are j to N-1
            # k = 0..N-j-1
            # h[j+k] corresponds to current_h[j : N]
            # delta_acf[0 : N-j] += -2 * hj * current_h[j : N]
            
            # Mask for j-k >= 0 => k <= j
            mask2 = np.arange(N) <= j
            # Indices for h[j-k] are j, j-1, ..., 0
            # k = 0..j
            # h[j-k] corresponds to current_h[j::-1]
            # delta_acf[0 : j+1] += -2 * hj * current_h[j::-1]
            
            # Apply updates
            if N - j > 0:
                delta_acf[:N-j] -= 2 * hj * current_h[j:N]
            if j + 1 > 0:
                delta_acf[:j+1] -= 2 * hj * current_h[j::-1]
            
            # New ACF = current_acf + delta_acf
            new_acf = current_acf + delta_acf
            
            # Compute new score (max abs for lags 1..N-1)
            if N > 1:
                new_score = np.max(np.abs(new_acf[1:N]))
            else:
                new_score = 0.0
            
            gain = current_score - new_score
            
            if gain > best_gain:
                best_gain = gain
                best_idx = j
        
        if best_idx >= 0:
            # Apply the best flip
            current_h[best_idx] *= -1
            # Update ACF
            # Re-use the delta calculation? 
            # It's safer to recompute or apply the delta we found.
            # Let's apply the delta for best_idx.
            hj = current_h[best_idx] # This is the NEW value? No, current_h was flipped.
            # Wait, I flipped current_h[best_idx] above.
            # The delta was calculated based on the OLD h[best_idx].
            # Let's just recompute ACF for simplicity and safety, it's O(N log N).
            # Or better, store the delta.
            
            # Let's just recompute ACF. It's fast enough.
            current_acf = _compute_acf(current_h)
            current_score = np.max(np.abs(current_acf[1:N]))
            
            if current_score < best_score:
                best_score = current_score
                best_h = current_h.copy()
            improved = True
    
    return best_h, best_score

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    # Initialize with random sequence
    # Use a fixed seed for reproducibility in local tests, but random in eval?
    # The prompt implies we should find a good one.
    # Let's try a few random starts and pick the best.
    
    best_h = None
    best_score = np.inf
    
    # Try 5 random initializations
    for trial in range(5):
        if time.time() > deadline - 10:
            break
        
        h = np.random.choice([-1.0, 1.0], size=N)
        score = _score(h)
        
        if score < best_score:
            best_score = score
            best_h = h.copy()
    
    # Local search from the best start
    if best_h is not None and time.time() < deadline - 5:
        time_left = deadline - time.time() - 5
        best_h, best_score = _local_search(best_h, max_passes=10000, time_limit=time_left)
    
    # If we have time, try another random start + local search
    if time.time() < deadline - 15:
        h = np.random.choice([-1.0, 1.0], size=N)
        score = _score(h)
        if score < best_score + 0.1: # Only if it's close
            time_left = deadline - time.time() - 5
            h2, score2 = _local_search(h, max_passes=5000, time_limit=time_left)
            if score2 < best_score:
                best_score = score2
                best_h = h2
    
    # Final safety check
    if best_h is None:
        best_h = np.ones(N)
    
    return best_h

if __name__ == "__main__":
    h = construct()
    print(f"Score: {_score(h)}")
