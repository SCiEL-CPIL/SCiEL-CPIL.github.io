import numpy as np
import time

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 115.0 # 5s margin
    
    # Initialize with random signed steps
    # Using a fixed seed for reproducibility in local testing, 
    # but the evaluator might run multiple times. 
    # Randomness is usually fine for this type of search.
    np.random.seed(42) # Optional, but good for stability. 
    # Actually, let's not seed it if we want diversity, 
    # but for a single run, a good initial guess helps.
    # Let's try a few different random starts if time permits, 
    # or just one very good local search.
    
    f = np.random.choice([-1, 1], N)
    
    def compute_autocorr(f):
        # Aperiodic autocorrelation
        # R[k] = sum_{i=0}^{N-1-k} f[i] f[i+k]
        # Using convolution: conv(f, f[::-1])
        # The result has length 2N-1.
        # The center is at N-1.
        # R[k] corresponds to index N-1+k in the convolution.
        # R[-k] corresponds to index N-1-k.
        conv = np.convolve(f, f[::-1])
        # Extract lags -N+1 to N-1
        # We only care about k != 0
        # lags: -N+1, ..., -1, 1, ..., N-1
        # Indices in conv:
        # k=0 is at N-1
        # k=1 is at N
        # k=-1 is at N-2
        # We want max abs of all except k=0
        
        # Create a mask or just slice
        # Conv indices: 0 to 2N-2
        # Center index: N-1
        # We exclude center index.
        # 
        # Let's just take the whole conv and set the center to 0
        R = conv.copy()
        R[N-1] = 0.0
        return R
    
    R = compute_autocorr(f)
    current_max_abs = np.max(np.abs(R))
    best_f = f.copy()
    best_R = R.copy()
    best_max_abs = current_max_abs
    
    while time.time() < deadline:
        # Find the lag with the maximum absolute autocorrelation
        k_max = np.argmax(np.abs(R))
        
        # Map k_max from conv index to lag value
        # k_max is index in [0, 2N-2]
        # lag = k_max - (N-1)
        lag = k_max - (N - 1)
        
        # If the max is 0 or very small, we might be stuck or done
        if np.abs(R[k_max]) < 1e-6:
            break
            
        # Determine which flips would reduce the magnitude of R[lag]
        # R[lag] = sum f[i] f[i+lag]
        # If R[lag] > 0, we want to change signs of pairs with product 1 to -1.
        # If R[lag] < 0, we want to change signs of pairs with product -1 to 1.
        
        sign = np.sign(R[k_max])
        
        # Candidates are indices i such that flipping f[i] changes R[lag] in the desired direction.
        # Flipping f[i] changes R[lag] by:
        # -2 * f[i] * f[i+lag] (if i+lag < N)
        # -2 * f[i] * f[i-lag] (if i-lag >= 0)
        
        # We want the change to be opposite to the sign of R[lag].
        # Change = -2 * f[i] * f[j]
        # We want Change * sign < 0 => -2 * f[i] * f[j] * sign < 0 => f[i]*f[j]*sign > 0
        # If sign > 0, we want f[i]*f[j] > 0 (product 1).
        # If sign < 0, we want f[i]*f[j] < 0 (product -1).
        
        candidates = []
        
        # Check pairs involved in lag 'lag'
        # Case 1: i + lag = j, j < N => i < N - lag
        # Case 2: i - lag = j, j >= 0 => i >= lag
        
        # It's easier to iterate through all i and check if flipping i helps.
        # But N=400, so 400 checks per iteration is fast.
        
        best_improvement = 0
        best_flip_idx = -1
        best_new_R = None
        
        for i in range(N):
            # Calculate change in R for all lags if we flip f[i]
            # This is O(N) to compute new R, but we can do it incrementally or just check the specific lag first.
            
            # Check if this flip reduces the current peak R[k_max]
            # Contribution of f[i] to R[lag]:
            c1 = 0
            c2 = 0
            if 0 <= i + lag < N:
                c1 = f[i] * f[i + lag]
            if 0 <= i - lag < N:
                c2 = f[i] * f[i - lag]
            
            # Total contribution to R[lag] from f[i] is c1 + c2
            # Flipping f[i] changes R[lag] by -2 * (c1 + c2)
            
            delta_R_lag = -2 * (c1 + c2)
            
            # We want the new R[lag] to be closer to 0.
            # New R[lag] = R[lag] + delta_R_lag
            # We want |R[lag] + delta_R_lag| < |R[lag]|
            # This implies delta_R_lag has opposite sign to R[lag] and |delta| < 2|R|
            
            if R[k_max] * delta_R_lag < 0:
                # This flip reduces the magnitude of the current peak.
                # Now we need to check if it increases other peaks.
                
                # To be efficient, we can estimate the new max.
                # But calculating the full new R is O(N).
                # Let's just do it for the most promising candidates.
                # How to prioritize?
                # Flips that reduce the peak the most are likely best.
                # delta_R_lag is either -4, -2, 0, 2, 4.
                # We want delta_R_lag to be large in magnitude and opposite sign.
                
                # Let's collect top candidates based on reduction of current peak.
                # Actually, just trying all 400 is fine.
                
                # Simulate flip
                f[i] *= -1
                R_new = compute_autocorr(f)
                max_new = np.max(np.abs(R_new))
                
                if max_new < best_max_abs - 1e-9:
                    best_max_abs = max_new
                    best_flip_idx = i
                    best_new_R = R_new.copy()
                    best_f = f.copy()
                
                # Undo flip
                f[i] *= -1
        
        if best_flip_idx == -1:
            # No single bit flip improved the global max.
            # We might be in a local minimum.
            # Try a random restart or a 2-bit flip?
            # Given time constraints, a random restart might be better.
            # Or just break.
            # Let's try a few random flips to escape local minima.
            escape_success = False
            for _ in range(10):
                idx = np.random.randint(N)
                f[idx] *= -1
                R_new = compute_autocorr(f)
                max_new = np.max(np.abs(R_new))
                if max_new < best_max_abs:
                    best_max_abs = max_new
                    best_f = f.copy()
                    best_R = R_new.copy()
                    R = R_new
                    f = best_f.copy() # Sync f with best
                    escape_success = True
                    break
                else:
                    f[idx] *= -1 # Undo
            
            if not escape_success:
                # Stuck. Restart with new random sequence?
                # If we have time, restart.
                if time.time() < deadline - 10.0:
                    f = np.random.choice([-1, 1], N)
                    R = compute_autocorr(f)
                    current_max_abs = np.max(np.abs(R))
                    if current_max_abs < best_max_abs:
                        best_max_abs = current_max_abs
                        best_f = f.copy()
                        best_R = R.copy()
                    # Continue loop
                else:
                    break
        else:
            # Apply the best flip found
            f = best_f
            R = best_R
            current_max_abs = best_max_abs
            
    return best_f
