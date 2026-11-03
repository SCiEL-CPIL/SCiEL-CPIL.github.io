import numpy as np
import time

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    # Initialize with random sequence
    rng = np.random.default_rng(42)
    s = rng.choice([-1.0, 1.0], size=N)
    
    # Function to compute full autocorrelation (aperiodic)
    def compute_acf(seq):
        n_fft = 1 << (2 * N - 1).bit_length()
        S = np.fft.rfft(seq, n=n_fft)
        acf = np.fft.irfft(S * np.conj(S), n=n_fft)[:N]
        return acf

    # Initial ACF and PSL
    acf = compute_acf(s)
    # PSL is max of |acf[1:]|
    current_psl = np.max(np.abs(acf[1:]))
    best_psl = current_psl
    best_s = s.copy()
    
    # Simulated Annealing parameters
    T = 10.0
    cooling_rate = 0.999
    min_T = 0.1
    
    # Precompute indices for delta update efficiency
    # We need to check which k values are affected by flipping index i
    # k ranges from 1 to N-1
    
    steps = 0
    while time.time() < deadline:
        # Propose a random flip
        i = rng.integers(0, N)
        
        # Calculate change in ACF
        # Delta R(k) = -2 * s[i] * (s[i+k] + s[i-k])
        # Valid k: 1 <= k < N
        # s[i+k] valid if i+k < N
        # s[i-k] valid if i-k >= 0
        
        # We only need to update the ACF values that change
        # And then check if the global max changed
        
        # Optimization: 
        # 1. Identify affected lags k
        # 2. Update acf[k] for those lags
        # 3. Check if new acf[k] exceeds current max, or if old max was at an affected lag
        
        # To keep it simple and fast, we can just recompute the max if we suspect improvement,
        # or maintain a max-heap / sorted list? 
        # For N=400, recomputing max is O(N).
        # Updating ACF is O(N).
        # Total O(N) per step.
        
        # Let's do the delta update on a copy of the relevant parts?
        # No, we update the acf array in place, and revert if rejected.
        
        old_s_i = s[i]
        s[i] = -old_s_i
        
        # Update ACF
        # k such that i+k < N => k < N-i
        # k such that i-k >= 0 => k <= i
        # So k in [1, min(i, N-i-1)]? 
        # Actually, R(k) is defined for k=1..N-1.
        # Term 1: s[i] s[i+k]. Exists if i+k < N.
        # Term 2: s[i-k] s[i]. Exists if i-k >= 0.
        
        # Let's just loop k from 1 to N-1 and update if applicable.
        # This is O(N).
        
        # To speed up, we can precompute the "neighbor" sums?
        # No, s changes.
        
        # Efficient Loop:
        # max_k1 = N - 1 - i
        # max_k2 = i
        # k must be <= max_k1 for first term, <= max_k2 for second term.
        
        # We update acf[k] for all k in 1..N-1 where either term is valid.
        
        # Let's just do a vectorized update if possible, or a tight python loop?
        # Python loop for N=400 is fast (~10us).
        
        # Save old values to revert
        old_acf_vals = []
        old_k_indices = []
        
        for k in range(1, N):
            delta = 0.0
            if i + k < N:
                delta += old_s_i * s[i+k] # Wait, s[i] is already flipped.
                # Original term: s[i]_old * s[i+k]
                # New term: s[i]_new * s[i+k] = -s[i]_old * s[i+k]
                # Change: -2 * s[i]_old * s[i+k]
                # My formula: -2 * s[i]_old * (s[i+k] + s[i-k])
                # So I should use old_s_i.
                
            if i - k >= 0:
                delta += old_s_i * s[i-k]
                
            if delta != 0.0:
                old_val = acf[k]
                new_val = old_val - 2.0 * delta # Wait.
                # Change in R(k) is -2 * s[i]_old * (s[i+k] + s[i-k])
                # So new_acf[k] = old_acf[k] - 2 * s[i]_old * (s[i+k] + s[i-k])
                
                # Let's re-verify the sign.
                # R(k) = sum s_j s_{j+k}
                # Flip i.
                # If j=i, term is s_i s_{i+k}. Becomes -s_i s_{i+k}.
                # Diff: -2 s_i s_{i+k}.
                # If j+k=i, term is s_{i-k} s_i. Becomes -s_{i-k} s_i.
                # Diff: -2 s_{i-k} s_i.
                # Total Diff: -2 s_i (s_{i+k} + s_{i-k}).
                
                acf[k] = old_val - 2.0 * old_s_i * ( (s[i+k] if i+k<N else 0) + (s[i-k] if i-k>=0 else 0) )
                old_acf_vals.append(old_val)
                old_k_indices.append(k)
                
        # Calculate new PSL
        # We only need to check if the max changed.
        # The max can only decrease if the previous max was at an updated index and decreased.
        # The max can only increase if an updated index increased above the current max.
        
        # Simple approach: Recompute max of acf[1:]
        # This is O(N).
        new_psl = np.max(np.abs(acf[1:]))
        
        # Acceptance
        if new_psl < current_psl:
            current_psl = new_psl
            if new_psl < best_psl:
                best_psl = new_psl
                best_s = s.copy()
        else:
            delta_psl = new_psl - current_psl
            if delta_psl < 0 or (T > min_T and rng.random() < np.exp(-delta_psl / T)):
                current_psl = new_psl
            else:
                # Revert
                s[i] = old_s_i
                for k, old_val in zip(old_k_indices, old_acf_vals):
                    acf[k] = old_val
                # current_psl remains unchanged (it was the max of the old acf)
                # Note: if we reverted, the acf is back to original, so current_psl is still valid.
                
        T *= cooling_rate
        steps += 1
        
    return best_s
